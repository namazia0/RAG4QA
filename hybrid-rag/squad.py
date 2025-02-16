from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import logging as transformers_logging
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from bert_score import score as bert_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import numpy as np
import statistics
import datasets
import requests
import evaluate 
import warnings
import argparse
import logging
import torch
import nltk
import time

nltk.download('punkt')

# Suppress warnings
warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

'''
1- Dense Retrieval (Semantic Search)
2- Sparse Retrieval (BM25)
3- Hybrid Scoring

WHY:

Bi-Encoder:
Query  →  Encoder  →  Embedding1     
                               → Similarity Score
Context →  Encoder  →  Embedding2    

Cross-Encoder:
[Query, Context] → Encoder → Relevance Score

Improvements for Enhanced Exact Matching:
- Use more robust transformer models for better semantic understanding.
- Implement batch processing to speed up embedding and reranking.
- Fine-tune models specifically on the SQuAD dataset.
- Optimize retrieval scoring and normalization.
- Ensure answer extraction strictly adheres to context boundaries.
'''

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQuADRAG:
    def __init__(self,
                 embedding_model="sentence-transformers/all-MiniLM-L12-v2",
                 rerank_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
                 tabby_api_url="http://localhost:5005/v1",
                 device="cuda"):
        # Add at the beginning of __init__
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your GPU setup.")
        
        # Load SQuAD dataset
        self.dataset = datasets.load_dataset("squad")
        
        # Initialize embedding model on second GPU
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_model.to(device)
        
        # Initialize reranker on second GPU
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model)
        self.rerank_model.to(device)
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model)
        
        # Store device
        self.device = device
        
        # Store context embeddings
        self.context_embeddings = None
        self.contexts = None
        
        # Add BM25 index
        self.bm25 = None
        self.tokenized_contexts = None
        
        # Add TabbyAPI URL
        self.tabby_api_url = tabby_api_url

    def prepare_database(self, num_samples=10000, batch_size=2048):
        """Prepare the context database with embeddings and BM25 index."""
        logger.info("Preparing database...")
        # Sample from validation set with increased samples for better coverage
        train_data = self.dataset['validation'].shuffle(seed=42).select(range(num_samples))
        
        # Get unique contexts
        self.contexts = list(set(train_data['context']))
        
        # Generate embeddings for contexts in batches
        embeddings = []
        for i in tqdm(range(0, len(self.contexts), batch_size), desc="Generating Context Embeddings"):
            batch = self.contexts[i:i+batch_size]
            batch_embeddings = self.embedding_model.encode(batch, batch_size=batch_size, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        self.context_embeddings = np.vstack(embeddings)
        
        # Normalize embeddings for cosine similarity
        self.context_embeddings = self.context_embeddings / np.linalg.norm(self.context_embeddings, axis=1, keepdims=True)
        
        # Prepare BM25 index
        self.tokenized_contexts = [word_tokenize(context.lower()) for context in self.contexts]
        self.bm25 = BM25Okapi(self.tokenized_contexts)
        logger.info("Database preparation complete.")
        
    def retrieve_relevant_contexts(self, query, k=10):
        """Initial retrieval phase using hybrid search"""
        # Dense retrieval scores
        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        dense_scores = cosine_similarity(query_embedding, self.context_embeddings)[0]
        
        # Sparse retrieval scores
        tokenized_query = word_tokenize(query.lower())
        sparse_scores = np.array(self.bm25.get_scores(tokenized_query))
        
        # Normalize scores to 0-1
        dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-8)
        sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-8)
        
        # Combine scores with weighted average
        combined_scores = 0.7 * dense_scores + 0.3 * sparse_scores  # Adjust weights as needed
        
        # Get top k contexts
        top_k_indices = np.argsort(combined_scores)[-k:][::-1]
        
        return [(self.contexts[i], combined_scores[i]) for i in top_k_indices]
    
    def rerank_contexts(self, query, contexts, k=5, batch_size=128):
        """Rerank the retrieved contexts using cross-encoder"""
        pairs = [[query, ctx[0]] for ctx in contexts]
        scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            features = self.rerank_tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            # Move input tensors to specified GPU
            features = {k: v.to(self.device) for k, v in features.items()}
            
            with torch.no_grad():
                outputs = self.rerank_model(**features)
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
                scores.extend(batch_scores)
        
        # Attach scores to contexts and maintain the (context, score) tuple format
        contexts_with_scores = [(ctx[0], score) for ctx, score in zip(contexts, scores)]
        
        # Sort by reranker score
        contexts_with_scores = sorted(contexts_with_scores, key=lambda x: x[1], reverse=True)
        
        return contexts_with_scores[:k]  # Returns list of (context, score) tuples
    
    def answer_question(self, question, k=3):
        """Answer a question using the RAG pipeline with TabbyAPI"""
        if self.context_embeddings is None:
            raise ValueError("Please run prepare_database() first!")
        
        # 1. Retrieve initial candidate contexts
        retrieved_contexts = self.retrieve_relevant_contexts(question, k=10)
        
        # 2. Rerank contexts
        reranked_contexts = self.rerank_contexts(question, retrieved_contexts, k=k)
        
        # 3. Generate answer using TabbyAPI
        best_answer = None
        best_score = float('-inf')
        
        for context, similarity_score in reranked_contexts:
            # Create improved prompt for TabbyAPI
            prompt = f"""Answer the question in minimum words possible using only the given context.
    
    Context: {context}
    Question: {question}
    
    Answer:"""
    
            # Call TabbyAPI
            response = requests.post(
                f"{self.tabby_api_url}/completions",
                json={
                    "prompt": prompt,
                    "max_tokens": 100,
                    "temperature": 0.1,
                    "n": 1,
                    "stop": ["\n", "Question:", "Context:"]
                }
            )
            
            if response.status_code == 200:
                answer_data = response.json()
                answer = answer_data['choices'][0]['text'].strip()
                
                # Use the model's score from the API response
                qa_score = answer_data['choices'][0].get('score', 0)
                
                # Combine scores
                combined_score = qa_score * similarity_score
                
                if combined_score > best_score and answer.strip():
                    best_score = combined_score
                    best_answer = {
                        'answer': answer,
                        'context': context,
                        'similarity': similarity_score,
                        'combined_score': combined_score
                    }
        
        return best_answer
    
    def answer_question_no_context(self, question):
        """Answer a question using only TabbyAPI without any context"""
        # Create prompt without context
        prompt = f"""Answer the question in minimum words possible.

    Question: {question}

    Answer:"""

        # Call TabbyAPI
        response = requests.post(
            f"{self.tabby_api_url}/completions",
            json={
                "prompt": prompt,
                "max_tokens": 100,
                "temperature": 0.1,
                "n": 1,
                "stop": ["\n", "Question:"]
            }
        )
        
        if response.status_code == 200:
            answer_data = response.json()
            answer = answer_data['choices'][0]['text'].strip()
            
            return {
                'answer': answer,
                'context': None,
                'similarity': 0,
                'combined_score': 0
            }
        return None
    
    def evaluate_predictions(self, num_samples=50, verbose=True, llm_batch_size=50):
        """
        Evaluate the RAG system using multiple metrics.
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        logger.info("Starting evaluation...")
        # Initialize metrics
        rouge_calculator = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        exact_match_metric = evaluate.load("exact_match")
        smoothie = SmoothingFunction().method1
        
        # Initialize results storage
        results = {
            'rouge1_precision': [],
            'rouge1_recall': [],
            'rouge1_f1': [],
            'rouge2_precision': [],
            'rouge2_recall': [],
            'rouge2_f1': [],
            'rougeL_precision': [],
            'rougeL_recall': [],
            'rougeL_f1': [],
            'bert_precision': [],
            'bert_recall': [],
            'bert_f1': [],
            'bleu_scores': [],
            'exact_match': [],
            'retrieval_precision': [],
            'llm_judge_score': []
        }
        
        # Get validation samples
        validation_set = self.dataset['validation'].shuffle(seed=42).select(range(num_samples))
        
        # To store LLM judge input
        llm_inputs = []
        llm_indices = []
        
        for idx, example in enumerate(tqdm(validation_set, desc="Evaluating")):
            question = example['question']
            ground_truth = example['answers']['text'][0]
            
            # Get prediction
            prediction = self.answer_question(question)
            
            if prediction and prediction['answer']:
                # Calculate ROUGE scores without printing
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rouge_scores = rouge_calculator.score(ground_truth, prediction['answer'])
                results['rouge1_precision'].append(rouge_scores['rouge1'].precision)
                results['rouge1_recall'].append(rouge_scores['rouge1'].recall)
                results['rouge1_f1'].append(rouge_scores['rouge1'].fmeasure)
                results['rouge2_precision'].append(rouge_scores['rouge2'].precision)
                results['rouge2_recall'].append(rouge_scores['rouge2'].recall)
                results['rouge2_f1'].append(rouge_scores['rouge2'].fmeasure)
                results['rougeL_precision'].append(rouge_scores['rougeL'].precision)
                results['rougeL_recall'].append(rouge_scores['rougeL'].recall)
                results['rougeL_f1'].append(rouge_scores['rougeL'].fmeasure)
                
                # Calculate BERTScore
                P, R, F1 = bert_score(
                    [prediction['answer']], 
                    [ground_truth], 
                    lang='en',
                    verbose=False
                )
                results['bert_precision'].append(P[0].item())
                results['bert_recall'].append(R[0].item())
                results['bert_f1'].append(F1[0].item())
                
                # Calculate BLEU score with smoothing
                reference = word_tokenize(ground_truth.lower())
                candidate = word_tokenize(prediction['answer'].lower())
                try:
                    weights = (0.5, 0.5)  # Equal weights for unigrams and bigrams
                    bleu = sentence_bleu(
                        [reference], 
                        candidate,
                        weights=weights,
                        smoothing_function=smoothie
                    )
                    results['bleu_scores'].append(bleu)
                except:
                    pass
                
                # Calculate Exact Match
                em_score = exact_match_metric.compute(
                    predictions=[prediction['answer'].lower().strip()],
                    references=[ground_truth.lower().strip()]
                )['exact_match']
                results['exact_match'].append(em_score)
                
                # Modified retrieval precision calculation
                if prediction['context'] is not None and example['context'] in prediction['context']:
                    results['retrieval_precision'].append(1)
                else:
                    results['retrieval_precision'].append(0)
                
                # Prepare data for LLM judge
                llm_inputs.append({
                    'ground_truth': ground_truth,
                    'prediction': prediction['answer'],
                    'context': prediction['context'],
                    'question': question
                })
                llm_indices.append(idx)
            else:
                # Handle cases where prediction is None or empty
                results['rouge1_precision'].append(0)
                results['rouge1_recall'].append(0)
                results['rouge1_f1'].append(0)
                results['rouge2_precision'].append(0)
                results['rouge2_recall'].append(0)
                results['rouge2_f1'].append(0)
                results['rougeL_precision'].append(0)
                results['rougeL_recall'].append(0)
                results['rougeL_f1'].append(0)
                results['bert_precision'].append(0)
                results['bert_recall'].append(0)
                results['bert_f1'].append(0)
                results['bleu_scores'].append(0)
                results['exact_match'].append(0)
                results['retrieval_precision'].append(0)
                results['llm_judge_score'].append(0)
        
        # Batch process LLM judge scores
        logger.info("Calculating LLM judge scores...")
        for i in tqdm(range(0, len(llm_inputs), llm_batch_size), desc="LLM Judge Scoring"):
            batch = llm_inputs[i:i+llm_batch_size]
            batch_scores = []
            
            for item in batch:
                # Create prompt for TabbyAPI to judge answer quality
                judge_prompt = f"""You are an expert judge evaluating question-answering systems. Compare the predicted answer with the ground truth answer and rate the prediction's accuracy on a scale of 0 to 1, where:
- 1.0: Perfect match or semantically equivalent
- 0.7-0.9: Mostly correct with minor differences
- 0.4-0.6: Partially correct
- 0.1-0.3: Mostly incorrect but has some relevant information
- 0.0: Completely incorrect or irrelevant

Question: {item['question']}
Ground Truth: {item['ground_truth']}
Predicted Answer: {item['prediction']}

Rate the accuracy (respond with only the numerical score):"""

                try:
                    # Call TabbyAPI for judgment
                    response = requests.post(
                        f"{self.tabby_api_url}/completions",
                        json={
                            "prompt": judge_prompt,
                            "max_tokens": 10,
                            "temperature": 0.1,
                            "n": 1,
                            "stop": ["\n"]
                        }
                    )
                    
                    if response.status_code == 200:
                        score_text = response.json()['choices'][0]['text'].strip()
                        try:
                            # Extract just the numerical value by taking the first word
                            # and removing any trailing punctuation
                            score_text = score_text.split()[0].rstrip('.')
                            score = float(score_text)
                            # Ensure score is between 0 and 1
                            score = max(0.0, min(1.0, score))
                            batch_scores.append(score)
                        except (ValueError, IndexError):
                            logger.warning(f"Invalid score format: {score_text}")
                            batch_scores.append(0.0)
                    else:
                        logger.warning(f"TabbyAPI error: {response.status_code}")
                        batch_scores.append(0.0)
                        
                except Exception as e:
                    logger.error(f"Error during LLM judging: {e}")
                    batch_scores.append(0.0)
            
            results['llm_judge_score'].extend(batch_scores)
        
        # Calculate final metrics
        final_metrics = {
            'rouge1_precision': statistics.mean(results['rouge1_precision']) if results['rouge1_precision'] else 0,
            'rouge1_recall': statistics.mean(results['rouge1_recall']) if results['rouge1_recall'] else 0,
            'rouge1_f1': statistics.mean(results['rouge1_f1']) if results['rouge1_f1'] else 0,
            'rouge2_precision': statistics.mean(results['rouge2_precision']) if results['rouge2_precision'] else 0,
            'rouge2_recall': statistics.mean(results['rouge2_recall']) if results['rouge2_recall'] else 0,
            'rouge2_f1': statistics.mean(results['rouge2_f1']) if results['rouge2_f1'] else 0,
            'rougeL_precision': statistics.mean(results['rougeL_precision']) if results['rougeL_precision'] else 0,
            'rougeL_recall': statistics.mean(results['rougeL_recall']) if results['rougeL_recall'] else 0,
            'rougeL_f1': statistics.mean(results['rougeL_f1']) if results['rougeL_f1'] else 0,
            'bert_precision': statistics.mean(results['bert_precision']) if results['bert_precision'] else 0,
            'bert_recall': statistics.mean(results['bert_recall']) if results['bert_recall'] else 0,
            'bert_f1': statistics.mean(results['bert_f1']) if results['bert_f1'] else 0,
            'bleu_score': statistics.mean(results['bleu_scores']) if results['bleu_scores'] else 0,
            'exact_match': statistics.mean(results['exact_match']) if results['exact_match'] else 0,
            'retrieval_precision': statistics.mean(results['retrieval_precision']) if results['retrieval_precision'] else 0,
            'llm_judge_score': statistics.mean(results['llm_judge_score']) if results['llm_judge_score'] else 0
        }
        
        if verbose:
            print("\nEvaluation Results:")
            print("\nROUGE Metrics:")
            print("ROUGE-1:")
            print(f"  Precision: {final_metrics['rouge1_precision']:.4f}")
            print(f"  Recall: {final_metrics['rouge1_recall']:.4f}")
            print(f"  F1: {final_metrics['rouge1_f1']:.4f}")
            
            print("\nROUGE-2:")
            print(f"  Precision: {final_metrics['rouge2_precision']:.4f}")
            print(f"  Recall: {final_metrics['rouge2_recall']:.4f}")
            print(f"  F1: {final_metrics['rouge2_f1']:.4f}")
            
            print("\nROUGE-L:")
            print(f"  Precision: {final_metrics['rougeL_precision']:.4f}")
            print(f"  Recall: {final_metrics['rougeL_recall']:.4f}")
            print(f"  F1: {final_metrics['rougeL_f1']:.4f}")
            
            print("\nOther Metrics:")
            print(f"BERTScore F1: {final_metrics['bert_f1']:.4f}")
            print(f"BLEU Score: {final_metrics['bleu_score']:.4f}")
            print(f"Exact Match: {final_metrics['exact_match']:.4f}")
            print(f"Retrieval Precision: {final_metrics['retrieval_precision']:.4f}")
            print(f"LLM Judge Score: {final_metrics['llm_judge_score']:.4f}")
        
        return final_metrics

    def evaluate_ablation(self, num_samples=50, verbose=True):
        """
        Run ablation study comparing performance with and without context
        """
        logger.info("Starting ablation study...")
        
        # Initialize metrics for both approaches
        metrics_with_context = self.evaluate_predictions(num_samples=num_samples, verbose=False)
        
        # Store original answer_question method
        original_answer_question = self.answer_question
        
        # Replace with no-context version
        self.answer_question = self.answer_question_no_context
        
        # Evaluate without context
        metrics_no_context = self.evaluate_predictions(num_samples=num_samples, verbose=False)
        
        # Restore original method
        self.answer_question = original_answer_question
        
        if verbose:
            print("\nAblation Study Results:")
            print("\nWith Context vs No Context:")
            metrics_to_compare = [
                'rouge1_f1', 'rouge2_f1', 'rougeL_f1', 
                'bert_f1', 'bleu_score', 'exact_match', 
                'llm_judge_score'
            ]
            
            print("\n{:<20} {:<15} {:<15}".format("Metric", "With Context", "No Context"))
            print("-" * 50)
            for metric in metrics_to_compare:
                print("{:<20} {:<15.4f} {:<15.4f}".format(
                    metric,
                    metrics_with_context[metric],
                    metrics_no_context[metric]
                ))
        
        return {
            'with_context': metrics_with_context,
            'no_context': metrics_no_context
        }

    def run_llm_ablation(self, num_samples=50, verbose=True):
        """Run ablation study comparing different LLM models with fixed embedding/reranking models"""
        logger.info("Starting LLM ablation study...")
        
        # Define LLM configurations to test
        llm_configs = {
            'llama-1b': 'Llama-3.2-1B',
            'llama-3b': 'Llama-3.2-3B-Instruct-exl2-6_5',
            'llama-8b': 'llama3.1-8b'
        }
        
        results = {}
        original_url = self.tabby_api_url
        
        for llm_name, model_name in llm_configs.items():
            logger.info(f"\nTesting LLM: {llm_name}")
            
            # First load the model via TabbyAPI
            try:
                load_response = requests.post(
                    f"{original_url}/model/load",
                    json={"model": model_name}
                )
                
                if load_response.status_code != 200:
                    logger.error(f"Failed to load model {model_name}: {load_response.text}")
                    continue
                    
                # Wait for model to load
                while True:
                    status_response = requests.get(f"{original_url}/model/status")
                    if status_response.status_code == 200:
                        status = status_response.json()
                        if status.get('loaded'):
                            logger.info(f"Model {model_name} loaded successfully")
                            break
                        elif status.get('error'):
                            logger.error(f"Error loading model: {status['error']}")
                            break
                    time.sleep(2)  # Wait 2 seconds before checking again
                
                # Update TabbyAPI URL to use loaded model
                self.tabby_api_url = f"{original_url}?model={model_name}"
                
                # Test with context
                logger.info(f"Evaluating {llm_name} with context...")
                metrics_with_context = self.evaluate_predictions(num_samples=num_samples, verbose=False)
                
                # Test without context
                logger.info(f"Evaluating {llm_name} without context...")
                original_answer = self.answer_question
                self.answer_question = self.answer_question_no_context
                metrics_no_context = self.evaluate_predictions(num_samples=num_samples, verbose=False)
                self.answer_question = original_answer
                
                results[llm_name] = {
                    'with_context': metrics_with_context,
                    'no_context': metrics_no_context
                }
                
                # Unload the model after testing
                requests.post(f"{original_url}/model/unload")
                
            except Exception as e:
                logger.error(f"Error evaluating {llm_name}: {str(e)}")
                results[llm_name] = None
        
        # Restore original URL
        self.tabby_api_url = original_url
        
        if verbose:
            print("\nLLM Ablation Study Results")
            print("=========================")
            
            headers = ['LLM Model', 'Context', 'ROUGE-L', 'BERT-F1', 'BLEU', 'LLM Score']
            print("\n{:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(*headers))
            print("-" * 65)
            
            for llm_name, llm_results in results.items():
                if llm_results:
                    for context_type in ['with_context', 'no_context']:
                        metrics = llm_results[context_type]
                        print("{:<15} {:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                            llm_name,
                            context_type.replace('_', ' '),
                            metrics['rougeL_f1'],
                            metrics['bert_f1'],
                            metrics['bleu_score'],
                            metrics['llm_judge_score']
                        ))
        return results

def list_gpu_devices():
    """List all available GPU devices and their memory usage."""
    logger.info("\nAvailable GPU Devices:")
    if not torch.cuda.is_available():
        logger.info("No CUDA devices available")
        return

    for i in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(i)
        total_memory = gpu.total_memory / 1024**2  # Convert to MB
        allocated_memory = torch.cuda.memory_allocated(i) / 1024**2
        cached_memory = torch.cuda.memory_reserved(i) / 1024**2
        
        logger.info(f"\nGPU {i}: {gpu.name}")
        logger.info(f"- Total Memory: {total_memory:.2f} MB")
        logger.info(f"- Allocated Memory: {allocated_memory:.2f} MB")
        logger.info(f"- Cached Memory: {cached_memory:.2f} MB")
        logger.info(f"- Free Memory: {total_memory - allocated_memory:.2f} MB")

def run_model_ablation():
    """Run comprehensive ablation study comparing different model combinations with/without context"""
    logger.info("Starting comprehensive ablation study...")
    
    # Define model combinations to test
    embedding_models = {
        'base': 'sentence-transformers/all-MiniLM-L12-v2',
        'mpnet': 'sentence-transformers/all-mpnet-base-v2',
        'multi_qa': 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
    }
    
    reranker_models = {
        'minilm': 'cross-encoder/ms-marco-MiniLM-L-12-v2',
        'roberta': 'cross-encoder/qnli-distilroberta-base'
    }
    
    results = {}
    
    # Test each combination
    for emb_name, emb_model in embedding_models.items():
        for rerank_name, rerank_model in reranker_models.items():
            combo_name = f"{emb_name}_{rerank_name}"
            logger.info(f"\nTesting combination: {combo_name}")
            
            try:
                # Initialize RAG with current model combination
                rag = SQuADRAG(
                    embedding_model=emb_model,
                    rerank_model=rerank_model
                )
                
                # Prepare database
                rag.prepare_database(num_samples=10000)
                
                # Test with context
                logger.info(f"Evaluating {combo_name} with context...")
                metrics_with_context = rag.evaluate_predictions(num_samples=100, verbose=False)
                
                # Test without context
                logger.info(f"Evaluating {combo_name} without context...")
                original_answer = rag.answer_question
                rag.answer_question = rag.answer_question_no_context
                metrics_no_context = rag.evaluate_predictions(num_samples=100, verbose=False)
                rag.answer_question = original_answer
                
                results[combo_name] = {
                    'with_context': metrics_with_context,
                    'no_context': metrics_no_context
                }
                
                # Free GPU memory
                del rag
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error evaluating {combo_name}: {str(e)}")
                results[combo_name] = None
    
    # Save detailed results
    with open('ablation_study_results.txt', 'w') as f:
        f.write("Comprehensive Ablation Study Results\n")
        f.write("===================================\n\n")
        
        for combo_name, combo_results in results.items():
            f.write(f"\nModel Combination: {combo_name}\n")
            f.write("=" * (len(combo_name) + 19) + "\n\n")
            
            if combo_results:
                for context_type in ['with_context', 'no_context']:
                    f.write(f"\n{context_type.replace('_', ' ').title()}:\n")
                    f.write("-" * (len(context_type) + 1) + "\n")
                    metrics = combo_results[context_type]
                    
                    # Group metrics by category
                    categories = {
                        'ROUGE Scores': ['rouge1_f1', 'rouge2_f1', 'rougeL_f1',
                                       'rouge1_precision', 'rouge2_precision', 'rougeL_precision',
                                       'rouge1_recall', 'rouge2_recall', 'rougeL_recall'],
                        'BERT Scores': ['bert_precision', 'bert_recall', 'bert_f1'],
                        'Other Metrics': ['bleu_score', 'exact_match', 'retrieval_precision', 'llm_judge_score']
                    }
                    
                    for category, metric_list in categories.items():
                        f.write(f"\n{category}:\n")
                        for metric in metric_list:
                            if metric in metrics:
                                f.write(f"{metric:>20}: {metrics[metric]:.4f}\n")
            else:
                f.write("Failed to evaluate\n")
    
    # Print summary table
    print("\nAblation Study Summary")
    print("=====================")
    
    headers = ['Model Combo', 'Context', 'ROUGE-L', 'BERT-F1', 'BLEU', 'LLM Score']
    print("\n{:<20} {:<10} {:<10} {:<10} {:<10} {:<10}".format(*headers))
    print("-" * 70)
    
    for combo_name, combo_results in results.items():
        if combo_results:
            for context_type in ['with_context', 'no_context']:
                metrics = combo_results[context_type]
                print("{:<20} {:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                    combo_name,
                    context_type.replace('_', ' '),
                    metrics['rougeL_f1'],
                    metrics['bert_f1'],
                    metrics['bleu_score'],
                    metrics['llm_judge_score']
                ))
    
    return results

def main():
    # Add argument parser for ablation flag
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm-ablation', action='store_true', help='Run LLM ablation study')
    args = parser.parse_args()
    
    # List GPU devices
    list_gpu_devices()
    
    if args.llm_ablation:
        # Initialize RAG with fixed models
        rag = SQuADRAG(
            embedding_model='sentence-transformers/all-MiniLM-L12-v2',
            rerank_model='cross-encoder/ms-marco-MiniLM-L-12-v2'
        )
        rag.prepare_database(num_samples=10000)
        
        # Run LLM ablation
        ablation_results = rag.run_llm_ablation()
    else:
        # Run comprehensive model ablation
        ablation_results = run_model_ablation()

if __name__ == "__main__":
    main()