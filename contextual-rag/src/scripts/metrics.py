from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.bleu_score import sentence_bleu #, corpus_bleu
from rouge_score import rouge_scorer
from bert_score import BERTScorer
from dotenv import load_dotenv
import numpy as np
import requests
import json
import os

load_dotenv(dotenv_path="config/config.env")

class Metrics:
    def __init__(self,
                llm_model = os.getenv("LLM_MODEL"),
                tabby_api_url = "http://localhost:5001/v1",
                tabby_api_key = os.getenv("TABBY_API_KEY")):

        self.llm_model = llm_model
        self.tabby_api_url = tabby_api_url
        self.tabby_api_key = tabby_api_key

        self.all_scores = {} 
        self.all_scores['precision_at_k'] = []   
        self.all_scores['rouge-1'] = []   
        self.all_scores['rouge-2'] = []
        self.all_scores['rouge-l'] = []
        self.all_scores['BERTScore_p'] = []
        self.all_scores['BERTScore_r'] = []
        self.all_scores['BERTScore_f1'] = []
        self.all_scores['bleu'] = []
        self.all_scores['exact_match'] = []
        self.all_scores['LLM_as_a_judge'] = [] 
    
    def evaluate(self, item, retrieved_chunks_list, ground_truth_context, top_k):        
        """
        Evaluate the retrieval system using Precision@k for retrieval, and BLEU, ROUGE, BERTScore, LLM-as-a-judge for generation.
        Returns: A dictionary of evaluation metrics.
        """
        ### Retrieval performance (binary precision@k): Evaluate relevance of retrieved chunks
        
        ## On SQuAD
        relevant_chunks = [item['ground_truth'] in chunk for chunk in retrieved_chunks_list]
        precision_at_k = (1 if any(relevant_chunks) else 0)

        ## On Australian Legal QA Dataset
        # relevant_chunks = [ground_truth_context in doc for doc in retrieved_chunks_list]
        # precision_at_k = (1 if any(relevant_chunks) else 0)


        ### Generation

        # Calculate BLEU score 
        smoothing_function = SmoothingFunction().method2
        bleu = sentence_bleu(references=[item['ground_truth']], hypothesis=item['prediction'], smoothing_function=smoothing_function) 

        # Calculate ROUGE score
        # ROUGE-1 scores are excellent around 0.5, with scores above 0.5 considered good and 0.4 to 0.5 moderate. For ROUGE-2, scores above 0.4 are good, and 0.2 to 0.4 are moderate. ROUGE-L scores are good around 0.4 and low at 0.3 to 0.4.  
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)     # Initialize ROUGE evaluator
        scores = scorer.score(item['ground_truth'], item['prediction'])
        rouge_f1_scores = {rouge_type: score.fmeasure for rouge_type, score in scores.items()}

        # Calculate BERTScore
        scorer = BERTScorer(lang="en")      
        bert_p, bert_r, bert_f1 = scorer.score([item['prediction']], [item['ground_truth']])

        # Calculate LLM-as-a-judge (use an LLM as a judge for generation evaluation)
        llm_judge_score, _ = self.llm_as_a_judge(item)

        # Calculate exact matching score
        exact_match_score = self.exact_match(item['prediction'], item['ground_truth'])

        # print("Answer Found Rate (ARF): ", (answer_found / total_queries) * 100)

        ### Aggregate and print results
        scores = {
            "precision_at_k": precision_at_k,            
            "bleu": bleu,
            "rouge-l": rouge_f1_scores['rougeL'],
            "rouge-1": rouge_f1_scores['rouge1'],
            "rouge-2": rouge_f1_scores['rouge2'],
            "bert_score_p": bert_p.item(),                 # Precision scores (token similarity considering generated tokens)
            "bert_score_r": bert_r.item(),                 # Recall scores (token similarity considering reference tokens)
            "bert_score_f1": bert_f1.item(),               # F1 scores (harmonic mean of Precision and Recall)
            "llm_as_a_judge": llm_judge_score,
            "exact_match": exact_match_score
        }

        print("Retrieval:")
        print(f'\t Precision@{top_k} (Retrieval): {scores["precision_at_k"]}') 
        self.all_scores['precision_at_k'].append(float(f"{scores['precision_at_k']}"))

        print("Generation:")
        print(f'\t ROUGE-1 F1: {scores["rouge-1"]:.4f}') 
        self.all_scores['rouge-1'].append(float(f"{scores['rouge-1']:.4f}"))

        print(f'\t ROUGE-2 F1: {scores["rouge-2"]:.4f}') 
        self.all_scores['rouge-2'].append(float(f"{scores['rouge-2']:.4f}"))

        print(f'\t ROUGE-L F1: {scores["rouge-l"]:.4f}') 
        self.all_scores['rouge-l'].append(float(f"{scores['rouge-l']:.4f}"))

        print(f'\t BLEU: {scores["bleu"]:.4f}') 
        self.all_scores['bleu'].append(float(f"{scores['bleu']:.4f}"))

        print(f'\t BERTScore Precision: {scores["bert_score_p"]:.4f}')
        self.all_scores['BERTScore_p'].append(f"{scores['bert_score_p']:.4f}")

        print(f'\t BERTScore Recall: {scores["bert_score_r"]:.4f}')
        self.all_scores['BERTScore_r'].append(scores['bert_score_r'])

        print(f'\t BERTScore F1: {scores["bert_score_f1"]:.4f}')
        self.all_scores['BERTScore_f1'].append(f"{scores['bert_score_f1']:.4f}")

        print(f'\t Exact Match: {scores["exact_match"]}') 
        self.all_scores['exact_match'].append(float(scores["exact_match"]))

        print(f'\t LLM as a judge: {scores["llm_as_a_judge"]}')
        self.all_scores['LLM_as_a_judge'].append(float(scores["llm_as_a_judge"]))

        return self.all_scores
    
    def get_average_results(self, all_scores):
        """
            Calculate and print average results for the validation set.
        """
        print("Generation: ")
        print("\tROUGE-1 F1: {:.4f}".format(np.mean(all_scores['rouge-1'])))
        print("\tROUGE-2 F1: {:.4f}".format(np.mean(all_scores['rouge-2'])))
        print("\tROUGE-L F1: {:.4f}".format(np.mean(all_scores['rouge-l'])))
        print(f"\tBERTScore Precision: {np.mean(list(map(float, all_scores['BERTScore_p']))):.4f}")
        print(f"\tBERTScore Recall: {np.mean(list(map(float, all_scores['BERTScore_r']))):.4f}")
        print(f"\tBERTScore F1: {np.mean(list(map(float, all_scores['BERTScore_f1']))):.4f}")
        print("\tBLEU: {:.4f}".format(np.mean(all_scores['bleu'])))
        print("\tExact Matching: ", np.mean(all_scores['exact_match']))
        print("\tLLM as a judge: {:.4f}".format(np.mean(all_scores['LLM_as_a_judge'])))

    def exact_match(self, predicted, ground_truth):
        """
        Calculate Exact Match (EM) between the predicted and the ground-truth.
        Returns 1 if they match exactly, otherwise 0.
        """
        return 1 if predicted == ground_truth else 0

    def llm_as_a_judge(self, item):
        """
        Use LLM to evaluate the generated answer compared to the ground-truth.
        """
        JUDGE_PROMPT = f"""You are an expert judge evaluating question-answering systems. Compare the predicted answer with the ground truth answer and rate the prediction's accuracy on a scale of 0 to 1, where:
        1.0: Perfect match or semantically equivalent
        0.7-0.9: Mostly correct with minor differences
        0.4-0.6: Partially correct
        0.1-0.3: Mostly incorrect but has some relevant information
        0.0: Completely incorrect or irrelevant

        Question: {item['question']}
        Ground Truth: {item['ground_truth']}
        Predicted Answer: {item['prediction']}

        Respond with only the numerical score. Do not preamble.
        """

        url = f"{self.tabby_api_url}/chat/completions"
        headers = {
        "Content-Type": "application/json",
        "Authorization": f'Bearer {self.tabby_api_key}'
        }

        payload = json.dumps({
        "model": self.llm_model,
        "max_tokens": 3,
        "stream": "False",
        "min_p": 0.05,
        "messages": [
                {
                    "role": "user", 
                    "content": JUDGE_PROMPT
                }
            ],
        "repetition_penalty": 1.05
        })
        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        return response['choices'][0]['message']['content'], response['usage']