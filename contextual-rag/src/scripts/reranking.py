from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import torch
import os

load_dotenv(dotenv_path="config/config.env")

class Reranking:
    def __init__(self, 
                rerank_model = os.getenv("RERANK_MODEL")):

        # Initialize reranker
        self.rerank_model = AutoModelForSequenceClassification.from_pretrained(rerank_model)
        self.rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model)

    def rerank(self, question, retrieved_chunks, retrieved_contexts, top_k):
        """
        Re-rank chunks based on the question using the re-ranking model.

        Inputs:
            question (str): The input query / question.
            retrieved_chunks (list of str): The retrieved chunks to be re-ranked.
            retrieved_contexts (list of str): The corresponding original contexts from the dataset.

        Outputs:
            list of tuple: List of (chunk, context, score), sorted by score in descending order regarding the question.
        """
        scores = []
        batch_size=8
        
        # Combine chunks and contexts for re-ranking
        combined_data = [f"Chunk: {chunk}\nContext: {context}" for chunk, context in zip(retrieved_chunks, retrieved_contexts)]
        
        # Process in batches
        for i in range(0, len(combined_data), batch_size):
            batch_data = combined_data[i: i+batch_size]
            
            # Tokenize the batch
            inputs = self.rerank_tokenizer(
                [question] * len(batch_data),  # Repeat the question for all elements in the batch
                batch_data,
                return_tensors="pt",
                truncation=True,
                max_length=512,  # Adjust max_length if needed
                padding=True
            )
            
            # Move inputs to the appropriate device
            inputs = {key: value.to(self.rerank_model.device) for key, value in inputs.items()}
            
            # Forward pass to get scores
            with torch.no_grad():
                outputs = self.rerank_model(**inputs)
            
            # Extract scores (logits are the relevance scores)
            batch_scores = outputs.logits.squeeze().tolist()
            
            # Handle single-element batch case (ensure batch_scores is a list)
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            
            # Append the results with their corresponding chunks and contexts
            for j, score in enumerate(batch_scores):
                scores.append((retrieved_chunks[i + j], retrieved_contexts[i + j], score))
        
        # Sort by score in descending order
        reranked_chunks = sorted(scores, key=lambda x: x[2], reverse=True)
        
        # Extract the top-k reranked chunks and contexts
        top_reranked_chunks = [c[0] for c in reranked_chunks[:top_k]]
        top_reranked_context = [c[1] for c in reranked_chunks[:top_k]]
        
        return top_reranked_chunks, top_reranked_context

    
    def rerank_alqa_no_decomposition(self, question, retrieved_chunks, retrieved_contexts, retrieved_docs, top_k):
        """
        Re-rank chunks based on the question using the re-ranking model.

        Inputs:
            question (str): The input query / question.
            retrieved_chunks (list of str): The retrieved chunks to be re-ranked.
            retrieved_contexts (list of str): The corresponding original contexts from the dataset.

        Outputs:
            list of tuple: List of (chunk, context, score), sorted by score in descending order regarding the question.
        """
        scores = []
        batch_size=8
        
        # Combine chunks and contexts for re-ranking
        combined_data = [f"Chunk: {chunk}\nContext: {context}" for chunk, context in zip(retrieved_chunks, retrieved_contexts)]
        
        # Process in batches
        for i in range(0, len(combined_data), batch_size):
            batch_data = combined_data[i: i+batch_size]
            
            # Tokenize the batch
            inputs = self.rerank_tokenizer(
                [question] * len(batch_data),  # Repeat the question for all elements in the batch
                batch_data,
                return_tensors="pt",
                truncation=True,
                max_length=512,  # Adjust max_length if needed
                padding=True
            )
            
            # Move inputs to the appropriate device
            inputs = {key: value.to(self.rerank_model.device) for key, value in inputs.items()}
            
            # Forward pass to get scores
            with torch.no_grad():
                outputs = self.rerank_model(**inputs)
            
            # Extract scores (logits are the relevance scores)
            batch_scores = outputs.logits.squeeze().tolist()
            
            # Handle single-element batch case (ensure batch_scores is a list)
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            
            # Append the results with their corresponding chunks and contexts
            for j, score in enumerate(batch_scores):
                scores.append((retrieved_chunks[i + j], retrieved_contexts[i + j], retrieved_docs[i + j], score))
        
        # Sort by score in descending order
        reranked_chunks = sorted(scores, key=lambda x: x[3], reverse=True)
        
        # Extract the top-k reranked chunks and contexts
        top_reranked_chunks = [c[0] for c in reranked_chunks[:top_k]]
        top_reranked_context = [c[1] for c in reranked_chunks[:top_k]]
        top_reranked_docs = [c[2] for c in reranked_chunks[:top_k]]
        
        return top_reranked_chunks, top_reranked_context, top_reranked_docs

    def rerank_autocontext(self, question, retrieved_chunks, retrieved_contexts, retrieved_chunk_titles, top_k):
        """
        Re-rank chunks based on the question using the re-ranking model.

        Inputs:
            question (str): The input query / question.
            retrieved_chunks (list of str): The retrieved chunks to be re-ranked.
            retrieved_contexts (list of str): The corresponding generated contexts for each chunk.
            retrieved_chunk_titles (list of str): The corresponding generated chunk titles for each chunk.

        Outputs:
            list of tuple: List of (chunk, context, chunk_title, score), sorted by score in descending order regarding the question.
        """
        scores = []
        batch_size=8
        
        ## chunks+context+title for generation
        # Combine chunk title, chunks and contexts for re-ranking
        combined_data = [f"Title: {chunk_title}\nChunk: {chunk}\n\nContext: {context}" for chunk, context, chunk_title in zip(retrieved_chunks, retrieved_contexts, retrieved_chunk_titles)]
        
        # Process in batches
        for i in range(0, len(combined_data), batch_size):
            batch_data = combined_data[i: i+batch_size]
            
            # Tokenize the batch
            inputs = self.rerank_tokenizer(
                [question] * len(batch_data),  # Repeat the question for all elements in the batch
                batch_data,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move inputs to the appropriate device
            inputs = {key: value.to(self.rerank_model.device) for key, value in inputs.items()}
            
            # Forward pass to get scores
            with torch.no_grad():
                outputs = self.rerank_model(**inputs)
            
            # Extract scores (logits are the relevance scores)
            batch_scores = outputs.logits.squeeze().tolist()
            
            # Handle single-element batch case
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            
            # Append the results with their corresponding chunks and contexts
            for j, score in enumerate(batch_scores):
                scores.append((retrieved_chunks[i + j], retrieved_contexts[i + j], retrieved_chunk_titles[i + j], score))
        
        # Sort by score in descending order
        reranked_chunks = sorted(scores, key=lambda x: x[3], reverse=True)
        
        # Extract the top-k reranked chunks and contexts
        top_reranked_chunks = [c[0] for c in reranked_chunks[:top_k]]
        top_reranked_context = [c[1] for c in reranked_chunks[:top_k]]
        top_reranked_chunk_titles = [c[2] for c in reranked_chunks[:top_k]]
        
        return top_reranked_chunks, top_reranked_context, top_reranked_chunk_titles