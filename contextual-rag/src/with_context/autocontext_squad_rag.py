from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from collections import defaultdict
from typing import List, Dict, Any
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import pickle
import nltk
import os

from src.scripts.metrics import Metrics
from src.scripts.prompts import Prompts 
from src.scripts.reranking import Reranking
from src.scripts.reciprocal_rank_fusion import ReciprocalRankFusion
from src.scripts.chunking import semantic_chunking, recursive_splitter, character_splitter

nltk.download('punkt_tab')
load_dotenv(dotenv_path="config.env")

############
# Difference compared to other squad pipeline is that this creates a title for each chunk by asking an LLM to create a title for each chunk with reference to 
# its document. This title is then added to the chunk and the context to further improve the retrieval performance.
#############


class ElasticsearchBM25:
    def __init__(self, index_name = "contextual_bm25_index"):
        self.elasticsearch_user = os.getenv("ELASTICSEARCH_USER")
        self.elasticsearch_password = os.getenv("ELASTICSEARCH_PASSWORD")
        self.es_client = Elasticsearch("https://localhost:9200", basic_auth=(self.elasticsearch_user, self.elasticsearch_password), verify_certs=False)
        self.index_name = index_name
        self.create_index()

    def create_index(self):
        """
        Create the Elasticsearch index with BM25 similarity settings.
        """
        index_settings = {
            "settings": {
                "analysis": {"analyzer": {"default": {"type": "english"}}},
                "similarity": {"default": {"type": "BM25"}},
                "index.queries.cache.enabled": False  # Disable query cache
            },
            "mappings": {
                "properties": {
                    "id": {"type": "keyword", "index": False},  # Unique entry ID
                    "chunk_title": {"type": "text", "analyzer": "english"},
                    "chunk": {"type": "text", "analyzer": "english"},
                    "context": {"type": "text", "analyzer": "english"},
                    }
            },
        }
        if not self.es_client.indices.exists(index=self.index_name):
            self.es_client.indices.create(index=self.index_name, body=index_settings)
            print(f"Created index: {self.index_name}")

    def index_documents(self, squad_data: List[Dict[str, Any]]):   
        """
        Index SQuAD data into Elasticsearch.

        Input: 
            squad_data: List of SQuAD-formatted dictionaries.
        """
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "id": doc["id"],
                    "chunk_title": doc["chunk_title"],
                    "chunk": doc["chunk"],
                    "context": doc["context"],
                },
            }
            for doc in squad_data
        ]
        success, _ = bulk(self.es_client, actions)
        self.es_client.indices.refresh(index=self.index_name)
        return success

    def search(self, query: str, top_k: int) -> List[Dict[str, Any]]:      
        """
        Search the indexed contexts with a query.
        For each chunk, we will run each BM25 search on both the chunk content and the additional context that we generated.

        Inputs: 
            query: The search query.
            k: Number of top results to return.

        Output: 
            List of search results.
        """
        self.es_client.indices.refresh(index=self.index_name)
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["chunk_title", "chunk", "context"],
                }
            },
            "size": top_k,

        }
        response = self.es_client.search(index=self.index_name, body=search_body)
        return [
            {
                "id": hit["_source"]["id"],
                "chunk_title": hit["_source"]["chunk_title"],
                "chunk": hit["_source"]["chunk"],
                "context": hit["_source"]["context"],
                "score": hit["_score"],
            }
            for hit in response["hits"]["hits"]
        ]

class SQuADRAG_autocontext:
    def __init__(self, 
                embedding_model = os.getenv("EMBEDDING_MODEL")):
                
        # Load SQuAD dataset
        self.dataset = load_dataset("squad")

        # Initialize the llm and embedding model
        self.embedding_model = SentenceTransformer(embedding_model)

        # Store context embeddings
        self.embeddings = []
        self.chunks = None
        self.documents = []
        self.metadata = []

        # Initialize the vector database
        self.db_name = "contextRAG_SQuAD_autocontext"
        self.db_path = f"./data/{self.db_name}/vector_db_squad_autocontext.pkl"      
    
    def save_db(self):
        """     
        Save the vector database to disk.
        """
        data = {
            "embeddings": self.embeddings,
            "metadata": self.metadata
        }
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Vector DB saved at {self.db_path}")

    def load_db(self):
        if not os.path.exists(self.db_path):
            raise ValueError("Vector database file not found. Create a new database.")
        with open(self.db_path, "rb") as file:
            data = pickle.load(file)
        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]
    
    def _embed_and_store(self, texts: List[str]):   
        """
        Add chunks and their context to the vector database.
        """
        encodings = self.embedding_model.encode(texts, show_progress_bar=True)
        self.embeddings.extend(encodings)

    def prepare_database(self, prompt, num_samples=256):
        """
        Prepare the chunk + created context database with embeddings.
        Process the SQuAD dataset, chunk the contexts, and store in the vector database.
        """
        if os.path.exists(self.db_path):
            print(f"VectorDB {self.db_name} exists. Loading vector database from disk.")
            self.load_db()
            return

        train_data = self.dataset['validation']

        # Get unique contexts
        self.chunks = [{"id": entry["id"], "title": entry["title"], "context": context} 
                        for context, entry in {entry["context"]: entry for entry in train_data}.items()]

        # Group contexts by title
        title_to_contexts = defaultdict(list)
        for chunks in self.chunks:
            title_to_contexts[chunks["title"]].append(chunks["context"])

        # Concatenate contexts for each title to create a document for each title
        self.documents = [
            {"title": title, "document": " ".join(contexts)} 
            for title, contexts in title_to_contexts.items()]

        chunk_id = 0
        chunk_and_context_embed = []
        total_number_of_chunks = 0
        avg_number_of_chunks_per_doc = 0

        for i, entry in enumerate(tqdm(self.documents, desc = "Creating chunks and a title for each new generated chunk")):
            title = entry['title']
            document = entry['document']
            document = document.replace("}", ")").replace("{", "(")
            list_chunks = semantic_chunking(document)  # Change chunking method here
            number_of_chunks = len(list_chunks)
            print(f"Number of total splits in document ({title}) {i+1}: {number_of_chunks}")
            total_number_of_chunks += number_of_chunks

            for chunk in list_chunks:
                print("\nChunk:", chunk, "\n")
                # summarized_chunk, _ = prompt.summarize_chunk(title, chunk)
                context = prompt.create_context(document, chunk)
                chunk_title = prompt.create_chunk_title(document, chunk)

                print("title: ", chunk_title)

                self.metadata.append({
                    'id': chunk_id,
                    'document_title': title,
                    'chunk_title': chunk_title,
                    'context': context, 
                    'chunk': chunk,
                    'document': document,
                })
                chunk_and_context_embed.append(f"{chunk_title}\n{context}\n{chunk}")
                chunk_id += 1

        print("Total documents: ", len(self.documents))
        print("Total number of chunks: ", total_number_of_chunks)
        avg_number_of_chunks_per_doc = total_number_of_chunks / len(self.documents)
        print("Avg number of chunks per document: ", avg_number_of_chunks_per_doc)
    
        self._embed_and_store(chunk_and_context_embed)
        self.save_db()

    def search(self, query: str, top_k: int):
        """
        Search the vector database for the top-k most similar chunks to the query.
        Inputs:
            query (str): The search query (e.g., a question).
            top_k (int): The number of top results to return.
        Output:
            List[Dict]: Top-k results with metadata and similarity scores.
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query]).flatten()

        if not self.embeddings:
            raise ValueError("The database is empty. Add some embeddings first!")

        # Compute similarities
        similarities = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]        # Indices of top-k scores

        # Retrieve top-k metadata
        retrieved_chunks_list = []
        retrieved_context_list = []
        retrieved_chunk_titles_list = []
        metadata_list = []
        for idx in top_indices:
            retrieved_chunks_list.append(self.metadata[idx]['chunk'])
            retrieved_context_list.append(self.metadata[idx]['context'])
            retrieved_chunk_titles_list.append(self.metadata[idx]['chunk_title'])
            metadata_list.append(self.metadata[idx])

        return metadata_list, retrieved_chunks_list, retrieved_context_list, retrieved_chunk_titles_list
    
    def answer_question(self, prompt, question, retrieved_chunks_list, retrieved_context_list_, retrieved_chunk_titles_list_):
        retrieved_chunks_list_text = " ".join(retrieved_chunks_list)

        ## Using the generated context and chunk title and the chunk for answering the question.
        # This enriched context is then used to generate an answer to the user's query.
        _input = [f"Title: {chunk_title}\nContext: {context}\nChunk: {chunk}\n\n" for chunk, context, chunk_title in zip(retrieved_chunks_list, retrieved_context_list_, retrieved_chunk_titles_list_)]
        input_text = " ".join(_input)

        answer = prompt.generate_response_squad(context=input_text, question=question)
        return answer

class HybridRetrieval:
    def create_elasticsearch_bm25_index(self, db: SQuADRAG_autocontext):
        es_bm25 = ElasticsearchBM25()
        es_bm25.index_documents(db.metadata)
        return es_bm25
    
    def hybrid_search(self, query: str, db: SQuADRAG_autocontext, es_bm25: ElasticsearchBM25, top_k: int, top_n: int):    
        """
        Retrieve 30 chunks using the hybrid search (sparse + dense retrieval) and then select only the top-n chunks.
        """
        # Semantic search
        metadata_list, _, _, _ = db.search(query, top_k=30)
        ranked_chunk_ids = [result['id'] for result in metadata_list]

        # BM25 search using Elasticsearch
        bm25_results = es_bm25.search(query, top_k=30)
        ranked_bm25_chunk_ids = [result['id'] for result in bm25_results]

        # Combine results
        chunk_ids = list(set(ranked_chunk_ids + ranked_bm25_chunk_ids))
        
        rrf = ReciprocalRankFusion()
        chunk_id_to_score = rrf.reciprocal_rank_fusion(chunk_ids, ranked_chunk_ids, ranked_bm25_chunk_ids)

        # Sort chunk IDs by their scores in descending order
        sorted_chunk_ids = [chunk_id for chunk_id, _ in sorted(chunk_id_to_score.items(), key=lambda x: x[1], reverse=True)]

        # Assign new scores based on the sorted order
        for index, chunk_id in enumerate(sorted_chunk_ids):
            chunk_id_to_score[chunk_id] = 1 / (index + 60)

        # Prepare the final results
        retrieved_docs = []
        semantic_count = 0
        bm25_count = 0
        if(int(os.getenv("RERANKING")) == 0):
            top_n = top_k
        for chunk_id in sorted_chunk_ids[:top_n]:
            chunk_metadata = next(chunk for chunk in db.metadata if chunk['id'] == chunk_id)

            is_from_semantic = chunk_id in ranked_chunk_ids
            is_from_bm25 = chunk_id in ranked_bm25_chunk_ids
            retrieved_docs.append({
                'metadata': chunk_metadata,
                'score': chunk_id_to_score[chunk_id],
                'from_semantic': is_from_semantic,
                'from_bm25': is_from_bm25
            })
            
            if is_from_semantic and not is_from_bm25:
                semantic_count += 1
            elif is_from_bm25 and not is_from_semantic:
                bm25_count += 1
            else:  # it's in both
                semantic_count += 0.5
                bm25_count += 0.5 
        return retrieved_docs, semantic_count, bm25_count

    def search_db_advanced(self, db: SQuADRAG_autocontext, validation_set, top_k: int, top_n: int):
        es_bm25 = self.create_elasticsearch_bm25_index(db)
        retrieved_docs_list = []
        retrieved_context_list = []
        retrieved_chunk_titles_list = []
        try:        
            total_score = 0
            total_semantic_count = 0
            total_bm25_count = 0
            total_results = 0 
            for item in tqdm(validation_set, desc="Processing advanced hybrid retrieval"):
                question = item['question']
                retrieved_docs, semantic_count, bm25_count = self.hybrid_search(question, db, es_bm25, top_k, top_n)

                total_semantic_count += semantic_count
                total_bm25_count += bm25_count
                total_results += len(retrieved_docs)

                temp_1 = []
                temp_2 = []
                temp_3 = []
                for data in retrieved_docs:
                    temp_1.append(data['metadata']['chunk'])
                    temp_2.append(data['metadata']['context'])
                    temp_3.append(data['metadata']['chunk_title'])
                retrieved_docs_list.append(temp_1)
                retrieved_context_list.append(temp_2)
                retrieved_chunk_titles_list.append(temp_3)
            
            semantic_percentage = (total_semantic_count / total_results) * 100 if total_results > 0 else 0
            bm25_percentage = (total_bm25_count / total_results) * 100 if total_results > 0 else 0

            return retrieved_docs_list, retrieved_context_list, retrieved_chunk_titles_list, {"semantic": semantic_percentage, "bm25": bm25_percentage}

        finally:
            # Delete the Elasticsearch index
            if es_bm25.es_client.indices.exists(index=es_bm25.index_name):
                es_bm25.es_client.indices.delete(index=es_bm25.index_name)
                print(f"Deleted Elasticsearch index: {es_bm25.index_name}")

def main():
    # Initialize RAG pipeline
    rag = SQuADRAG_autocontext()
    hybrid_retrieval = HybridRetrieval()
    prompt = Prompts()
    reranker = Reranking()
    metrics = Metrics()

    print("Preparing database...")
    rag.prepare_database(prompt, num_samples=10570)

    hybrid = int(os.getenv("HYBRID"))       # 1 = True, 0 = False
    reranking = int(os.getenv("RERANKING")) 
    
    # Select a subset of validation data for testing
    total_queries = int(os.getenv("TOTAL_QUERIES"))
    validation_set = rag.dataset['validation'].shuffle(seed=42).select(range(total_queries))
    top_k = int(os.getenv("TOP_K"))
    top_n = int(os.getenv("TOP_N"))
    
    if hybrid == 1:
        print("Hybrid search is running.")
        retrieved_docs_list, retrieved_context_list, retrieved_chunk_titles_list, bm25_dict = hybrid_retrieval.search_db_advanced(rag, validation_set, top_k, top_n)

    for i, example in enumerate(tqdm(validation_set, desc = 'Processing Questions')):
        question = example['question']
        ground_truth_context = example['context']
        ground_truth = example['answers']['text'][0]    # Get first answer as ground truth 

        print(f'\nExample {i+1}:')
        print(f'Question: {question}')
        print(f'Ground Truth: {ground_truth}')

        if hybrid == 1:
            retrieved_chunks_list = retrieved_docs_list[i] 
            retrieved_context_list_ = retrieved_context_list[i]
            retrieved_chunk_titles_list_ = retrieved_chunk_titles_list[i]
        
        # Get RAG answer and calculate scores
        if hybrid == 1 and reranking == 1:
            retrieved_chunks_list, retrieved_context_list_, retrieved_chunk_titles_list_  = reranker.rerank_autocontext(question, retrieved_chunks_list, retrieved_context_list_, retrieved_chunk_titles_list_, top_k)
        if hybrid == 0 and reranking == 0:   # no reranking, only embedding
            _, retrieved_chunks_list, retrieved_context_list_, retrieved_chunk_titles_list_ = rag.search(query=question, top_k=top_k)     
        
        # if hybrid == 1 and reranking == 0 only fetch answer_question()
        answer = rag.answer_question(prompt, question, retrieved_chunks_list, retrieved_context_list_, retrieved_chunk_titles_list_)   

        item = {}
        item['question'] = question
        item['ground_truth'] = ground_truth
        item['prediction'] = answer        
        if answer:
            print(f'Prediction: {answer}')
            print('Evaluation Scores:')
            all_scores = metrics.evaluate(item, retrieved_chunks_list, ground_truth_context, top_k)
        else: 
            print('RAG answer: No valid answer found.')
        print('-' * 50)   

    print("\n\nAverage Results:")
    print(f"Total Queries: {total_queries}")
    if reranking == 1:
        print(f"Number of retrieved chunks after hybrid search and reciprocal rank fusion: top-{top_n}")
        print(f"Number of retrieved chunks after reranking: top-{top_k}")
    else:
        print(f"Generating response from top-{top_k} retrieved chunks.")
    print("Retrieval:")
    print("\tPrecision@{}: {:.4f}".format(top_k, np.mean(all_scores['precision_at_k'])))

    if hybrid == 1:
        print(f"\tPercentage of results from semantic search in the top-{top_n} before reranking: {bm25_dict['semantic']:.2f}%")
        print(f"\tPercentage of results from BM25 in the top-{top_n} before reranking: {bm25_dict['bm25']:.2f}%")

    metrics.get_average_results(all_scores)    

    print('\n\nEND\n')
    print('-' * 50)
    print('-' * 50)

if __name__ == "__main__":
    main()