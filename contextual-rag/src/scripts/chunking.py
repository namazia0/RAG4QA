from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from semantic_router.encoders import HuggingFaceEncoder
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List

import os

load_dotenv(dotenv_path="config/config.env")

class MyEmbeddings:
    def __init__(self):
        embedding_model = os.getenv("EMBEDDING_MODEL")
        self.model = SentenceTransformer(embedding_model)
        self.encoder = HuggingFaceEncoder(name=embedding_model)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]
    
    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query])

def semantic_chunking(document: str):
    """
    Function to perform semantic chunking based on embedding similarity of group of sentences.
    """
    # breakpoint_threshold_type="standard_deviation"
    # breakpoint_threshold_type="interquartile"
    # breakpoint_threshold_type="gradient"
    embedding = MyEmbeddings()
    text_splitter = SemanticChunker(
        embedding, breakpoint_threshold_type="percentile", min_chunk_size=50, breakpoint_threshold_amount=70, buffer_size=1)  
    list_chunks = text_splitter.split_text(document)
    return list_chunks

def recursive_splitter(document: str):  
    """
    Splits text recursively into chunks with specified overlap and size.
    """ 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    chunks = text_splitter.create_documents([document])

    list_chunks = []
    for chunk in chunks:
        list_chunks.append(chunk.page_content)

    return list_chunks
    
def character_splitter(document: str):
    """
    Splits text into fixed-size character chunks.
    """
    list_chunks = []
    chunk_size = 500 # Characters
    overlap = 50
    for i in range(0, len(document), chunk_size - overlap):
        chunk = document[i:i + chunk_size]
        list_chunks.append(chunk)
    return list_chunks   