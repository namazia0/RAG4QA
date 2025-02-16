from dotenv import load_dotenv
from typing import Any
import requests
import json
import os

load_dotenv(dotenv_path="config/config.env")

class Prompts:
    def __init__(self, 
                llm_model = os.getenv("LLM_MODEL"), 
                tabby_api_url = "http://localhost:5001/v1",
                tabby_api_key = os.getenv("TABBY_API_KEY")):

        self.llm_model = llm_model
        self.tabby_api_url = tabby_api_url
        self.tabby_api_key = tabby_api_key

    def create_context(self, document, chunk):       
        PROMPT = f"""Please give a short succinct context to situate the chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. 

        Document: {document}

        Chunk: {chunk}
        """

        if not self.tabby_api_key:
            raise ValueError("API key is not set in the environment variable 'TABBY_API_KEY'.")

        url = f'{self.tabby_api_url}/chat/completions'
        headers = {
        'Content-Type': 'application/json',
        "Authorization": f'Bearer {self.tabby_api_key}'
        }

        payload = json.dumps({
        "model": self.llm_model,
        "max_tokens": 100,
        "stream": "False",
        "min_p": 0.05,
        "messages": [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT.format(document=document, chunk=chunk),
                        },
                    ]
                },
            ],
        "repetition_penalty": 1.05
        })

        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        return response['choices'][0]['message']['content']
    
    def generate_response_no_context_squad(self, question):
        ANSWER_PROMPT = f"""Answer the question in minimum words possible.

        Question: {question}

        Answer:
        """

        url = f"{self.tabby_api_url}/chat/completions"
        headers = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {self.tabby_api_key}"
        }

        payload = json.dumps({
        "model": self.llm_model,
        "max_tokens": 100,
        "stream": 'False',
        "min_p": 0.05,
        "temperature": 0.1,
        "messages": [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": ANSWER_PROMPT.format(QUESTION=question),
                        },
                    ]
                },
            ],
        "repetition_penalty": 1.05
        })

        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        return response['choices'][0]['message']['content']

    def generate_response_no_context_alqa(self, question):
        ANSWER_PROMPT = f"""Answer the question.

        Question: {question}

        Answer:
        """

        url = f"{self.tabby_api_url}/chat/completions"
        headers = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {self.tabby_api_key}"
        }

        payload = json.dumps({
        "model": self.llm_model,
        "max_tokens": 250,
        "stream": 'False',
        "min_p": 0.05,
        "temperature": 0.1,
        "messages": [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": ANSWER_PROMPT.format(QUESTION=question),
                        },
                    ]
                },
            ],
        "repetition_penalty": 1.05
        })

        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        return response['choices'][0]['message']['content']
    
    def generate_response_squad(self, context: str, question: str) -> tuple[str, Any]:
        ## only chunks
        ANSWER_PROMPT = f"""Answer the question in minimum words possible using only the given context and no full stop at the end.

        Context: {context}
        Question: {question}

        Answer:"""

        ## chunk + context
        # ANSWER_PROMPT = f"""Answer the question in minimum words possible using only the given relevant information which consists of a context and a chunk where
        # the initial document is divided into multiple chunks and for each chunk you created a context to situate it into the document.
        
        # Relevant information: {context}
        # Question: {question}.  

        # Do not preamble. 

        # Answer:"""

        ## chunk title + context + chunk 
        # ANSWER_PROMPT = f"""Answer the question in minimum words possible using only the given relevant information which consists of chunk title, context, and a chunk 
        # where the initial document is divided into multiple chunks and for each chunk you created a context and a chunk title to situate it into the document.
        
        # Relevant information: {context}
        # Question: {question}.  

        # Do not preamble. 

        # Answer:"""

        url = f"{self.tabby_api_url}/chat/completions"
        headers = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {self.tabby_api_key}"
        }

        payload = json.dumps({
        "model": self.llm_model,
        "max_tokens": 100,
        "stream": 'False',
        "min_p": 0.05,
        "temperature": 0.1,
        "messages": [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": ANSWER_PROMPT.format(context, question),
                        },
                    ]
                },
            ],
        "repetition_penalty": 1.05
        })

        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        return response['choices'][0]['message']['content']
    
    def generate_response_alqa(self, context: str, question: str) -> tuple[str, Any]:
        ANSWER_PROMPT = f"""Answer the question using only the given context.

        Context: {context}
        Question: {question}

        Answer:"""

        url = f"{self.tabby_api_url}/chat/completions"
        headers = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {self.tabby_api_key}"
        }

        payload = json.dumps({
        "model": self.llm_model,
        "max_tokens": 250,
        "stream": 'False',
        "min_p": 0.05,
        "temperature": 0.1,
        "messages": [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": ANSWER_PROMPT.format(context, question),
                        },
                    ]
                },
            ],
        "repetition_penalty": 1.05
        })

        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        return response['choices'][0]['message']['content']

    def multi_query_decomposition(self, original_query):
        """
        Decompose the original query into simpler sub-queries.
        
        Input:
        original_query (str): The original complex query
        
        Returns:
        sub_queries (str): String of simpler sub-queries
        """

        DECOMPOSTION_PROMPT = f"""
        You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
        Given the original query, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

        Original query: {original_query}

        Only return the sub-queries and do not preamble.
        """

        ### Few-shot Chain-Of-Thought CoT Prompting example
        # f"""
        # You are a helpful assistant that prepares queries that will be sent to a search component.
        # Sometimes, these queries are very complex.
        # Your job is to simplify complex queries into multiple queries that can be answered
        # in isolation to eachother.

        # If the query is simple, then keep it as it is.
        # Examples
        # 1. Query: Did Microsoft or Google make more money last year?
        # Decomposed Questions: 1.How much profit did Microsoft make last year? 2.How much profit did Google make last year?]
        # 2. Query: What is the capital of France?
        # Decomposed Questions: What is the capital of France?
        # 3. Query: {question}
        # Decomposed Questions:
        # """

        ### One-shot example
        # Example query: In the case of Australis Construction Company v Leichhardt Municipal Council [2006] NSWLEC 38, what was the issue with the proposed floor space ratio (FSR) and how did it impact the desired future character of the Nanny Goat Hill Distinctive Neighbourhood?

        # Example sub-queries:
        # 1. What was the issue with the proposed floor space ratio (FSR)?
        # 2. How did it impact the desired future character of the Nanny Goat Hill Distinctive Neighbourhood?

        url = f"{self.tabby_api_url}/chat/completions"
        headers = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {self.tabby_api_key}"
        }

        payload = json.dumps({
        "model": self.llm_model,
        "max_tokens": 200,
        "stream": 'False',
        "min_p": 0.05,
        "temperature": 0.1,
        "messages": [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": DECOMPOSTION_PROMPT.format(original_query),
                        },
                    ]
                },
            ],
        "repetition_penalty": 1.05
        })

        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        return response['choices'][0]['message']['content']
    
    def reasoning_multi_query(self, original_query, question_answer_pair):
        """
        Answer the original query by using the question answer pair for each generated sub-query.
        """
        REASONING_TEMPLATE = f"""
        You are a helpful assistant that can answer complex queries.
        Here is the original question: {original_query}

        You have split this original question up into simpler sub-queries and generated an answer for each sub-query. 

        Sub-query answer pair: {question_answer_pair}

        Reason about the final answer regarding the original question based on these sub-queries and their answers that you have generated.
        Only return the detailed answer and do not preamble.

        Final Answer:
        """

        url = f"{self.tabby_api_url}/chat/completions"
        headers = {
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {self.tabby_api_key}"
        }

        payload = json.dumps({
        "model": self.llm_model,
        "max_tokens": 250,
        "stream": 'False',
        "min_p": 0.05,
        "temperature": 0.1,
        "messages": [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": REASONING_TEMPLATE.format(original_query, question_answer_pair),
                        },
                    ]
                },
            ],
        "repetition_penalty": 1.05
        })

        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        return response['choices'][0]['message']['content']
    
    def create_chunk_title(self, document, chunk):
        CHUNK_TITLE = f"""
        What is the title of the following chunk within the given document?
        Your response MUST be the title of the chunk, and nothing else. DO NOT respond with anything else.
        Document: {document}
        Chunk: {chunk}
        """

        if not self.tabby_api_key:
            raise ValueError("API key is not set in the environment variable 'TABBY_API_KEY'.")

        url = f'{self.tabby_api_url}/chat/completions'
        headers = {
        'Content-Type': 'application/json',
        "Authorization": f'Bearer {self.tabby_api_key}'
        }

        payload = json.dumps({
        "model": self.llm_model,
        "max_tokens": 15,
        "stream": "False",
        "min_p": 0.05,
        "messages": [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": CHUNK_TITLE.format(document, chunk),
                        },
                    ]
                },
            ],
        "repetition_penalty": 1.05
        })

        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        return response['choices'][0]['message']['content']
    
    def summarize_chunk(self, title: str, chunk: str):
        PROMPT_SUMMARIZE = f"""
        You are tasked with creating a short summary of the following content which is about {title}. 

        Content to summarize: {chunk}

        Please provide a brief summary of the above content in 2-3 sentences. The summary should capture the key points and be concise. We will be using it as a key part of our search pipeline when answering user queries about this content. 
        
        Avoid using any preamble whatsoever in your response. Statements such as 'here is the summary' or 'the summary is as follows' are prohibited. You should get straight into the summary itself and be concise. Every word matters.
        """

        if not self.tabby_api_key:
            raise ValueError("API key is not set in the environment variable 'TABBY_API_KEY'.")

        url = f'{self.tabby_api_url}/chat/completions'
        headers = {
        'Content-Type': 'application/json',
        "Authorization": f'Bearer {self.tabby_api_key}'
        }

        payload = json.dumps({
        "model": self.llm_model,
        "max_tokens": 500,
        "stream": "False",
        "min_p": 0.05,
        "messages": [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT_SUMMARIZE.format(title, chunk),
                        },
                    ]
                },
            ],
        "repetition_penalty": 1.05
        })

        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        return response['choices'][0]['message']['content']
    
    def summarize(self, retrieved_chunks):        
        """
        Summarize the retrieved chunks to reduce irrelevant content before passing them to the LLM.
        """
        
        PROMPT_SUMMARIZE = f"""
        You are a highly skilled assistant designed to summarize relevant information from text. Below is a collection of text chunks that have been ranked based on their relevance to a specific query. Your task is to:
            
        Read and analyze the chunks provided.
        Identify the most critical, relevant, and insightful information that directly addresses the main query.
        Synthesize the information into a concise, coherent summary, ensuring that no important details are missed.
        Avoid redundant information and provide a balanced summary that reflects the content's intent and coverage.
        
        Chunks: {retrieved_chunks}
        
        Respond only with the summary and do not start with "Here is the summary".
        """

        if not self.tabby_api_key:
            raise ValueError("API key is not set in the environment variable 'TABBY_API_KEY'.")

        url = f'{self.tabby_api_url}/chat/completions'
        headers = {
        'Content-Type': 'application/json',
        "Authorization": f'Bearer {self.tabby_api_key}'
        }

        payload = json.dumps({
        "model": self.llm_model,
        "max_tokens": 500,
        "stream": "False",
        "min_p": 0.05,
        "messages": [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": PROMPT_SUMMARIZE.format(retrieved_chunks),
                        },
                    ]
                },
            ],
        "repetition_penalty": 1.05
        })

        response = requests.request("POST", url, headers=headers, data=payload)
        response = json.loads(response.text)
        return response['choices'][0]['message']['content']