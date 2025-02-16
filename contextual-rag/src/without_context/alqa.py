from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import os

from src.scripts.metrics import Metrics
from src.scripts.prompts import Prompts 

load_dotenv(dotenv_path="config/config.env") 

# ALQA: Australian Legal Question Answering dataset

class ALQA_NoContext:
    def __init__(self):
        self.dataset = load_dataset("Ramitha/open-australian-legal-qa-simplified-sent-tokenized")

def main():
    alqa = ALQA_NoContext()
    prompt = Prompts()
    metrics = Metrics()

    total_queries = int(os.getenv("TOTAL_QUERIES"))
    validation_set = alqa.dataset['train'].shuffle(seed=42).select(range(total_queries))

    retrieved_chunks_list = []
    ground_truth_context = ""
    top_k = 0

    decompose = False

    for i, example in enumerate(tqdm(validation_set, desc='Processing Questions')):
        question = example['question']
        ground_truth = example['answer']

        print(f'\nExample {i+1}:')
        print(f'Question: {question}\n')
        print(f'Ground Truth: {ground_truth}\n')

        if decompose:
            question_answer_pair = []
            multi_query = prompt.multi_query_decomposition(original_query=question)
            queries_list = multi_query.split("\n")
            for sub_query in queries_list:
                sub_answer = prompt.generate_response_no_context_alqa(sub_query)
                question_answer_pair.append(f"Sub-query: {sub_query}\nAnswer: {sub_answer}\n\n")        
            question_answer_pair_text = " ".join(question_answer_pair)
            answer = prompt.reasoning_multi_query(question, question_answer_pair_text)

        if not decompose: 
            answer = prompt.generate_response_no_context_alqa(question)

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
    
    print("Retrieval:")
    print("\tPrecision@0: -")

    metrics.get_average_results(all_scores)  

    print('\n\nEND\n')
    print('-' * 50)
    print('-' * 50)

if __name__ == "__main__":
    main()