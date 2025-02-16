from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
import os

from src.scripts.metrics import Metrics
from src.scripts.prompts import Prompts 

load_dotenv(dotenv_path="config/config.env") 

class SQuAD_NoContext:
    def __init__(self):
        self.dataset = load_dataset("squad")

def main():
    squad = SQuAD_NoContext()
    prompt = Prompts()
    metrics = Metrics()

    total_queries = int(os.getenv("TOTAL_QUERIES"))
    validation_set = squad.dataset['validation'].shuffle(seed=42).select(range(total_queries))

    retrieved_chunks_list = []
    ground_truth_context = ""
    top_k = 0

    for i, example in enumerate(tqdm(validation_set, desc='Processing Questions')):
        question = example['question']
        ground_truth = example['answers']['text'][0]

        print(f'\nExample {i+1}:')
        print(f'Question: {question}')
        print(f'Ground Truth: {ground_truth}')

        answer = prompt.generate_response_no_context_squad(question=question)

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