from datasets import load_dataset

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the SQuAD dataset
squad = load_dataset("squad")

# Extract unique topics (titles) from the dataset
train_topics = set(squad['train']['title'])
validation_topics = set(squad['validation']['title'])

print(f"Number of unique topics in the train: {len(train_topics)}")
print(f"Number of unique topics in the valiation: {len(validation_topics)}")

# Calculate the average number of questions per context
total_contexts = set(squad['train']['context'])
total_question_train = set(squad['train']['question'])
average_questions_per_context = len(total_question_train) / len(total_contexts)
print(f"Average number of questions per context in train: {average_questions_per_context:.2f}")

total_contexts = set(squad['validation']['context'])
total_question_val = set(squad['validation']['question'])
average_questions_per_context = len(total_question_val) / len(total_contexts)
print(f"Average number of questions per context in validation: {average_questions_per_context:.2f}")



# Extract questions and answers from the dataset
questions = squad['train']['question']
answers = [item['text'][0] for item in squad['train']['answers'] if item['text']]  # Use the first answer for simplicity

# Tokenize and calculate average number of tokens
question_tokens = [len(word_tokenize(question)) for question in questions]
answer_tokens = [len(word_tokenize(answer)) for answer in answers]

average_question_tokens = sum(question_tokens) / len(question_tokens)
average_answer_tokens = sum(answer_tokens) / len(answer_tokens)

print(f"Average number of tokens in questions: {average_question_tokens:.2f}")
print(f"Average number of tokens in answers: {average_answer_tokens:.2f}")

questions = squad['validation']['question']
answers = [item['text'][0] for item in squad['validation']['answers'] if item['text']]

# Tokenize and calculate average number of tokens
question_tokens = [len(word_tokenize(question)) for question in questions]
answer_tokens = [len(word_tokenize(answer)) for answer in answers]

average_question_tokens = sum(question_tokens) / len(question_tokens)
average_answer_tokens = sum(answer_tokens) / len(answer_tokens)

print(f"Average number of tokens in questions val: {average_question_tokens:.2f}")
print(f"Average number of tokens in answers val: {average_answer_tokens:.2f}")