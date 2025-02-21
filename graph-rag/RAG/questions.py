import sys
import os

# Add the utils directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../pipeline/utils')))

# Now import the function
from load_huggingface_dataset import get_context_merged_datset


# Call the function to get the merged dataset
merged_df = get_context_merged_datset()

# Extract questions
questions = merged_df['question']

# Print the questions
for i, question_list in enumerate(questions[:50]):
    for question in question_list:
        print(f"  - {question}")
