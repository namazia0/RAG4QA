# SQuAD RAG System

A hybrid retrieval-augmented generation (RAG) system for question answering using the SQuAD dataset. This implementation combines dense and sparse retrieval with reranking and LLM-based answer generation.

## Features

- Hybrid retrieval combining dense (semantic) and sparse (BM25) search
- Cross-encoder reranking of retrieved contexts
- Integration with TabbyAPI for LLM-based answer generation
- Comprehensive evaluation metrics (ROUGE, BERT Score, BLEU, etc.)
- Ablation study capabilities for model components

## Architecture

The system uses a multi-stage pipeline:
1. **Retrieval**: Combines semantic search (Sentence Transformers) with BM25 for robust context retrieval
2. **Reranking**: Uses cross-encoder to rerank retrieved passages
3. **Generation**: Leverages TabbyAPI for answer generation
4. **Evaluation**: Multiple metrics for comprehensive performance assessment

## Requirements

```
torch
transformers
sentence-transformers
scikit-learn
rank_bm25
nltk
tqdm
requests
rouge_score
bert_score
evaluate
datasets
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure TabbyAPI is running locally or update the API URL in the code.

3. Make sure you have CUDA-capable GPU(s) available.

## Usage

### Basic Usage

```python
from squad import SQuADRAG

# Initialize the system
rag = SQuADRAG(
    embedding_model="sentence-transformers/all-MiniLM-L12-v2",
    rerank_model="cross-encoder/ms-marco-MiniLM-L-12-v2",
    tabby_api_url="http://localhost:5005/v1"
)

# Prepare the database
rag.prepare_database(num_samples=10000)

# Answer questions
answer = rag.answer_question("What is the capital of France?")
```

### Running Ablation Studies

For model ablation:
```bash
python squad.py
```

For LLM ablation:
```bash
python squad.py --llm-ablation
```

## Configuration

Key parameters that can be configured:

- `embedding_model`: Model for semantic search (default: all-MiniLM-L12-v2)
- `rerank_model`: Cross-encoder for reranking (default: ms-marco-MiniLM-L-12-v2)
- `tabby_api_url`: URL for TabbyAPI
- `device`: GPU device to use (default: cuda)
- `num_samples`: Number of samples for database preparation

## Evaluation Metrics

The system provides comprehensive evaluation using:
- ROUGE-1, ROUGE-2, ROUGE-L scores
- BERTScore
- BLEU score
- Exact Match
- Retrieval Precision
- LLM Judge Score

## Performance

Results from ablation studies are saved in:
- `ablation_study_results.txt`: Detailed model ablation results
- Console output: Summary metrics and comparisons
-

## License

MIT License
