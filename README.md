# RTEB: Retrieval-focused Text Embedding Benchmark

<h3 align="center">
    <a href="https://huggingface.co/spaces/embedding-benchmark/ebr"><img style="float: middle; padding: 10px 10px 10px 10px;" width="60" height="55" src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" /></a>
</h3>

## Overview

RTEB (Retrieval-focused Text Embedding Benchmark) is a new, reliable, high-quality benchmark designed to evaluate the retrieval accuracy of embedding models and rerankers. Unlike existing benchmarks, RTEB focuses specifically on retrieval tasks that reflect real-world applications, particularly search and RAG (Retrieval-Augmented Generation) systems.

## Motivation

### Limitations of Existing Benchmarks

Most existing benchmarks used to evaluate embedding models and rerankers have significant shortcomings for real-world applications:

**MTEB Issues:**
- Many models train on the test set, some intentionally, leading to inflated performance scores
- Measures performance on tasks such as STS (Semantic Textual Similarity) and classification which are less relevant for typical embedding applications today (search and RAG)
- Uses only academic datasets, many of which are irrelevant to actual enterprise use cases and problems

**TREC Issues:**
- Too large to be of practical use for most evaluation scenarios
- Contains many false negatives due to the data collection process
- Has data which is less relevant for enterprise retrieval applications

### Our Solution

RTEB addresses these shortcomings by:
- **Retrieval-focused**: Concentrating on high-quality retrieval-specific datasets
- **Real-world relevance**: Including datasets that reflect actual enterprise use cases
- **Tiered approach**: Implementing a multi-tier system (0-3) to prevent overfitting while maintaining transparency
- **Practical dataset sizes**: Ensuring datasets are large enough to be meaningful but small enough for efficient evaluation

## Dataset Structure

RTEB uses a hierarchical organization system:

1. **Dataset**: The minimal unit of evaluation. Each dataset produces one score per model for a fixed evaluation metric (e.g., NDCG@10)
2. **Dataset Groups**: Collections of related datasets (e.g., "legal", "healthcare", "code"). Datasets can belong to multiple groups
3. **Group Scoring**: Each group has a single aggregated score per model, calculated as the weighted average of constituent dataset scores
4. **Main Leaderboard**: Features the "text overall" group containing almost all text datasets

### Available Groups

Based on production interactions with embedding model users, RTEB includes these key groups:

- **Domain-specific**: text, legal, code, healthcare, finance, conversation, long-context, multilingual
- **Multimodal** (future): text-to-photo, document screenshots, figures & tables, image-to-image

## Tiered Dataset System

To prevent overfitting while maintaining transparency, RTEB implements a multi-tier system:

### Tier 0 Datasets (Fully Open)
- All files publicly available: `corpus.jsonl`, `queries.jsonl`, `relevance.jsonl`
- Includes existing high-quality datasets with open test sets
- At least one open dataset per task group

### Tier 1 Datasets (Docs & Queries Open)
- Documents and queries publicly available, but relevance judgments are held out
- Allows for development and analysis while preventing direct optimization on labels

### Tier 2 Datasets (Only Docs Open)
- Only document corpus is publicly available
- Queries and relevance judgments are held out

### Tier 3 Datasets (Fully Held Out)
- Private evaluation sets to prevent manipulation
- All data is held out - corpus, queries, and relevance judgments
- Public metadata includes: basic description, data sources, dataset statistics, token length distributions
- Five sample (query, document, relevance) triplets provided for reference
- Most closed datasets in RTEB are Tier 3

## Getting Started

### Installation

```bash
git clone <repository-url>
cd rteb
pip install -r requirements.txt
```

**Note**: Additional packages may be required depending on the models you want to evaluate. The framework uses lazy loading, so model-specific dependencies (like `sentence-transformers`, `openai`, `cohere`, etc.) are only imported when needed. Install additional packages as required:

```bash
# For sentence-transformers models
pip install sentence-transformers

# For OpenAI models
pip install openai

# For Cohere models
pip install cohere

# For VoyageAI models
pip install voyageai

# For other specific model requirements, check the model's documentation
```

### Quick Start

1. **List available models and datasets:**
```bash
python -m ebr --list-models
python -m ebr --list-tasks
```

2. **Run evaluation on all datasets with all models:**
```bash
python -m ebr
```

3. **Evaluate specific models on specific datasets:**
```bash
python -m ebr --models "text-embedding-3-small_float32_1536d,all-MiniLM-L6-v2" --tasks "FinanceBench,LegalQuAD"
```

### Usage Examples

#### Example 1: Evaluate OpenAI models on finance datasets
```bash
python -m ebr \
  --models "text-embedding-3-small_float32_1536d,text-embedding-3-large_float32_3072d,text-embedding-004_float32_768d" \
  --tasks "FinanceBench,HC3Finance,FinQA" \
  --gpus 1 \
  --batch_size 32
```

#### Example 2: Compare sentence-transformers models on code datasets
```bash
python -m ebr \
  --models "sentence-transformers__all-MiniLM-L6-v2_float32_384d,sentence-transformers__all-mpnet-base-v2_float32_768d" \
  --tasks "APPS,DS1000,HumanEval,MBPP,WikiSQL,FreshStack" \
  --save_path "output/code_evaluation" \
  --gpus 1
```

#### Example 3: Evaluate on healthcare datasets with CPU
```bash
python -m ebr \
  --models "sentence-transformers__all-MiniLM-L6-v2_float32_384d" \
  --tasks "ChatDoctor_HealthCareMagic,CUREv1_en" \
  --cpus 4 \
  --batch_size 16
```

#### Example 4: Run with memory optimization
```bash
python -m ebr \
  --models "nvidia__NV-Embed-v2_float32_4096d" \
  --gpus 1 \
  --bf16 \
  --offload-model \
  --embd_in_memory_threshold 100000
```

### Available Models

RTEB supports multiple model types organized by provider:

- **OpenAI**: text-embedding-3-small, text-embedding-3-large, text-embedding-004
- **Sentence-transformers**: all-MiniLM-L6-v2, all-mpnet-base-v2, etc.
- **NVIDIA**: NV-Embed-v2
- **Cohere**: embed-english-v3.0, embed-multilingual-v3.0
- **VoyageAI**: voyage-3, voyage-3.5
- **And many more...**

Use `--list-models` to see all available models with their specifications.

### Available Datasets

RTEB includes datasets across multiple domains:

- **Legal**: AILACasedocs, AILAStatutes, LegalSummarization, LegalQuAD
- **Finance**: FinanceBench, HC3Finance, FinQA  
- **Code**: APPS, DS1000, HumanEval, MBPP, WikiSQL, FreshStack
- **Healthcare**: ChatDoctor_HealthCareMagic, CUREv1_en, CUREv1_fr
- **Closed datasets**: 13 held-out datasets across various domains and languages

Use `--list-tasks` to see all available datasets with their tier information and groups.

## Results Format

### Individual Model Results

Results for each model-dataset combination are stored in JSON format with comprehensive metrics:

```json
{
  "ndcg_at_1": 0.18,
  "ndcg_at_3": 0.16432,
  "ndcg_at_5": 0.16492,
  "ndcg_at_10": 0.19717,
  "ndcg_at_20": 0.22987,
  "ndcg_at_50": 0.28899,
  "ndcg_at_100": 0.35235,
  "map_at_1": 0.04421,
  "map_at_3": 0.08374,
  "map_at_5": 0.10432,
  "map_at_10": 0.12772,
  "map_at_20": 0.13851,
  "map_at_50": 0.15196,
  "map_at_100": 0.16262,
  "recall_at_1": 0.04421,
  "recall_at_3": 0.09949,
  "recall_at_5": 0.14982,
  "recall_at_10": 0.25601,
  "recall_at_20": 0.35048,
  "recall_at_50": 0.55142,
  "recall_at_100": 0.83315,
  "precision_at_1": 0.18,
  "precision_at_3": 0.14667,
  "precision_at_5": 0.12,
  "precision_at_10": 0.092,
  "precision_at_20": 0.065,
  "precision_at_50": 0.0428,
  "precision_at_100": 0.0314,
  "model_name": "all-MiniLM-L6-v2",
  "embd_dim": 384,
  "embd_dtype": "float32"
}
```

### Evaluation Metrics

Models are evaluated using multiple retrieval metrics:

- **NDCG@{1,3,5,10,20,40,100}**: Normalized Discounted Cumulative Gain
- **MAP@{1,3,5,10,20,50,100}**: Mean Average Precision  
- **Recall@{1,3,5,10,20,50,100}**: Recall at different cutoffs
- **Precision@{1,3,5,10,20,50,100}**: Precision at different cutoffs

**Default metric**: NDCG@10 is used as the primary ranking metric on leaderboards.

### Output Structure

Results are organized in the following directory structure:

```
output/
├── {dataset_name}/
│   └── {model_id}/
│       ├── retrieve_eval.json      # Evaluation metrics
│       └── retrieve_pred.json      # Predictions (if --save_prediction)
│
results/
├── results.json                    # Compiled results for all datasets
├── models.json                     # Model metadata
└── datasets.json                   # Dataset group information
```

### Dataset File Format

Each dataset follows a standard format:

- **corpus.jsonl**: Document collection
  ```json
  {"_id": "doc1", "title": "Document Title", "text": "Document content..."}
  ```

- **queries.jsonl**: Query collection  
  ```json
  {"_id": "query1", "text": "What is the query?"}
  ```

- **relevance.jsonl**: Relevance judgments
  ```json
  {"query1": {"doc1": 1, "doc2": 0}}
  ```

## Configuration Options

### GPU/CPU Configuration
- `--gpus N`: Number of GPUs for encoding (default: 0)
- `--cpus N`: Number of CPUs for computation (default: 1)
- `--bf16`: Use bfloat16 precision for memory efficiency

### Batch and Memory Settings
- `--batch_size N`: Encoding batch size (default: 16)
- `--embd_batch_size N`: Similarity computation batch size (default: 1024)
- `--embd_in_memory_threshold N`: Embedding memory threshold (default: 200000)
- `--offload-model`: Offload model after encoding to save memory

### Output Options
- `--save_path DIR`: Output directory (default: "output/")
- `--save_prediction`: Save detailed predictions
- `--keep_embds`: Keep embedding files after retrieval
- `--topk N`: Number of top documents per query (default: 100)
- `--overwrite`: Overwrite existing results

## Contributing an Embedding Model

We welcome contributions of new embedding models! The framework supports:

1. **Sentence-transformers models**: Inherit from `SentenceTransformersEmbeddingModel`
2. **API-based models**: Inherit from the appropriate API model class
3. **Custom models**: Implement the `EmbeddingModel` interface

### Model Submission Process

1. Create a model class inheriting from the appropriate base class
2. Implement required methods (`embed`, `_id`, etc.)
3. Add model metadata using `ModelMeta`
4. Submit a PR with your model and results on Tier 1 datasets
5. Model will be evaluated on Tier 2 datasets by organizers

### Model Metadata Requirements

Each model must include:
- **Dimensionality**: Embedding vector size
- **Model parameters**: Total parameter count  
- **Precision**: float32, int8, binary, etc.
- **Context length**: Maximum input token length
- **Similarity metric**: cosine, dot_product, etc.
- **Reference**: Link to model documentation/paper

## Leaderboard and Results

Results are compiled into a comprehensive leaderboard showing:
- Model rankings across different dataset groups
- Detailed performance breakdowns by domain
- Model specifications and efficiency metrics
- Statistical significance testing

Only models with complete results across all datasets are included in the final leaderboard to ensure fair comparison.

## Dataset Size Guidelines

Datasets in RTEB follow specific size constraints for practical evaluation:

**Lower bounds:**
- Minimum 1000 documents
- Minimum 50 queries (ideally 100+)

**Upper bounds:**
- Most datasets: <100M tokens
- Large datasets: <1B tokens (typically closed)
- Target: Evaluation within 10 minutes on 8xH100 + CPU

## Acknowledgments

RTEB is designed to address the needs of the embedding community by providing a practical, fair, and comprehensive evaluation framework. We thank all contributors and the broader NLP community for their feedback and support.

## License

[License information to be added]

## Citation

If you use RTEB in your research, please cite:

```bibtex
[Citation to be added]
```