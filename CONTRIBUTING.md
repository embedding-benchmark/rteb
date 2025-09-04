# Contributing to RTEB (Retrieval Embedding Benchmark)

Thank you for your interest in contributing to RTEB! This guide will help you add new models and datasets to the benchmark.

## Table of Contents

- [Overview](#overview)
- [Adding a New Model](#adding-a-new-model)
- [Adding a New Dataset](#adding-a-new-dataset)
- [Development Setup](#development-setup)
- [Testing Your Contributions](#testing-your-contributions)
- [Submitting Your Contribution](#submitting-your-contribution)

## Overview

RTEB is designed with a modular architecture that makes it easy to add new embedding models and retrieval datasets. The framework supports:

- **Models**: Both local models (via sentence-transformers, transformers) and API-based models (OpenAI, Cohere, etc.)
- **Datasets**: Text retrieval datasets with queries, corpus, and relevance judgments

## Adding a New Model

### Step 1: Choose the Right Base Class

RTEB provides several base classes for different types of models:

- `EmbeddingModel`: Base class for all embedding models
- `APIEmbeddingModel`: For API-based models (OpenAI, Cohere, etc.)
- `SentenceTransformersEmbeddingModel`: For models compatible with sentence-transformers

### Step 2: Create Your Model Class

Create a new file in `rteb/models/` or add to an existing one. Here are examples for different model types:

#### Local Model Example (Sentence Transformers)

```python
from rteb.core.base import EmbeddingModel
from rteb.core.meta import ModelMeta
from rteb.utils.lazy_import import LazyImport

# Use lazy import for optional dependencies
SentenceTransformer = LazyImport("sentence_transformers", attribute="SentenceTransformer")


class MyCustomEmbeddingModel(EmbeddingModel):
    def __init__(self, model_meta: ModelMeta, device: str = None, **kwargs):
        super().__init__(model_meta, **kwargs)
        self._model = SentenceTransformer(
            self.model_name,
            device=device,
            trust_remote_code=True
        )

    def embed(self, data: list[str], input_type: str) -> list[list[float]]:
        """
        Embed a list of texts.
        
        Args:
            data: List of texts to embed
            input_type: Either "query" or "document"
            
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        return self._model.encode(data).tolist()


# Create model metadata
my_custom_model = ModelMeta(
    loader=MyCustomEmbeddingModel,
    model_name="my-organization/my-model-name",
    embd_dtype="float32",
    embd_dim=768,
    num_params=110_000_000,  # Number of parameters
    max_tokens=512,  # Maximum input tokens
    similarity="cosine",  # Similarity function: "cosine" or "dot"
    reference="https://huggingface.co/my-organization/my-model-name"
)
```

#### API Model Example

```python
from rteb.core.base import APIEmbeddingModel
from rteb.core.meta import ModelMeta
from rteb.utils.lazy_import import LazyImport

# Use lazy import for API client
my_api_client = LazyImport("my_api_package")


class MyAPIEmbeddingModel(APIEmbeddingModel):
    def __init__(self, model_meta: ModelMeta, api_key: str = None, **kwargs):
        super().__init__(model_meta, api_key=api_key, **kwargs)
        self._client = None

    @property
    def client(self):
        if not self._client:
            self._client = my_api_client.Client(api_key=self._api_key)
        return self._client

    def embed(self, data: list[str], input_type: str) -> list[list[float]]:
        response = self.client.embed(
            texts=data,
            model=self.model_name,
            input_type=input_type
        )
        return [embedding.values for embedding in response.embeddings]

    @staticmethod
    def rate_limit_error_type() -> type:
        """Return the exception type for rate limit errors"""
        return my_api_client.RateLimitError

    @staticmethod
    def service_error_type() -> type:
        """Return the exception type for service errors"""
        return my_api_client.ServiceError


# Create model metadata
my_api_model = ModelMeta(
    loader=MyAPIEmbeddingModel,
    model_name="my-api-model-v1",
    embd_dtype="float32",
    embd_dim=1024,
    max_tokens=8192,
    similarity="cosine",
    reference="https://docs.my-api.com/embeddings"
)
```

### Step 3: Register Your Model

Add your model to the appropriate file in `rteb/models/` and make sure it's imported in `rteb/models/__init__.py`:

```python
# In rteb/models/__init__.py
from rteb.models.my_custom_models import *
```

### Step 4: Model Metadata Fields

When creating `ModelMeta`, include these fields:

- `loader`: Your model class
- `model_name`: The model identifier
- `embd_dtype`: Data type ("float32", "int8", "binary")
- `embd_dim`: Embedding dimension
- `num_params`: Number of parameters (optional)
- `max_tokens`: Maximum input tokens (optional)
- `similarity`: Similarity function ("cosine" or "dot")
- `query_instruct`: Instruction prefix for queries (optional)
- `corpus_instruct`: Instruction prefix for documents (optional)
- `reference`: URL to model documentation (optional)
- `alias`: Display name for the model (optional)

## Adding a New Dataset

### Step 1: Understand Dataset Structure

RTEB expects datasets to have this structure:
```
dataset_name/
├── corpus.jsonl      # Document collection
├── queries.jsonl     # Query collection
└── relevance.json    # Relevance judgments
```

#### File Formats

**corpus.jsonl**: One document per line
```json
{"_id": "doc1", "text": "Document content here..."}
{"_id": "doc2", "text": "Another document..."}
```

**queries.jsonl**: One query per line
```json
{"_id": "q1", "text": "What is the capital of France?"}
{"_id": "q2", "text": "How does photosynthesis work?"}
```

**relevance.json**: Relevance judgments
```json
{
  "q1": {"doc1": 1, "doc5": 1},
  "q2": {"doc2": 1, "doc3": 2}
}
```

### Step 2: Create Dataset Class (if needed)

Most datasets can use the existing `TextRetrievalDataset` class. Only create a custom class if you need special processing:

```python
from rteb.core.base import RetrievalDataset
from rteb.core.meta import DatasetMeta
from rteb.utils.data import JSONLDataset


class MyCustomDataset(RetrievalDataset):
    LEADERBOARD: str = "Text"  # or "Code", "Legal", etc.

    def __init__(self, data_path: str, dataset_meta: DatasetMeta, **kwargs):
        super().__init__(data_path, dataset_meta, **kwargs)
        # Custom initialization if needed

    def _corpus(self) -> Dataset:
        # Custom corpus loading logic
        return JSONLDataset(self.corpus_file)

    def _queries(self) -> Dataset:
        # Custom query loading logic
        return JSONLDataset(self.queries_file)

    @property
    def relevance(self) -> dict:
        # Custom relevance loading logic
        # Return dict[query_id][doc_id] = relevance_score
        pass
```

### Step 3: Create Dataset Metadata

Add your dataset metadata to `rteb/datasets/text.py` (or create a new file):

```python
from rteb.core.meta import DatasetMeta
from rteb.datasets.text import TextRetrievalDataset  # or your custom class

MyDataset = DatasetMeta(
    loader=TextRetrievalDataset,  # or your custom class
    dataset_name="MyDataset",
    tier=0,  # 0=fully open, 1=docs+queries open, 2=docs only, 3=held out
    groups={
        "text": 1,  # Primary category
        "domain": 1,  # Domain (legal, finance, code, healthcare, etc.)
        "language": 1  # Language (english, french, german, etc.)
    },
    reference="https://example.com/dataset-paper"  # Optional reference
)
```

### Step 4: Dataset Tiers

Choose the appropriate tier for your dataset:

- **Tier 0**: Fully open (documents, queries, and relevance judgments public)
- **Tier 1**: Documents and queries released, relevance held out
- **Tier 2**: Only documents released
- **Tier 3**: Fully held out (for evaluation only)

### Step 5: Groups and Categories

Use these standard group categories:

**Primary Categories:**
- `text`: Text retrieval
- `code`: Code retrieval
- `multimodal`: Multimodal retrieval

**Domains:**
- `legal`: Legal documents
- `finance`: Financial documents
- `healthcare`: Medical/health documents
- `academic`: Academic papers
- `news`: News articles
- `web`: Web pages

**Languages:**
- `english`, `french`, `german`, `spanish`, `chinese`, etc.

**Special:**
- `performance`: Performance testing datasets
- `token_length`: Specific token length (with value)

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/embedding-benchmark/rteb.git
   cd rteb
   ```

2. **Install dependencies:**
   ```bash
   pip install -e .
   ```

3. **Install optional dependencies for your model:**
   ```bash
   pip install sentence-transformers  # for sentence-transformers models
   pip install openai                 # for OpenAI models
   pip install cohere                 # for Cohere models
   # etc.
   ```

## Testing Your Contributions

### Test Your Model

```python
from rteb.models import get_embedding_model

# Test loading your model
model = get_embedding_model(
    model_name="your-model-name",
    embd_dim=768,
    embd_dtype="float32"
)

# Test embedding
texts = ["Hello world", "This is a test"]
embeddings = model.embed(texts, "query")
print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
```

### Test Your Dataset

```python
from rteb.datasets import get_retrieval_dataset

# Test loading your dataset
dataset = get_retrieval_dataset(
    data_path="/path/to/your/data",
    dataset_name="YourDataset"
)

# Test data loading
print(f"Corpus size: {len(dataset.corpus)}")
print(f"Queries size: {len(dataset.queries)}")
print(f"Relevance judgments: {len(dataset.relevance)}")
```

### Run Evaluation

```bash
python -m rteb \
    --model_name your-model-name \
    --embd_dim 768 \
    --embd_dtype float32 \
    --dataset_name YourDataset \
    --data_path /path/to/data \
    --output_path ./results
```

## Submitting Your Contribution

1. **Fork the repository** on GitHub

2. **Create a feature branch:**
   ```bash
   git checkout -b add-my-model-dataset
   ```

3. **Make your changes** following this guide

4. **Test thoroughly** using the testing steps above

5. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add MyModel and MyDataset"
   ```

6. **Push to your fork:**
   ```bash
   git push origin add-my-model-dataset
   ```

7. **Create a Pull Request** with:
   - Clear description of what you're adding
   - Test results showing your model/dataset works
   - Any special requirements or dependencies

## Best Practices

### For Models:
- Use lazy imports for optional dependencies
- Handle API rate limits and errors appropriately
- Include proper error handling in `embed()` method
- Test with various input sizes
- Document any special requirements

### For Datasets:
- Ensure data files are properly formatted
- Include comprehensive metadata
- Choose appropriate tier and groups
- Provide clear dataset description and reference
- Test with different models

### General:
- Follow existing code style and patterns
- Add appropriate error handling
- Include docstrings for public methods
- Test edge cases (empty inputs, large inputs, etc.)
- Keep dependencies minimal and optional when possible

## Getting Help

- **Issues**: Open an issue on GitHub for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Documentation**: Check existing model/dataset implementations for examples

Thank you for contributing to RTEB! Your additions help make embedding evaluation more comprehensive and accessible.