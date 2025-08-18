import torch
import numpy as np
from typing import Optional, Any
from pytorch_lightning import LightningDataModule

from ebr.datasets import get_retrieval_dataset
from ebr.utils.data import EmptyDataset, JSONLDataset


class EmbeddingDataCollator:

    def __init__(self, embd_dtype="float32"):
        self.embd_dtype = embd_dtype

    def _unpack_binary_embedding(self, byte_values):
        """Unpack bytes to binary bits for binary embeddings."""
        # Convert list of integers (bytes) to numpy array
        # Handle both unsigned (0-255) and signed (-128 to 127) byte values
        byte_array = np.array(byte_values, dtype=np.int32)  # Use int32 to handle full range
        
        # Convert signed bytes to unsigned if needed
        byte_array = byte_array % 256  # Wrap negative values to 0-255 range
        byte_array = byte_array.astype(np.uint8)
        
        # Unpack bits: each byte becomes 8 bits
        binary_bits = np.unpackbits(byte_array)
        return binary_bits.astype(np.float32)

    def __call__(self, examples):
        assert len(examples) > 0
        batch = {
            key: [example[key] for example in examples]
            for key in examples[0].keys()
        }
        
        # Handle binary embeddings by unpacking bits
        if self.embd_dtype in ["binary", "ubinary"]:
            unpacked_embds = []
            for embd in batch["embd"]:
                unpacked_embd = self._unpack_binary_embedding(embd)
                unpacked_embds.append(unpacked_embd)
            batch["embd"] = torch.tensor(np.array(unpacked_embds))
        else:
            batch["embd"] = torch.tensor(batch["embd"])
        return batch


class RetrieveDataCollator:

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self._early_truncate = True

    def __call__(self, examples):
        assert len(examples) > 0
        batch = {}
        batch["id"] = [ex["id"] for ex in examples]
        batch["text"] = [ex["text"] for ex in examples]

        if self.tokenizer:
            texts = [s.strip() for s in batch["text"]]

            if self._early_truncate:
                max_str_len = self.tokenizer.model_max_length * 6
                texts = [s[:max_str_len] for s in texts]
 
            batch["input"] = self.tokenizer(
                texts,
                padding=True, 
                truncation=True, 
                return_tensors="pt",
            )

        return batch


class RetrieveDataModule(LightningDataModule):

    def __init__(
        self, 
        data_path: str,
        dataset_name: str,
        batch_size: int = 32, 
        embd_batch_size: int = 1024, 
        num_workers: int = 4,
        dataset_kwargs: Optional[dict] = None,
        collator_kwargs: Optional[dict] = None,
        embd_dtype: str = "float32",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.embd_batch_size = embd_batch_size
        self.num_workers = num_workers
        self.embd_dtype = embd_dtype
        self.dataset = get_retrieval_dataset(
            data_path=data_path,
            dataset_name=dataset_name,
            **dataset_kwargs,
        )
        self.query_collator = None
        self.corpus_collator = None

    def prepare_data(self):
        self.dataset.prepare_data()

    def queries_dataloader(self, exclude_ids=None):
        dataset = self.dataset.queries
        if exclude_ids:
            # Create filtered dataset excluding already processed IDs
            dataset = EmptyDataset(
                [item for item in self.dataset.queries if item["id"] not in exclude_ids]
            )
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.query_collator,
        )

    def corpus_dataloader(self, exclude_ids=None):
        dataset = self.dataset.corpus
        if exclude_ids:
            # Create filtered dataset excluding already processed IDs
            dataset = EmptyDataset(
                [item for item in self.dataset.corpus if item["id"] not in exclude_ids]
            )
        return torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers,
            collate_fn=self.corpus_collator,
        )

    def set_queries_embds(self, queries_embds=None, queries_embds_files=None):
        if queries_embds:
            self.queries_embds = queries_embds
            self.queries_embd_ds = EmptyDataset(queries_embds)
        else:
            self.queries_embd_ds = JSONLDataset(queries_embds_files)
        assert len(self.queries_embd_ds) == len(self.dataset.queries)

    def set_corpus_embds(self, corpus_embds=None, corpus_embds_files=None):
        if corpus_embds:
            self.corpus_embds = corpus_embds
            self.corpus_embd_ds = EmptyDataset(corpus_embds)
        else:
            self.corpus_embd_ds = JSONLDataset(corpus_embds_files)
        # TODO: check this assertion later, removed for chunk model
        # assert len(self.corpus_embd_ds) == len(self.dataset.corpus)

    def queries_embd_dataloader(self):
        return torch.utils.data.DataLoader(
            self.queries_embd_ds,
            batch_size=self.embd_batch_size,
            num_workers=self.num_workers,
            collate_fn=EmbeddingDataCollator(embd_dtype=self.embd_dtype),
        )

    def corpus_embd_dataloader(self):
        return torch.utils.data.DataLoader(
            self.corpus_embd_ds,
            batch_size=self.embd_batch_size,
            num_workers=self.num_workers,
            collate_fn=EmbeddingDataCollator(embd_dtype=self.embd_dtype),
        )