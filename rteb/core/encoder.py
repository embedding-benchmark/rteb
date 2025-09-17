import os
import json
import logging
from typing import List

from numpy import ndarray
from pytorch_lightning import LightningModule

from rteb.core.base import EmbeddingModel
from rteb.utils.data import JSONLDataset
from rteb.utils.distributed import gather_list

logger = logging.getLogger(__name__)


class Encoder(LightningModule):

    def __init__(
            self,
            model: EmbeddingModel,
            load_embds: bool = False,
            **kwargs,
    ):
        super().__init__()
        self._model = model
        self._load_embds = load_embds
        # Always save embeddings to disk and never keep in memory for memory efficiency
        self.in_memory = False
        self.is_query = False
        self.save_file = None
        self.expected_dataset_size = None  # Will be set before encoding

    @property
    def model(self) -> EmbeddingModel:
        return self._model

    @property
    def load_embds(self) -> bool:
        return self._load_embds

    @property
    def save_embds(self) -> bool:
        # Always save embeddings to disk for memory efficiency
        return True

    @property
    def local_embd_file_name(self) -> str:
        assert self.save_file is not None
        num_shards = self.trainer.num_devices
        return f"{self.save_file}-{self.local_rank}-of-{num_shards}"

    def get_local_embd_files(self, num_shards=None) -> List[str]:
        # Return local (intermediate) file names, which are jsonl files
        assert self.save_file is not None
        if num_shards is None:
            num_shards = self.trainer.num_devices
        return [f"{self.save_file}-{i}-of-{num_shards}" for i in range(num_shards)]

    def get_embd_files(self, num_shards=None) -> List[str]:
        # Return the final file names, which are arrow files
        local_files = self.get_local_embd_files(num_shards=num_shards)
        return local_files

    def embd_files_exist(self, num_shards=None) -> bool:
        files = self.get_embd_files(num_shards=num_shards)
        return all(os.path.exists(file) for file in files)


    def load_existing_embeddings(self, num_devices=1):
        """Load existing embeddings for resumable processing (can be called before trainer attachment)."""
        if self.load_embds:
            self.local_existing_ids = set()
            # Get file name with explicit num_devices parameter
            embd_file_name = f"{self.save_file}-{0}-of-{num_devices}"
            if os.path.exists(embd_file_name):
                logger.warning(f"Load embeddings from {embd_file_name}")
                ds = JSONLDataset(embd_file_name)
                for example in ds:
                    self.local_existing_ids.add(example["id"])
                
                # Validate loaded embeddings and enable resumable processing
                if self.expected_dataset_size is not None:
                    loaded_count = len(self.local_existing_ids)
                    if loaded_count == self.expected_dataset_size:
                        logger.info(f"All {loaded_count} {'queries' if self.is_query else 'corpus'} embeddings already exist - skipping encoding")
                    elif loaded_count < self.expected_dataset_size:
                        missing_count = self.expected_dataset_size - loaded_count
                        logger.info(f"Resumable processing: {loaded_count}/{self.expected_dataset_size} "
                                  f"{'queries' if self.is_query else 'corpus'} embeddings found. "
                                  f"Will generate {missing_count} missing embeddings.")
                    else:
                        logger.warning(f"More embeddings found ({loaded_count}) than expected ({self.expected_dataset_size}). "
                                     f"This may indicate a mismatch in dataset or embedding files.")
            else:
                logger.warning(f"load_embds is True but {embd_file_name} doesn't exist. Skipping the loading.")

    def on_predict_epoch_start(self):
        self.embds = None

        if self.in_memory:
            self.local_embds = []

        # Load existing embeddings if not already loaded
        if self.load_embds and not hasattr(self, 'local_existing_ids'):
            self.load_existing_embeddings(self.trainer.num_devices)

        # Don't open file yet - will open lazily when first write is needed
        self.local_embd_file = None

    def _ensure_file_open(self):
        """Lazily open the embedding file only when we need to write data."""
        if self.save_embds and self.local_embd_file is None:
            if self.load_embds:
                # append to the file
                self.local_embd_file = open(self.local_embd_file_name, "a")
                logger.debug(f"Opened embedding file for appending: {self.local_embd_file_name}")
            else:
                # rewrite the file
                self.local_embd_file = open(self.local_embd_file_name, "w")
                logger.debug(f"Opened embedding file for writing: {self.local_embd_file_name}")

    def predict_step(self, batch, batch_idx):
        indices = batch["id"]

        if self.load_embds and self.local_existing_ids:
            # Check if all items in this batch are already processed
            masks = [id in self.local_existing_ids for id in indices]
            num_existed = sum(masks)
            if num_existed == len(indices):
                return  # Skip entire batch - all items already processed
            # Note: Partial batches are handled by pre-filtering the dataset

        embds = self._model(batch)

        for idx, embd in zip(indices, embds):
            embd_list = embd
            if isinstance(embd, ndarray):
                embd_list = embd.tolist()
            obj = {"id": idx, "embd": embd_list}
            # Always save to disk, never keep in memory for memory efficiency
            if self.save_embds:
                self._ensure_file_open()  # Lazy file opening
                self.local_embd_file.write(json.dumps(obj) + "\n")

    def on_predict_epoch_end(self):
        if self.save_embds and self.local_embd_file is not None:
            self.local_embd_file.close()
            logger.debug(f"Closed embedding file: {self.local_embd_file_name}")
        if self.in_memory:
            self.embds = gather_list(self.local_embds, self.trainer.num_devices)
        self.trainer.strategy.barrier()

    def offload_model(self):
        """Offload the model to free memory after encoding is complete."""
        if self.model is not None and hasattr(self.model, "offload"):
            self.model.offload()
        else:
            logger.info("No model to offload")
