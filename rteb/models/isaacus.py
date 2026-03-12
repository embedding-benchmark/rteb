from __future__ import annotations
from typing import Any, TYPE_CHECKING
import time
import logging

from rteb.core.base import APIEmbeddingModel
from rteb.core.meta import ModelMeta
from rteb.utils.lazy_import import LazyImport

if TYPE_CHECKING:
    import isaacus
else:
    isaacus = LazyImport("isaacus")


class IsaacusEmbeddingModel(APIEmbeddingModel):

    def __init__(
        self,
        model_meta: ModelMeta,
        api_key: str | None = None,
        num_retries: int | None = None,
        **kwargs
    ):
        super().__init__(
            model_meta,
            api_key=api_key,
            num_retries=num_retries,
            **kwargs
        )
        self._client = None

    @property
    def client(self) -> isaacus.Isaacus:
        if not self._client:
            self._client = isaacus.Isaacus(api_key=self._api_key)
        return self._client

    def _truncate_text(self, text: str, max_chars: int = 60000) -> str:
        """Rough truncation to stay under the API size limit."""
        if len(text) > max_chars:
            return text[:max_chars]
        return text

    def embed(self, data: list[str], input_type: str) -> list[list[float]]:
        task = "retrieval/query" if input_type == "query" else "retrieval/document"
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                texts=data,
                task=task,
                dimensions=self.embd_dim,
            )
            return [item.embedding for item in response.embeddings]
        except isaacus.APIStatusError as e:
            if e.status_code == 413:
                # Batch too large — truncate long texts and retry
                print(f"413: truncating {len(data)} texts and retrying")
                truncated = [self._truncate_text(t) for t in data]
                response = self.client.embeddings.create(
                    model=self.model_name,
                    texts=truncated,
                    task=task,
                    dimensions=self.embd_dim,
                )
                return [item.embedding for item in response.embeddings]
            raise

    def forward(self, batch: dict[str, Any]) -> list[list[float]]:
        num_tries = 0
        while not self._num_retries or num_tries < self._num_retries:
            try:
                num_tries += 1
                result = self.embed(batch["text"], batch["input_type"][0])
                return result
            except Exception as e:
                logging.error(e)
                if isinstance(e, isaacus.RateLimitError):
                    print("Rate limit hit, sleeping 60s")
                    time.sleep(60)
                elif isinstance(e, isaacus.APIStatusError):
                    print(f"API error {e.status_code}, sleeping 300s")
                    time.sleep(300)
                else:
                    raise e
        raise Exception(f"Calling the Isaacus API failed {num_tries} times")


kanon_2_embedder = ModelMeta(
    loader=IsaacusEmbeddingModel,
    model_name="kanon-2-embedder",
    embd_dtype="float32",
    embd_dim=1792,
    max_tokens=16384,
    similarity="cosine",
    reference="https://docs.isaacus.com/embeddings",
    vendor="Isaacus",
)
