from __future__ import annotations
from typing import Any, TYPE_CHECKING

from rteb.core.base import APIEmbeddingModel
from rteb.core.meta import ModelMeta
from rteb.utils.lazy_import import LazyImport

if TYPE_CHECKING:
    import voyageai
else:
    voyageai = LazyImport("voyageai")


BATCH_TOKEN_LIMITS = {
    "voyage-4-large": 115_000,
    "voyage-4": 320_000,
    "voyage-4-lite": 1_000_000,
    "voyage-3.5": 320_000,
    "voyage-3.5-lite": 1_000_000,
    "voyage-3-large": 120_000,
    "voyage-code-3": 120_000,
    "voyage-3": 120_000,
    "voyage-2": 320_000,
}

BATCH_SIZE_LIMITS = {
    "voyage-4-large": 1000,
    "voyage-4": 1000,
    "voyage-4-lite": 1000,
}


class VoyageAIEmbeddingModel(APIEmbeddingModel):

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
    def client(self) -> voyageai.Client:
        if not self._client:
            self._client = voyageai.Client(api_key=self._api_key)
        return self._client

    def get_token_counts(self, texts: list[str]) -> list[int]:
        tokenized = self.client.tokenize(texts, model=self.model_name)
        return [len(t) for t in tokenized]

    def build_token_batches(self, items: list[dict]) -> list[dict]:
        texts = [item["text"] if item["text"] else " " for item in items]
        token_counts = self.get_token_counts(texts)
        token_limit = BATCH_TOKEN_LIMITS.get(self.model_name, 120_000)
        size_limit = BATCH_SIZE_LIMITS.get(self.model_name, 128)

        batches = []
        current_indices, current_tokens = [], 0
        for i, count in enumerate(token_counts):
            if current_indices and (current_tokens + count > token_limit or len(current_indices) >= size_limit):
                batches.append(self._indices_to_batch(items, current_indices))
                current_indices, current_tokens = [], 0
            current_indices.append(i)
            current_tokens += count
        if current_indices:
            batches.append(self._indices_to_batch(items, current_indices))

        return batches

    @staticmethod
    def _indices_to_batch(items, indices):
        return {
            "id": [items[i]["id"] for i in indices],
            "text": [items[i]["text"] for i in indices],
            "input_type": [items[i]["input_type"] for i in indices],
        }

    def embed(self, data: Any, input_type: str) -> list[list[float]]:
        request = {
            "texts": data,
            "model": self.model_name,
            "input_type": None
        }
        if self.model_name in ["voyage-3-large", "voyage-3.5", "voyage-3.5-lite", "voyage-code-3",
                                "voyage-4-large", "voyage-4", "voyage-4-lite"]:
            request["output_dimension"] = self.embd_dim

            output_dtype = self.embd_dtype
            if output_dtype == "float32":
                output_dtype = "float"
            request["output_dtype"] = output_dtype

        result = self.client.embed(
            **request
        )
        return result.embeddings

    @staticmethod
    def rate_limit_error_type() -> type:
        return voyageai.error.RateLimitError

    @staticmethod
    def service_error_type() -> type:
        return voyageai.error.ServiceUnavailableError

voyage_code_3 = ModelMeta(
    loader=VoyageAIEmbeddingModel,
    model_name="voyage-code-3",
    embd_dtype="float32",
    embd_dim=2048,
    max_tokens=32_000,
    similarity="cosine",
    query_instruct="Represent the query for retrieving supporting documents: ",
    corpus_instruct="Represent the document for retrieval: ",
    reference="https://docs.voyageai.com/docs/embeddings",
    vendor="Voyage AI",
    tooltip="Voyage's top model with retrieval instructions"
)

voyage_3 = ModelMeta(
    loader=VoyageAIEmbeddingModel,
    model_name="voyage-3",
    embd_dtype="float32",
    embd_dim=1024,
    max_tokens=32_000,
    similarity="cosine",
    query_instruct="Represent the query for retrieving supporting documents: ",
    corpus_instruct="Represent the document for retrieval: ",
    reference="https://docs.voyageai.com/docs/embeddings",
    vendor="Voyage AI",
    tooltip="Voyage's top model with retrieval instructions"
)

voyage_3_large = ModelMeta(
    loader=VoyageAIEmbeddingModel,
    model_name="voyage-3-large",
    embd_dtype="float32",
    embd_dim=2048,
    max_tokens=32_000,
    similarity="cosine",
    query_instruct="Represent the query for retrieving supporting documents: ",
    corpus_instruct="Represent the document for retrieval: ",
    reference="https://docs.voyageai.com/docs/embeddings",
    vendor="Voyage AI"
)

voyage_35 = ModelMeta(
    loader=VoyageAIEmbeddingModel,
    model_name="voyage-3.5",
    embd_dtype="float32",
    embd_dim=1024,
    max_tokens=32_000,
    similarity="cosine",
    query_instruct="Represent the query for retrieving supporting documents: ",
    corpus_instruct="Represent the document for retrieval: ",
    reference="https://docs.voyageai.com/docs/embeddings",
    vendor="Voyage AI",
    tooltip="Voyage's top model with retrieval instructions"
)

voyage_35_int8_512 = ModelMeta(
    loader=VoyageAIEmbeddingModel,
    model_name="voyage-3.5",
    alias="voyage-3.5 (int8, 512d)",
    embd_dtype="int8",
    embd_dim=512,
    max_tokens=32_000,
    similarity="cosine",
    query_instruct="Represent the query for retrieving supporting documents: ",
    corpus_instruct="Represent the document for retrieval: ",
    reference="https://docs.voyageai.com/docs/embeddings",
    vendor="Voyage AI",
    tooltip="Voyage's top model with retrieval instructions"
)

voyage_35_binary_256 = ModelMeta(
    loader=VoyageAIEmbeddingModel,
    model_name="voyage-3.5",
    alias="voyage-3.5 (binary, 256d)",
    embd_dtype="binary",
    embd_dim=256,
    max_tokens=32_000,
    similarity="cosine",
    query_instruct="Represent the query for retrieving supporting documents: ",
    corpus_instruct="Represent the document for retrieval: ",
    reference="https://docs.voyageai.com/docs/embeddings",
    vendor="Voyage AI",
    tooltip="Voyage's top model with retrieval instructions"
)

voyage_4 = ModelMeta(
    loader=VoyageAIEmbeddingModel,
    model_name="voyage-4",
    embd_dtype="float32",
    embd_dim=1024,
    max_tokens=32_000,
    similarity="cosine",
    query_instruct="Represent the query for retrieving supporting documents: ",
    corpus_instruct="Represent the document for retrieval: ",
    reference="https://docs.voyageai.com/docs/embeddings",
    vendor="Voyage AI"
)

voyage_4_large = ModelMeta(
    loader=VoyageAIEmbeddingModel,
    model_name="voyage-4-large",
    embd_dtype="float32",
    embd_dim=2048,
    max_tokens=32_000,
    similarity="cosine",
    query_instruct="Represent the query for retrieving supporting documents: ",
    corpus_instruct="Represent the document for retrieval: ",
    reference="https://docs.voyageai.com/docs/embeddings",
    vendor="Voyage AI"
)

voyage_4_lite = ModelMeta(
    loader=VoyageAIEmbeddingModel,
    model_name="voyage-4-lite",
    embd_dtype="float32",
    embd_dim=1024,
    max_tokens=32_000,
    similarity="cosine",
    query_instruct="Represent the query for retrieving supporting documents: ",
    corpus_instruct="Represent the document for retrieval: ",
    reference="https://docs.voyageai.com/docs/embeddings",
    vendor="Voyage AI"
)
