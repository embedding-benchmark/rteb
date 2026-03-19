from __future__ import annotations
import base64
import io
import logging
from typing import Any, TYPE_CHECKING

import requests
from PIL import Image

from rteb.core.base import APIEmbeddingModel
from rteb.core.meta import ModelMeta
from rteb.utils.lazy_import import LazyImport

if TYPE_CHECKING:
    import voyageai
else:
    voyageai = LazyImport("voyageai")

logger = logging.getLogger(__name__)

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
    "voyage-law-2": 120_000,
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

voyage_law_2 = ModelMeta(
    loader=VoyageAIEmbeddingModel,
    model_name="voyage-law-2",
    embd_dtype="float32",
    embd_dim=1024,
    max_tokens=16_000,
    similarity="cosine",
    query_instruct=None,
    corpus_instruct=None,
    reference="https://docs.voyageai.com/docs/embeddings",
    vendor="Voyage AI",
)

MULTIMODAL_BATCH_TOKEN_LIMIT = 32_000
MULTIMODAL_BATCH_SIZE_LIMIT = 1000
TOKENS_PER_IMAGE = 3571
MAX_IMAGE_PIXELS = 4_000_000  # ~2000x2000; Voyage API rejects very large images


class VoyageMultimodalEmbeddingModel(APIEmbeddingModel):

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

    def forward(self, batch: dict[str, Any]) -> list[list[float]]:
        if "content" in batch:
            return self.embed(batch["content"], batch["input_type"][0])
        # Fallback: wrap text-only items as content lists
        content_lists = [[{"type": "text", "text": t}] for t in batch["text"]]
        return self.embed(content_lists, batch["input_type"][0])

    def embed(self, data: Any, input_type: str) -> list[list[float]]:
        sdk_inputs = [self._content_to_sdk_input(content) for content in data]
        result = self.client.multimodal_embed(
            inputs=sdk_inputs,
            model=self.model_name,
            input_type=input_type,
        )
        return result.embeddings

    @staticmethod
    def _cap_image_resolution(img: Image.Image) -> Image.Image:
        w, h = img.size
        if w * h > MAX_IMAGE_PIXELS:
            scale = (MAX_IMAGE_PIXELS / (w * h)) ** 0.5
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img

    @staticmethod
    def _content_to_sdk_input(content_list: list[dict]) -> list:
        sdk_items = []
        for item in content_list:
            item_type = item["type"]
            if item_type == "text":
                sdk_items.append(item["text"])
            elif item_type == "image_url":
                response = requests.get(item["url"], timeout=30)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
                sdk_items.append(VoyageMultimodalEmbeddingModel._cap_image_resolution(img))
            elif item_type == "image_base64":
                img_bytes = base64.b64decode(item["data"])
                img = Image.open(io.BytesIO(img_bytes))
                sdk_items.append(VoyageMultimodalEmbeddingModel._cap_image_resolution(img))
            elif item_type == "image_path":
                img = Image.open(item["path"])
                sdk_items.append(VoyageMultimodalEmbeddingModel._cap_image_resolution(img))
            elif item_type == "video_path":
                with open(item["path"], "rb") as vf:
                    video_data = vf.read()
                sdk_items.append(voyageai.video_utils.Video(data=video_data, model=item.get("model", "voyage-multimodal-3.5")))
            else:
                raise ValueError(f"Unknown content type: {item_type}")
        return sdk_items

    def build_token_batches(self, items: list[dict]) -> list[dict]:
        token_counts = []
        for item in items:
            if "content" in item:
                count = 0
                for piece in item["content"]:
                    if piece["type"] == "text":
                        count += len(piece["text"].split()) * 2
                    else:
                        count += TOKENS_PER_IMAGE
                token_counts.append(count)
            elif "text" in item:
                token_counts.append(len(item["text"].split()) * 2)
            else:
                token_counts.append(0)

        batches = []
        current_indices, current_tokens = [], 0
        for i, count in enumerate(token_counts):
            if current_indices and (current_tokens + count > MULTIMODAL_BATCH_TOKEN_LIMIT
                                    or len(current_indices) >= MULTIMODAL_BATCH_SIZE_LIMIT):
                batches.append(self._indices_to_batch(items, current_indices))
                current_indices, current_tokens = [], 0
            current_indices.append(i)
            current_tokens += count
        if current_indices:
            batches.append(self._indices_to_batch(items, current_indices))

        return batches

    @staticmethod
    def _indices_to_batch(items, indices):
        batch = {
            "id": [items[i]["id"] for i in indices],
            "input_type": [items[i]["input_type"] for i in indices],
        }
        contents = []
        for i in indices:
            item = items[i]
            if "content" in item:
                contents.append(item["content"])
            else:
                contents.append([{"type": "text", "text": item.get("text", "")}])
        batch["content"] = contents
        return batch

    @staticmethod
    def rate_limit_error_type() -> type:
        return voyageai.error.RateLimitError

    @staticmethod
    def service_error_type() -> type:
        return voyageai.error.ServiceUnavailableError


voyage_multimodal_35 = ModelMeta(
    loader=VoyageMultimodalEmbeddingModel,
    model_name="voyage-multimodal-3.5",
    embd_dtype="float32",
    embd_dim=1024,
    max_tokens=32_000,
    similarity="cosine",
    query_instruct=None,
    corpus_instruct=None,
    reference="https://docs.voyageai.com/docs/multimodal-embeddings",
    vendor="Voyage AI",
    leaderboards=["Multimodal"],
)
