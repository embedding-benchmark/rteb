from __future__ import annotations
from typing import Any, TYPE_CHECKING
import base64
import io
import logging

import requests

from rteb.core.meta import ModelMeta
from rteb.core.base import APIEmbeddingModel
from rteb.utils.lazy_import import LazyImport

if TYPE_CHECKING:
    import cohere
else:
    cohere = LazyImport("cohere")

COHERE_BATCH_SIZE_LIMIT = 96  # API limit: 96 total items per request
COHERE_IMAGE_BATCH_SIZE = 3  # With 5MB cap per image, 3 images stays safely under 20MB total
COHERE_MAX_IMAGE_BYTES = 5_000_000  # 5MB per image (API limit)


class CohereEmbeddingModel(APIEmbeddingModel):

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
    def client(self) -> cohere.ClientV2:
        if not self._client:
            self._client = cohere.ClientV2(api_key=self._api_key)
        return self._client

    @property
    def embedding_type(self) -> str:
        if self.embd_dtype == "float32":
            return "float"
        else:
            return self.embd_dtype

    def embed(self, data: str, input_type: str) -> list[list[float]]:
        
        return getattr(self.client.embed(
            model=self.model_name,
            texts=data,
            input_type="search_query" if input_type == "query" else "search_document",
            embedding_types=[self.embedding_type]
        ).embeddings, self.embedding_type)

    @staticmethod
    def rate_limit_error_type() -> type:
        return cohere.errors.too_many_requests_error.TooManyRequestsError


embed_multilingual_v3_0 = ModelMeta(
    loader=CohereEmbeddingModel,
    model_name="embed-multilingual-v3.0",
    embd_dtype="float32",
    embd_dim=1024,
    max_tokens=512,
    similarity="cosine",
    reference="https://docs.cohere.com/v2/docs/cohere-embed",
    vendor="Cohere",
    tooltip="Cohere's 1024d multilingual embedding model"
)

embed_v4_0 = ModelMeta(
    loader=CohereEmbeddingModel,
    model_name="embed-v4.0",
    embd_dtype="float32",
    embd_dim=1536,
    max_tokens=128_000,
    similarity="cosine",
    reference="https://docs.cohere.com/v2/docs/cohere-embed",
    vendor="Cohere",
    tooltip="Cohere's latest 1536d model with 128K context"
)

embed_v4_0_int8_512 = ModelMeta(
    loader=CohereEmbeddingModel,
    model_name="embed-v4.0",
    alias="embed-v4.0 (int8, 512d)",
    embd_dtype="int8",
    embd_dim=512,
    max_tokens=128_000,
    similarity="cosine",
    reference="https://docs.cohere.com/v2/docs/cohere-embed",
    vendor="Cohere",
    tooltip="Cohere's latest 1536d model with 128K context"
)

embed_v4_0_binary_256 = ModelMeta(
    loader=CohereEmbeddingModel,
    model_name="embed-v4.0",
    alias="embed-v4.0 (binary, 256d)",
    embd_dtype="binary",
    embd_dim=256,
    max_tokens=128_000,
    similarity="cosine",
    reference="https://docs.cohere.com/v2/docs/cohere-embed",
    vendor="Cohere",
    tooltip="Cohere's latest 1536d model with 128K context"
)


class CohereMultimodalEmbeddingModel(APIEmbeddingModel):

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
    def client(self) -> cohere.ClientV2:
        if not self._client:
            self._client = cohere.ClientV2(api_key=self._api_key)
        return self._client

    def forward(self, batch: dict[str, Any]) -> list[list[float]]:
        if "content" in batch:
            return self.embed(batch["content"], batch["input_type"][0])
        content_lists = [[{"type": "text", "text": t}] for t in batch["text"]]
        return self.embed(content_lists, batch["input_type"][0])

    def embed(self, data: Any, input_type: str) -> list[list[float]]:
        sdk_inputs = [self._content_to_cohere_input(content) for content in data]
        result = self.client.embed(
            model="embed-v4.0",
            inputs=sdk_inputs,
            input_type="search_query" if input_type == "query" else "search_document",
            embedding_types=["float"],
            output_dimension=self.embd_dim,
        )
        return result.embeddings.float

    @staticmethod
    def _cap_image(img_bytes: bytes, mime_type: str) -> bytes:
        if len(img_bytes) <= COHERE_MAX_IMAGE_BYTES:
            return img_bytes
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        for quality in [85, 60, 40, 20]:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            if buf.tell() <= COHERE_MAX_IMAGE_BYTES:
                return buf.getvalue()
        # Last resort: downscale
        while True:
            w, h = img.size
            img = img.resize((w // 2, h // 2), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=60)
            if buf.tell() <= COHERE_MAX_IMAGE_BYTES:
                return buf.getvalue()

    @staticmethod
    def _image_bytes_to_data_uri(img_bytes: bytes, mime_type: str = "image/jpeg") -> str:
        img_bytes = CohereMultimodalEmbeddingModel._cap_image(img_bytes, mime_type)
        encoded = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _content_to_cohere_input(content_list: list[dict]) -> dict:
        parts = []
        for item in content_list:
            item_type = item["type"]
            if item_type == "text":
                parts.append({"type": "text", "text": item["text"]})
            elif item_type == "image_url":
                response = requests.get(item["url"], timeout=30)
                response.raise_for_status()
                mime = response.headers.get("Content-Type", "image/jpeg")
                data_uri = CohereMultimodalEmbeddingModel._image_bytes_to_data_uri(response.content, mime)
                parts.append({"type": "image_url", "image_url": {"url": data_uri}})
            elif item_type == "image_base64":
                mime = item.get("mime_type", "image/jpeg")
                img_bytes = base64.b64decode(item["data"])
                data_uri = CohereMultimodalEmbeddingModel._image_bytes_to_data_uri(img_bytes, mime)
                parts.append({"type": "image_url", "image_url": {"url": data_uri}})
            elif item_type == "image_path":
                mime = "image/png" if item["path"].endswith(".png") else "image/jpeg"
                with open(item["path"], "rb") as f:
                    img_bytes = f.read()
                data_uri = CohereMultimodalEmbeddingModel._image_bytes_to_data_uri(img_bytes, mime)
                parts.append({"type": "image_url", "image_url": {"url": data_uri}})
            else:
                raise ValueError(f"Unknown content type: {item_type}")
        return {"content": parts}

    def build_token_batches(self, items: list[dict]) -> list[dict]:
        batches = []
        current_indices = []
        for i, item in enumerate(items):
            has_image = "content" in item and any(
                p["type"] != "text" for p in item["content"]
            )
            if current_indices and (
                len(current_indices) >= COHERE_BATCH_SIZE_LIMIT
                or (has_image and len(current_indices) >= COHERE_IMAGE_BATCH_SIZE)
            ):
                batches.append(self._indices_to_batch(items, current_indices))
                current_indices = []
            current_indices.append(i)
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
        return cohere.errors.too_many_requests_error.TooManyRequestsError


embed_v4_0_multimodal = ModelMeta(
    loader=CohereMultimodalEmbeddingModel,
    model_name="embed-v4.0-multimodal",
    alias="embed-v4.0 (multimodal)",
    embd_dtype="float32",
    embd_dim=1536,
    max_tokens=128_000,
    similarity="cosine",
    reference="https://docs.cohere.com/v2/docs/cohere-embed",
    vendor="Cohere",
)
