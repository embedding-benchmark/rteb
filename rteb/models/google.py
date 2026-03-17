

from __future__ import annotations
from typing import Any, TYPE_CHECKING
import base64
import io
import time
import logging

import requests

from rteb.core.base import APIEmbeddingModel
from rteb.core.meta import ModelMeta

from google import genai
from google.genai import types
from google.genai.types import EmbedContentConfig
import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.oauth2 import service_account

import os
USE_VERTEX = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "True").lower() == "true"

GOOGLE_MULTIMODAL_BATCH_LIMIT = 6  # API limit: 6 images per request
GOOGLE_BATCH_SIZE_LIMIT = 100  # API limit: 100 items per batch
MAX_IMAGE_PIXELS = 4_000_000


class GoogleEmbeddingModel(APIEmbeddingModel):
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
        if USE_VERTEX:
            credentials = service_account.Credentials.from_service_account_file("/Users/fodizoltan/Downloads/gemini-430116-95a3f0ae96f3.json")
            vertexai.init(credentials=credentials, project="gemini-430116")

    @property
    def client(self) -> genai.Client:
        if not self._client:
            print("Initializing the client")
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def embed(self, data: Any, input_type: str) -> list[list[float]]:
        data = [text if len(text) > 0 else 'none' for text in data ]
        if USE_VERTEX:
            model = TextEmbeddingModel.from_pretrained(self._model_meta.model_name)
            embeddings = model.get_embeddings(
                texts=data,
                auto_truncate=True,
            )
            return [embedding.values for embedding in embeddings]
        else:
            response = self.client.models.embed_content(
                model=self._model_meta.model_name,
                contents=data,
                config=EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY" if input_type == "query" else "RETRIEVAL_DOCUMENT",
                    output_dimensionality=self.embd_dim,
                ),
            )
            return [embedding.values for embedding in response.embeddings]

    def forward(self, batch: dict[str, Any]) -> list[list[float]]:
        num_tries = 0
        while not self._num_retries or num_tries < self._num_retries:
            try:
                num_tries += 1
                result = self.embed(batch["text"], batch["input_type"][0])
                return result
            except Exception as e:
                logging.error(e)
                if hasattr(e, "code"):
                    if str(e.code) == "429":
                        print("RLE")
                        time.sleep(60)
                    elif str(e.code) >= "500":
                        print("Other error")
                        time.sleep(300)
                    else:
                        raise e
                else:
                    raise e
        raise Exception(f"Calling the API failed {num_tries} times")


text_embedding_004 = ModelMeta(
    loader=GoogleEmbeddingModel,
    model_name="text-embedding-004",
    embd_dtype="float32",
    embd_dim=768,
    max_tokens=2048,
    similarity="cosine",
    reference="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    vendor="Google",
    tooltip="Google's latest 768d embedding model"
)

gemini_embedding_001 = ModelMeta(
    loader=GoogleEmbeddingModel,
    model_name="gemini-embedding-001",
    embd_dtype="float32",
    embd_dim=3072,
    max_tokens=2048,
    similarity="cosine",
    reference="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
    vendor="Google",
)


class GoogleMultimodalEmbeddingModel(APIEmbeddingModel):

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
    def client(self) -> genai.Client:
        if not self._client:
            if USE_VERTEX:
                credentials = service_account.Credentials.from_service_account_file(
                    "/Users/fodizoltan/Downloads/gemini-430116-95a3f0ae96f3.json"
                )
                self._client = genai.Client(
                    vertexai=True,
                    project="gemini-430116",
                    location="us-central1",
                    credentials=credentials,
                )
            else:
                self._client = genai.Client(api_key=self._api_key)
        return self._client

    def forward(self, batch: dict[str, Any]) -> list[list[float]]:
        if "content" in batch:
            return self.embed(batch["content"], batch["input_type"][0])
        content_lists = [[{"type": "text", "text": t}] for t in batch["text"]]
        return self.embed(content_lists, batch["input_type"][0])

    def embed(self, data: Any, input_type: str) -> list[list[float]]:
        num_tries = 0
        while not self._num_retries or num_tries < self._num_retries:
            try:
                num_tries += 1
                sdk_contents = [self._content_to_sdk(content) for content in data]
                task_type = "RETRIEVAL_QUERY" if input_type == "query" else "RETRIEVAL_DOCUMENT"
                response = self.client.models.embed_content(
                    model=self._model_meta.model_name,
                    contents=sdk_contents,
                    config=EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.embd_dim,
                    ),
                )
                return [embedding.values for embedding in response.embeddings]
            except Exception as e:
                logging.error(e)
                if hasattr(e, "code"):
                    if str(e.code) == "429":
                        time.sleep(60)
                    elif str(e.code) >= "500":
                        time.sleep(300)
                    else:
                        raise e
                else:
                    raise e
        raise Exception(f"Calling the API failed {num_tries} times")

    @staticmethod
    def _cap_image_resolution(img_bytes: bytes, mime_type: str) -> bytes:
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        if w * h > MAX_IMAGE_PIXELS:
            scale = (MAX_IMAGE_PIXELS / (w * h)) ** 0.5
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            fmt = "JPEG" if "jpeg" in mime_type or "jpg" in mime_type else "PNG"
            img.save(buf, format=fmt)
            return buf.getvalue()
        return img_bytes

    @staticmethod
    def _content_to_sdk(content_list: list[dict]) -> list:
        parts = []
        for item in content_list:
            item_type = item["type"]
            if item_type == "text":
                parts.append(item["text"])
            elif item_type == "image_url":
                response = requests.get(item["url"], timeout=30)
                response.raise_for_status()
                mime = response.headers.get("Content-Type", "image/jpeg")
                img_bytes = GoogleMultimodalEmbeddingModel._cap_image_resolution(response.content, mime)
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
            elif item_type == "image_base64":
                img_bytes = base64.b64decode(item["data"])
                mime = item.get("mime_type", "image/jpeg")
                img_bytes = GoogleMultimodalEmbeddingModel._cap_image_resolution(img_bytes, mime)
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
            elif item_type == "image_path":
                mime = "image/png" if item["path"].endswith(".png") else "image/jpeg"
                with open(item["path"], "rb") as f:
                    img_bytes = f.read()
                img_bytes = GoogleMultimodalEmbeddingModel._cap_image_resolution(img_bytes, mime)
                parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
            else:
                raise ValueError(f"Unknown content type: {item_type}")
        return parts

    def build_token_batches(self, items: list[dict]) -> list[dict]:
        batches = []
        current_indices = []
        current_image_count = 0
        for i, item in enumerate(items):
            has_image = False
            if "content" in item:
                has_image = any(p["type"] != "text" for p in item["content"])
            new_image_count = current_image_count + (1 if has_image else 0)
            if current_indices and (
                len(current_indices) >= GOOGLE_BATCH_SIZE_LIMIT
                or new_image_count > GOOGLE_MULTIMODAL_BATCH_LIMIT
            ):
                batches.append(self._indices_to_batch(items, current_indices))
                current_indices = []
                current_image_count = 0
                new_image_count = 1 if has_image else 0
            current_indices.append(i)
            current_image_count = new_image_count
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


gemini_embedding_2 = ModelMeta(
    loader=GoogleMultimodalEmbeddingModel,
    model_name="gemini-embedding-2-preview",
    embd_dtype="float32",
    embd_dim=3072,
    max_tokens=8192,
    similarity="cosine",
    reference="https://ai.google.dev/gemini-api/docs/embeddings",
    vendor="Google",
)
