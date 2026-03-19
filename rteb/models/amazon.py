from __future__ import annotations
from typing import Any, TYPE_CHECKING
import base64
import io
import json
import logging
import os
import time

import requests
from PIL import Image

from rteb.core.base import APIEmbeddingModel
from rteb.core.meta import ModelMeta
from rteb.utils.lazy_import import LazyImport

if TYPE_CHECKING:
    import boto3
    import botocore
else:
    boto3 = LazyImport("boto3")
    botocore = LazyImport("botocore")

logger = logging.getLogger(__name__)

AMAZON_BATCH_SIZE = 20  # No batch API; group for framework overhead management
AMAZON_MAX_IMAGE_PIXELS = 2048 * 2048  # Titan limit: ~4.2M pixels


class AmazonMultimodalEmbeddingModel(APIEmbeddingModel):

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
    def client(self):
        if not self._client:
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=os.environ.get("AWS_REGION", "us-east-1"),
            )
        return self._client

    def forward(self, batch: dict[str, Any]) -> list[list[float]]:
        if "content" in batch:
            return self.embed(batch["content"], batch["input_type"][0])
        content_lists = [[{"type": "text", "text": t}] for t in batch["text"]]
        return self.embed(content_lists, batch["input_type"][0])

    def embed(self, data: Any, input_type: str) -> list[list[float]]:
        all_embeddings = []
        for content_list in data:
            embedding = self._embed_single(content_list)
            all_embeddings.append(embedding)
        return all_embeddings

    @staticmethod
    def _cap_image(img_bytes: bytes) -> bytes:
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        if w * h > AMAZON_MAX_IMAGE_PIXELS:
            scale = (AMAZON_MAX_IMAGE_PIXELS / (w * h)) ** 0.5
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()
        return img_bytes

    def _embed_single(self, content_list: list[dict]) -> list[float]:
        body = {}
        for item in content_list:
            item_type = item["type"]
            if item_type == "text":
                body["inputText"] = item["text"]
            elif item_type == "image_url":
                response = requests.get(item["url"], timeout=30)
                response.raise_for_status()
                img_bytes = self._cap_image(response.content)
                body["inputImage"] = base64.b64encode(img_bytes).decode("utf-8")
            elif item_type == "image_base64":
                img_bytes = self._cap_image(base64.b64decode(item["data"]))
                body["inputImage"] = base64.b64encode(img_bytes).decode("utf-8")
            elif item_type == "image_path":
                with open(item["path"], "rb") as f:
                    img_bytes = self._cap_image(f.read())
                body["inputImage"] = base64.b64encode(img_bytes).decode("utf-8")
            else:
                raise ValueError(f"Unknown content type: {item_type}")

        num_tries = 0
        while not self._num_retries or num_tries < self._num_retries:
            try:
                num_tries += 1
                response = self.client.invoke_model(
                    modelId=self._model_meta.model_name,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body),
                )
                result = json.loads(response["body"].read())
                return result["embedding"]
            except Exception as e:
                logging.error(e)
                error_code = None
                if hasattr(e, "response") and isinstance(e.response, dict):
                    error_code = e.response.get("Error", {}).get("Code")
                if error_code == "ThrottlingException":
                    time.sleep(60)
                elif error_code and error_code.startswith("5"):
                    time.sleep(300)
                elif "connection" in type(e).__name__.lower() or "timeout" in type(e).__name__.lower():
                    time.sleep(30)
                else:
                    raise e
        raise Exception(f"Calling the API failed {num_tries} times")

    def build_token_batches(self, items: list[dict]) -> list[dict]:
        batches = []
        current_indices = []
        for i in range(len(items)):
            current_indices.append(i)
            if len(current_indices) >= AMAZON_BATCH_SIZE:
                batches.append(self._indices_to_batch(items, current_indices))
                current_indices = []
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


amazon_titan_embed_image_v1 = ModelMeta(
    loader=AmazonMultimodalEmbeddingModel,
    model_name="amazon.titan-embed-image-v1",
    embd_dtype="float32",
    embd_dim=1024,
    max_tokens=128,
    similarity="cosine",
    reference="https://docs.aws.amazon.com/bedrock/latest/userguide/titan-multiemb-models.html",
    vendor="Amazon",
    leaderboards=["Multimodal"],
)
