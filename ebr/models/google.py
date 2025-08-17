

from __future__ import annotations
from typing import Any, TYPE_CHECKING
import time
import logging

from ebr.core.base import APIEmbeddingModel
from ebr.core.meta import ModelMeta

from google import genai
from google.genai.types import EmbedContentConfig
import vertexai
from vertexai.language_models import TextEmbeddingModel
from google.oauth2 import service_account

USE_VERTEX = True


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
)

# gemini_embedding_001 = ModelMeta(
#     loader=GoogleEmbeddingModel,
#     model_name="gemini-embedding-001",
#     embd_dtype="float32",
#     embd_dim=3072,
#     max_tokens=2048,
#     similarity="cosine",
#     reference="https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/get-text-embeddings",
#     vendor="Google",
# )
