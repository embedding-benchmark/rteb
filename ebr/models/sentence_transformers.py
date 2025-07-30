import logging

from ebr.core.base import EmbeddingModel
from ebr.utils.lazy_import import LazyImport
from ebr.core.meta import ModelMeta
from ebr.utils.memory import force_garbage_collection

SentenceTransformer = LazyImport("sentence_transformers", attribute="SentenceTransformer")

logger = logging.getLogger(__name__)


class SentenceTransformersEmbeddingModel(EmbeddingModel):

    def __init__(
        self,
        model_meta: ModelMeta,
        device: str = None,
        **kwargs
    ):
        super().__init__(model_meta, **kwargs)
        self.device = device
        self.__model = None  # Initialize for lazy loading

    def embed(self, data: str, input_type: str) -> list[list[float]]:
        return self._model.encode(data)

    @property
    def model_name_prefix(self) -> str:
        return "sentence-transformers"

    @property
    def _model(self):
        if self.__model is None:
            logger.info("Loading in the SentenceTransformer model...")
            self.__model = SentenceTransformer(
                f"{self.model_name_prefix}/{self.model_name}",
                device=self.device,
                trust_remote_code=True
            )
        return self.__model

    @property
    def _id(self) -> str:
        return f"{self.model_name_prefix}__{self._model_meta._id}"

    def offload(self) -> None:
        """Offload the SentenceTransformer model to free memory."""
        if self.__model is not None:
            logger.info("Offloading SentenceTransformer model...")
            
            # Get memory before offloading for reporting
            import torch
            memory_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            
            # Delete the model
            del self.__model
            self.__model = None
            
            # Force comprehensive garbage collection
            force_garbage_collection()
            
            # Report memory savings
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                memory_saved = memory_before - memory_after
                logger.info(f"Model offloaded successfully, saved {memory_saved / 1024**3:.1f} GB")
            else:
                logger.info("Model offloaded successfully")


class NvidiaEmbeddingModel(SentenceTransformersEmbeddingModel):
    @property
    def model_name_prefix(self) -> str:
        return "nvidia"


class E5EmbeddingModel(SentenceTransformersEmbeddingModel):
    @property
    def model_name_prefix(self) -> str:
        return "intfloat"


class JinaEmbeddingModel(SentenceTransformersEmbeddingModel):
    @property
    def model_name_prefix(self) -> str:
        return "jinaai"


class QwenEmbeddingModel(SentenceTransformersEmbeddingModel):
    @property
    def model_name_prefix(self) -> str:
        return "Qwen"


NV_Embed_v2 = ModelMeta(
    loader=NvidiaEmbeddingModel,
    model_name="NV-Embed-v2",
    embd_dtype="float32",
    embd_dim=4096,
    num_params=7_850_000_000,
    max_tokens=32768,
    similarity="cosine",
    reference="https://huggingface.co/nvidia/NV-Embed-v2"
)


qwen3_embedding_06B = ModelMeta(
    loader=QwenEmbeddingModel,
    model_name="Qwen3-Embedding-0.6B",
    embd_dtype="float32",
    embd_dim=1024,
    max_tokens=32768,
    num_params=600_000_000,
    similarity="cosine",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-0.6B",
    vendor="Alibaba",
)

qwen3_embedding_8B = ModelMeta(
    loader=QwenEmbeddingModel,
    model_name="Qwen3-Embedding-8B",
    embd_dtype="float32",
    embd_dim=4096,
    max_tokens=32768,
    num_params=8_000_000_000,
    similarity="cosine",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-8B",
    vendor="Alibaba",
)


e5_mistral_7b_instruct = ModelMeta(
    loader=E5EmbeddingModel,
    model_name="e5-mistral-7b-instruct",
    embd_dtype="float32",
    embd_dim=4096,
    similarity="cosine",
    reference="https://huggingface.co/intfloat/e5-mistral-7b-instruct",
    vendor="Microsoft"
)


all_MiniLM_L6_v2 = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="all-MiniLM-L6-v2",
    embd_dtype="float32",
    embd_dim=384,
    num_params=22_700_000,
    max_tokens=256,
    similarity="cosine",
    reference="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
)


all_MiniLM_L12_v2 = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="all-MiniLM-L12-v2",
    embd_dtype="float32",
    embd_dim=384,
    num_params=33_400_000,
    max_tokens=256,
    similarity="cosine",
    reference="https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2"
)


labse = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="LaBSE",
    embd_dtype="float32",
    embd_dim=768,
    num_params=471_000_000,
    max_tokens=512,
    similarity="cosine",
    reference="https://huggingface.co/sentence-transformers/LaBSE",
    vendor="Google",
)


multi_qa_MiniLM_L6_cos_v1 = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="multi-qa-MiniLM-L6-cos-v1",
    embd_dtype="float32",
    embd_dim=384,
    num_params=22_700_000,
    max_tokens=512,
    similarity="cosine",
    reference="https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)


all_mpnet_base_v2 = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="all-mpnet-base-v2",
    embd_dtype="float32",
    embd_dim=768,
    num_params=109_000_000,
    max_tokens=384,
    similarity="cosine",
    reference="https://huggingface.co/sentence-transformers/all-mpnet-base-v2"
)


jina_embeddings_v2_base_en = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="jina-embeddings-v2-base-en",
    embd_dtype="float32",
    embd_dim=768,
    num_params=137_000_000,
    max_tokens=8192,
    similarity="cosine",
    reference="https://huggingface.co/jinaai/jina-embeddings-v2-base-en",
    vendor="Jina AI",
)


jina_embeddings_v2_small_en = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="jina-embeddings-v2-small-en",
    embd_dtype="float32",
    embd_dim=512,
    num_params=32_700_000,
    max_tokens=8192,
    similarity="cosine",
    reference="https://huggingface.co/jinaai/jina-embeddings-v2-small-en",
    vendor="Jina AI",
)


