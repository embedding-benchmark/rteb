import logging

import torch
from rteb.core.base import EmbeddingModel
from rteb.utils.lazy_import import LazyImport
from rteb.core.meta import ModelMeta
from rteb.utils.memory import force_garbage_collection

SentenceTransformer = LazyImport("sentence_transformers", attribute="SentenceTransformer")

logger = logging.getLogger(__name__)

_sdpa_patched = False


def _patch_sdpa_for_embedding():
    """Monkey-patch SDPA attention to work with GQA models in fp32.

    The default SDPA path fails for GQA models (e.g. Qwen3/Octen with 32 Q heads
    vs 8 K/V heads) because the memory-efficient kernel requires matching head counts,
    and the 4D causal mask forces fallback to the O(n²) math backend.

    This patch:
    1. Expands K/V heads to match Q heads via repeat_kv
    2. Drops the 4D attention mask (avoids O(n²) mask allocation)
    3. Uses is_causal from the module (preserves causal attention for decoder models)
    """
    global _sdpa_patched
    if _sdpa_patched:
        return

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv

    def patched_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
        if hasattr(module, 'num_key_value_groups'):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        # Use the module's is_causal setting (True for decoder-based models like Qwen3)
        # The SDPA kernel generates the causal mask internally without O(n²) memory
        causal = getattr(module, 'is_causal', False) if is_causal is None else is_causal
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=dropout, scale=scaling, is_causal=causal
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, None

    ALL_ATTENTION_FUNCTIONS['sdpa'] = patched_sdpa
    _sdpa_patched = True
    logger.info("Patched SDPA attention for GQA models (fp32 memory-efficient)")


class SentenceTransformersEmbeddingModel(EmbeddingModel):

    def __init__(
        self,
        model_meta: ModelMeta,
        device: str = None,
        max_seq_length: int = None,
        cap_seq_length: bool = True,
        device_map: bool = False,
        **kwargs
    ):
        super().__init__(model_meta, **kwargs)
        self._device_str = device
        self._max_seq_length = max_seq_length
        self._cap_seq_length = cap_seq_length
        self._device_map = device_map
        self.__model = None  # Initialize for lazy loading

    def embed(self, data: str, input_type: str) -> list[list[float]]:
        return self._model.encode(data)

    @property
    def model_name_prefix(self) -> str:
        return "sentence-transformers"

    @property
    def _model(self):
        if self.__model is None:
            # Load outside inference_mode to avoid "Inference tensors do not track
            # version counter" errors when DDP syncs module states between predict calls.
            with torch.inference_mode(False):
                logger.info("Loading in the SentenceTransformer model...")
                _patch_sdpa_for_embedding()
                model_kwargs = {"attn_implementation": "sdpa"}
                logger.info("Using PyTorch SDPA for memory-efficient attention")
                if self._device_map:
                    # Load model in bf16 to reduce memory, skip SDPA patch
                    # which adds overhead. With bf16 weights (~24GB for 12B model)
                    # there should be enough room for activations on 80GB GPU.
                    model_kwargs_dm = {
                        "torch_dtype": torch.bfloat16,
                        "attn_implementation": "flash_attention_2",
                    }
                    logger.info("Using flash_attention_2 + bf16 for memory-efficient inference")
                    self.__model = SentenceTransformer(
                        f"{self.model_name_prefix}/{self.model_name}",
                        device="cuda:0",
                        trust_remote_code=True,
                        model_kwargs=model_kwargs_dm
                    )
                else:
                    # Resolve the correct device for DDP: each process should load on its own GPU
                    device = self._device_str
                    if device == "cuda" and torch.cuda.is_available():
                        device = f"cuda:{torch.cuda.current_device()}"
                        logger.info(f"DDP: loading model on {device}")
                    self.__model = SentenceTransformer(
                        f"{self.model_name_prefix}/{self.model_name}",
                        device=device,
                        trust_remote_code=True,
                        model_kwargs=model_kwargs
                    )
                if self._max_seq_length is not None:
                    logger.info(f"Overriding max_seq_length: {self.__model.max_seq_length} -> {self._max_seq_length}")
                    self.__model.max_seq_length = self._max_seq_length
                elif self._cap_seq_length and self._model_meta.max_tokens:
                    logger.info(f"Capping max_seq_length to model max_tokens: {self.__model.max_seq_length} -> {self._model_meta.max_tokens}")
                    self.__model.max_seq_length = self._model_meta.max_tokens
        return self.__model

    @property
    def _id(self) -> str:
        return f"{self.model_name_prefix}__{self._model_meta._id}"

    def ensure_loaded(self) -> None:
        """Eagerly load the model outside of inference_mode to avoid inference tensor errors."""
        _ = self._model

    def offload(self) -> None:
        """Offload the SentenceTransformer model to free memory."""
        if self.__model is not None:
            logger.info("Offloading SentenceTransformer model...")

            # Get memory before offloading for reporting
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


class OctenEmbeddingModel(SentenceTransformersEmbeddingModel):
    @property
    def model_name_prefix(self) -> str:
        return "Octen"


class TencentEmbeddingModel(SentenceTransformersEmbeddingModel):
    @property
    def model_name_prefix(self) -> str:
        return "tencent"


NV_Embed_v2 = ModelMeta(
    loader=NvidiaEmbeddingModel,
    model_name="NV-Embed-v2",
    embd_dtype="float32",
    embd_dim=4096,
    num_params=7_850_000_000,
    max_tokens=32768,
    similarity="cosine",
    reference="https://huggingface.co/nvidia/NV-Embed-v2",
    vendor="NVidia",
    tooltip="High-performance 4096d model with 32K context, 7.8B params"
)


qwen3_embedding_8B = ModelMeta(
    loader=QwenEmbeddingModel,
    model_name="Qwen3-Embedding-8B",
    embd_dtype="float32",
    embd_dim=4096,
    max_tokens=32000,
    num_params=7_570_000_000,
    similarity="cosine",
    reference="https://huggingface.co/Qwen/Qwen3-Embedding-8B",
    vendor="Alibaba",
    tooltip="Alibaba's large 4096d model with 32K context"
)


e5_mistral_7b_instruct = ModelMeta(
    loader=E5EmbeddingModel,
    model_name="e5-mistral-7b-instruct",
    embd_dtype="float32",
    embd_dim=4096,
    num_params=7_110_000_000,
    max_tokens=4096,
    similarity="cosine",
    reference="https://huggingface.co/intfloat/e5-mistral-7b-instruct",
    vendor="Microsoft",
    tooltip="Large 4096d instruction-tuned model, 7.1B params"
)


all_MiniLM_L6_v2 = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="all-MiniLM-L6-v2",
    embd_dtype="float32",
    embd_dim=384,
    num_params=22_700_000,
    max_tokens=256,
    similarity="cosine",
    reference="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    tooltip="Ultra-compact 384d model, fastest inference, 23M params"
)


all_MiniLM_L12_v2 = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="all-MiniLM-L12-v2",
    embd_dtype="float32",
    embd_dim=384,
    num_params=33_400_000,
    max_tokens=256,
    similarity="cosine",
    reference="https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2",
    tooltip="Lightweight 384d model, fast inference, 33M params"
)


labse = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="LaBSE",
    embd_dtype="float32",
    embd_dim=768,
    num_params=471_000_000,
    max_tokens=256,
    similarity="cosine",
    reference="https://huggingface.co/sentence-transformers/LaBSE",
    vendor="Google",
    tooltip="Google's multilingual sentence encoder, 768d"
)


multi_qa_MiniLM_L6_cos_v1 = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="multi-qa-MiniLM-L6-cos-v1",
    embd_dtype="float32",
    embd_dim=384,
    num_params=22_700_000,
    max_tokens=512,
    similarity="cosine",
    reference="https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    tooltip="QA-optimized compact model, 384d, 23M params"
)


all_mpnet_base_v2 = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="all-mpnet-base-v2",
    embd_dtype="float32",
    embd_dim=768,
    num_params=109_000_000,
    max_tokens=384,
    similarity="cosine",
    reference="https://huggingface.co/sentence-transformers/all-mpnet-base-v2",
    tooltip="Balanced 768d model, good quality-speed tradeoff"
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
    tooltip="English-focused 768d model with 8K context"
)


jina_embeddings_v2_small_en = ModelMeta(
    loader=SentenceTransformersEmbeddingModel,
    model_name="jina-embeddings-v2-small-en",
    embd_dtype="float32",
    embd_dim=512,
    num_params=33_000_000,
    max_tokens=8192,
    similarity="cosine",
    reference="https://huggingface.co/jinaai/jina-embeddings-v2-small-en",
    vendor="Jina AI",
    tooltip="Compact 512d English model with 8K context"
)


# Octen models
octen_embedding_8B = ModelMeta(
    loader=OctenEmbeddingModel,
    model_name="Octen-Embedding-8B",
    embd_dtype="float32",
    embd_dim=4096,
    max_tokens=32768,
    num_params=7_600_000_000,
    similarity="cosine",
    reference="https://huggingface.co/Octen/Octen-Embedding-8B",
    vendor="Octen",
    tooltip="Octen's large 4096d model based on Qwen3, 40K context"
)


octen_embedding_4B = ModelMeta(
    loader=OctenEmbeddingModel,
    model_name="Octen-Embedding-4B",
    embd_dtype="float32",
    embd_dim=2560,
    max_tokens=32768,
    num_params=4_000_000_000,
    similarity="cosine",
    reference="https://huggingface.co/Octen/Octen-Embedding-4B",
    vendor="Octen",
    tooltip="Octen's efficient 2560d model based on Qwen3, 40K context"
)


# NVIDIA Llama Embed model
llama_embed_nemotron_8b = ModelMeta(
    loader=NvidiaEmbeddingModel,
    model_name="llama-embed-nemotron-8b",
    embd_dtype="float32",
    embd_dim=4096,
    max_tokens=32768,
    num_params=7_500_000_000,
    similarity="cosine",
    query_instruct="Instruct: Given a query, retrieve passages that answer the query\nQuery: ",
    reference="https://huggingface.co/nvidia/llama-embed-nemotron-8b",
    vendor="NVIDIA",
    tooltip="NVIDIA's instruction-aware 4096d model based on Llama-3.1-8B"
)


# Tencent KaLM model
kalm_embedding_gemma3_12B = ModelMeta(
    loader=TencentEmbeddingModel,
    model_name="KaLM-Embedding-Gemma3-12B-2511",
    embd_dtype="float32",
    embd_dim=3840,
    max_tokens=32000,
    num_params=11_760_000_000,
    similarity="cosine",
    query_instruct="Instruct: Given a query, retrieve documents that answer the query\nQuery: ",
    reference="https://huggingface.co/tencent/KaLM-Embedding-Gemma3-12B-2511",
    vendor="Tencent",
    tooltip="Tencent's large 3840d model based on Gemma3-12B"
)


