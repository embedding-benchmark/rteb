from rteb.core.base import EmbeddingModel
from rteb.utils.lazy_import import LazyImport
from rteb.core.meta import ModelMeta

BGEM3FlagModel = LazyImport("FlagEmbedding", attribute="BGEM3FlagModel")


class BGEM3EmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_meta: ModelMeta,
        **kwargs
    ):
        super().__init__(model_meta, **kwargs)
        self._model = BGEM3FlagModel(
            model_name_or_path=f"BAAI/{model_meta.model_name}",
        )

    def embed(self, data: list[str], input_type: str) -> list[list[float]]:
        result = self._model.encode(sentences=data, batch_size=12)['dense_vecs']
        return [[float(str(x)) for x in result[i]] for i in range(len(result))]


bge_m3 = ModelMeta(
    loader=BGEM3EmbeddingModel,
    model_name='bge-m3',
    embd_dtype="float32",
    embd_dim=1024,
    num_params=569_000_000,
    max_tokens=8192,
    similarity="cosine",
    reference="https://huggingface.co/BAAI/bge-m3",
    vendor="BAAI",
    tooltip="Multilingual 1024d model with 8K context"
)

bge_m3_unsupervised = ModelMeta(
    loader=BGEM3EmbeddingModel,
    model_name='bge-m3-unsupervised',
    embd_dtype="float32",
    embd_dim=1024,
    max_tokens=8192,
    similarity="cosine",
    reference="https://huggingface.co/BAAI/bge-m3-unsupervised",
    vendor = "BAAI",
    tooltip="M3 variant trained without supervision"
)

bge_m3_retromae = ModelMeta(
    loader=BGEM3EmbeddingModel,
    model_name='bge-m3-retromae',
    embd_dtype="float32",
    embd_dim=1024,
    max_tokens=8192,
    similarity="cosine",
    reference="https://huggingface.co/BAAI/bge-m3-retromae",
    vendor="BAAI",
    tooltip="M3 variant with RetroMAE pretraining enhancement"
)

bge_large_en_v15 = ModelMeta(
    loader=BGEM3EmbeddingModel,
    model_name='bge-large-en-v1.5',
    embd_dtype="float32",
    embd_dim=1024,
    num_params=335_000_000,
    max_tokens=512,
    similarity="cosine",
    reference="https://huggingface.co/BAAI/bge-large-en-v1.5",
    vendor="BAAI",
    tooltip="High-quality 1024d English model, 335M params"
)

bge_base_en_v15 = ModelMeta(
    loader=BGEM3EmbeddingModel,
    model_name='bge-base-en-v1.5',
    embd_dtype="float32",
    embd_dim=768,
    num_params=109_000_000,
    max_tokens=512,
    similarity="cosine",
    reference="https://huggingface.co/BAAI/bge-base-en-v1.5",
    vendor="BAAI",
    tooltip="Solid 768d English model, 109M params"
)

bge_small_en_v15 = ModelMeta(
    loader=BGEM3EmbeddingModel,
    model_name='bge-small-en-v1.5',
    embd_dtype="float32",
    embd_dim=384,
    num_params=33_400_000,
    max_tokens=512,
    similarity="cosine",
    reference="https://huggingface.co/BAAI/bge-small-en-v1.5",
    vendor="BAAI",
    tooltip="Compact 384d English model, 33M params"
)
