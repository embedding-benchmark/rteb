from rteb.core.base import EmbeddingModel
from rteb.utils.lazy_import import LazyImport
from rteb.core.meta import ModelMeta

GritLM = LazyImport("gritlm", attribute="GritLM")


class GRITLMEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_meta: ModelMeta,
        **kwargs
    ):
        super().__init__(model_meta, **kwargs)
        self._model = GritLM(
            model_name_or_path= "GritLM/"+model_meta.model_name,
            normalized=False,
            torch_dtype=model_meta.embd_dtype,
            mode="embedding",
        )

    def embed(self, data: list[str], input_type: str) -> list[list[float]]:
        result = self._model.encode(sentences=data)
        return [[float(str(x)) for x in result[i]] for i in range(len(result))]


gritlm_7b = ModelMeta(
    loader=GRITLMEmbeddingModel,
    model_name="GritLM-7B",
    embd_dtype="float32",
    embd_dim=384,
    num_params=7_240_000_000,
    max_tokens=8192,
    similarity="cosine",
    reference="https://huggingface.co/GritLM/GritLM-7B",
    tooltip="Generative retrieval model, 384d output, 7.2B params"
)

gritlm_8x7b = ModelMeta(
    loader=GRITLMEmbeddingModel,
    model_name="GritLM-8x7B",
    embd_dtype="float32",
    embd_dim=384,
    num_params=46_700_000_000,
    similarity="cosine",
    reference="https://huggingface.co/GritLM/GritLM-8x7B",
    tooltip="Generative retrieval model, 384d output, 7.2B params"
)
