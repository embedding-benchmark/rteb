from rteb.core.base import EmbeddingModel
from rteb.core.meta import ModelMeta, model_id
from rteb.models.cohere import *
from rteb.models.openai import *
from rteb.models.sentence_transformers import *
from rteb.models.voyageai import *
from rteb.models.bgem3 import *
from rteb.models.gritlm import *
from rteb.models.google import *


MODEL_REGISTRY: dict[str, ModelMeta] = {}
for name in dir():
    meta = eval(name)
    # Explicitly exclude `LazyImport` instances since the latter check invokes the import.
    if not isinstance(meta, LazyImport) and isinstance(meta, ModelMeta):
        MODEL_REGISTRY[meta._id] = eval(name)


def get_embedding_model(
    model_name: str, 
    embd_dim: int,
    embd_dtype: str,
    **kwargs
) -> EmbeddingModel:
    key = model_id(model_name, embd_dim, embd_dtype)
    #TODO: add logic to dynamically load missing model
    return MODEL_REGISTRY[key].load_model(**kwargs)
