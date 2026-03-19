from functools import cache
import json
import os

from torch.utils.data import Dataset

from rteb.core.base import RetrievalDataset
from rteb.core.meta import DatasetMeta
from rteb.utils.data import JSONLDataset


class MultimodalRetrievalDataset(RetrievalDataset):

    LEADERBOARD: str = "Multimodal"

    def __init__(
        self,
        data_path: str,
        dataset_meta: DatasetMeta,
        query_instruct: str | None = None,
        corpus_instruct: str | None = None,
        **kwargs
    ):
        super().__init__(
            data_path,
            dataset_meta,
            query_instruct=query_instruct,
            corpus_instruct=corpus_instruct,
            **kwargs
        )
        assert os.path.isdir(self._task_path), f"{self._task_path} is not a directory."

    def _resolve_content_paths(self, ds: Dataset) -> Dataset:
        """Resolve relative image paths in content items to absolute paths."""
        for item in ds.data:
            if "content" in item:
                for piece in item["content"]:
                    if piece.get("type") == "image_path" and not os.path.isabs(piece["path"]):
                        piece["path"] = os.path.join(self._task_path, piece["path"])
        return ds

    @property
    def corpus_file(self) -> str:
        for name in ["corpus.jsonl", "corpus.arrow"]:
            file = os.path.join(self._task_path, name)
            if os.path.exists(file):
                return file
        raise FileNotFoundError(
            f"Corpus file (corpus.{{jsonl/arrow}}) does not exist under {self._task_path}."
        )

    @cache
    def _corpus(self) -> Dataset:
        return self._resolve_content_paths(JSONLDataset(self.corpus_file))

    @property
    def queries_file(self) -> str:
        for name in ["queries.jsonl", "queries.arrow"]:
            file = os.path.join(self._task_path, name)
            if os.path.exists(file):
                return file
        raise FileNotFoundError(
            f"Queries file (queries.{{jsonl/arrow}}) does not exist under {self._task_path}."
        )

    @cache
    def _queries(self) -> Dataset:
        return self._resolve_content_paths(JSONLDataset(self.queries_file))

    @property
    def relevance_file(self) -> str:
        for name in ["relevance.json", "relevance.jsonl"]:
            file = os.path.join(self._task_path, name)
            if os.path.exists(file):
                return file
        raise FileNotFoundError(
            f"Relevance file (relevance.{{json/jsonl}}) does not exist under {self._task_path}."
        )

    @property
    @cache
    def relevance(self) -> dict:
        relevant_docs = {}
        try:
            with open(self.relevance_file) as f:
                for line in f:
                    data = json.loads(line)
                    for key, value in data.items():
                        if key not in relevant_docs:
                            relevant_docs[key] = value
                        else:
                            relevant_docs[key].update(value)
        except FileNotFoundError:
            return {}
        return relevant_docs


VidoreDocVQA = DatasetMeta(
    loader=MultimodalRetrievalDataset,
    dataset_name="VidoreDocVQA",
    tier=0,
    groups={"multimodal": 1, "document_understanding": 1},
    reference="https://huggingface.co/datasets/vidore/docvqa_test_subsampled_beir"
)

VidoreArxivQA = DatasetMeta(
    loader=MultimodalRetrievalDataset,
    dataset_name="VidoreArxivQA",
    tier=0,
    groups={"multimodal": 1, "document_understanding": 1},
    reference="https://huggingface.co/datasets/vidore/arxivqa_test_subsampled_beir"
)

VidoreInfoVQA = DatasetMeta(
    loader=MultimodalRetrievalDataset,
    dataset_name="VidoreInfoVQA",
    tier=0,
    groups={"multimodal": 1, "document_understanding": 1},
    reference="https://huggingface.co/datasets/vidore/infovqa_test_subsampled_beir"
)

VidoreShiftProject = DatasetMeta(
    loader=MultimodalRetrievalDataset,
    dataset_name="VidoreShiftProject",
    tier=0,
    groups={"multimodal": 1, "document_understanding": 1},
    reference="https://huggingface.co/datasets/vidore/shiftproject_test_beir"
)

VidoreSyntheticDocQAAI = DatasetMeta(
    loader=MultimodalRetrievalDataset,
    dataset_name="VidoreSyntheticDocQAAI",
    tier=0,
    groups={"multimodal": 1, "document_understanding": 1},
    reference="https://huggingface.co/datasets/vidore/syntheticDocQA_artificial_intelligence_test_beir"
)

VidoreSyntheticDocQAEnergy = DatasetMeta(
    loader=MultimodalRetrievalDataset,
    dataset_name="VidoreSyntheticDocQAEnergy",
    tier=0,
    groups={"multimodal": 1, "document_understanding": 1},
    reference="https://huggingface.co/datasets/vidore/syntheticDocQA_energy_test_beir"
)

VidoreSyntheticDocQAGovReports = DatasetMeta(
    loader=MultimodalRetrievalDataset,
    dataset_name="VidoreSyntheticDocQAGovReports",
    tier=0,
    groups={"multimodal": 1, "document_understanding": 1},
    reference="https://huggingface.co/datasets/vidore/syntheticDocQA_government_reports_test_beir"
)

VidoreSyntheticDocQAHealthcare = DatasetMeta(
    loader=MultimodalRetrievalDataset,
    dataset_name="VidoreSyntheticDocQAHealthcare",
    tier=0,
    groups={"multimodal": 1, "document_understanding": 1},
    reference="https://huggingface.co/datasets/vidore/syntheticDocQA_healthcare_industry_test_beir"
)

VidoreTabfquad = DatasetMeta(
    loader=MultimodalRetrievalDataset,
    dataset_name="VidoreTabfquad",
    tier=0,
    groups={"multimodal": 1, "document_understanding": 1},
    reference="https://huggingface.co/datasets/vidore/tabfquad_test_subsampled_beir"
)

VidoreTatdqa = DatasetMeta(
    loader=MultimodalRetrievalDataset,
    dataset_name="VidoreTatdqa",
    tier=0,
    groups={"multimodal": 1, "document_understanding": 1},
    reference="https://huggingface.co/datasets/vidore/tatdqa_test_beir"
)

Vidore2BioMedicalLectures = DatasetMeta(
    loader=MultimodalRetrievalDataset,
    dataset_name="Vidore2BioMedicalLectures",
    tier=0,
    groups={"multimodal": 1, "document_understanding": 1},
    reference="https://huggingface.co/datasets/vidore/biomedical_lectures_v2"
)

Vidore2ESGReportsHL = DatasetMeta(
    loader=MultimodalRetrievalDataset,
    dataset_name="Vidore2ESGReportsHL",
    tier=0,
    groups={"multimodal": 1, "document_understanding": 1},
    reference="https://huggingface.co/datasets/vidore/esg_reports_human_labeled_v2"
)

Vidore2ESGReports = DatasetMeta(
    loader=MultimodalRetrievalDataset,
    dataset_name="Vidore2ESGReports",
    tier=0,
    groups={"multimodal": 1, "document_understanding": 1},
    reference="https://huggingface.co/datasets/vidore/esg_reports_v2"
)

Vidore2EconomicsReports = DatasetMeta(
    loader=MultimodalRetrievalDataset,
    dataset_name="Vidore2EconomicsReports",
    tier=0,
    groups={"multimodal": 1, "document_understanding": 1},
    reference="https://huggingface.co/datasets/vidore/economics_reports_v2"
)
