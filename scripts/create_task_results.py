#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RTEB Task Evaluation Result Generation Script

This script generates MTEB-compatible task evaluation result files based on
RTEB evaluation data. It converts result.json and dataset.json into properly
formatted MTEB result files in the results_rteb directory.

Requirements:
    - Python 3.8+
    - Latest version of mteb library (recommended: pip install --upgrade mteb)
    - loguru library for enhanced logging (pip install loguru)
    - result.json: Contains evaluation results for different models and datasets
    - dataset.json: Contains dataset metadata and information

Usage:
    python create_task_results.py [options]

    # Basic usage with default files
    python create_task_results.py

    # Specify custom result and dataset files
    python create_task_results.py -r my_results.json -d my_datasets.json

For best results, ensure you have the latest mteb version installed:
    pip install --upgrade mteb

The script will automatically generate task names by appending 'Retrieval' to
dataset names and create proper MTEB-format directory structures.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import hashlib
from loguru import logger  # Import loguru for enhanced logging

try:
    import mteb

    MTEB_AVAILABLE = True
except ImportError:
    MTEB_AVAILABLE = False
    logger.warning("mteb library not installed, will use fallback method to create model_meta.json")

_LANGUAGES = {
    "en": ["eng-Latn", "eng-Latn"],
    "es": ["spa-Latn", "eng-Latn"],
    "fr": ["fra-Latn", "eng-Latn"],
}


class RTEBTaskResultCreator:
    """RTEB Task Result Creator"""

    def __init__(self, organization: str = "RTEB"):
        self.organization = organization
        self.base_dir = Path("results_rteb")
        self.results_dir = self.base_dir / "results"
        # Build mapping from model names to ModelMeta
        self.model_mapping: Dict[str, Any] = self._build_model_mapping()
        # Build mapping from task names to TaskResult name
        self.task_mapping: Dict[str, Any] = self._build_task_mapping()

    @staticmethod
    def _build_model_mapping() -> Dict[str, Any]:
        """Build mapping from model names to ModelMeta"""
        mapping = {}

        if not MTEB_AVAILABLE:
            logger.warning(
                "mteb library not available, skipping model mapping initialization"
            )
            return mapping

        try:
            logger.info("Fetching all model information from MTEB...")
            all_models = mteb.get_model_metas()

            for model_meta in all_models:
                full_name = dict(model_meta).get("name", "")
                if "/" not in full_name:
                    continue

                # Extract simplified model name (remove organization name)
                model_name = full_name.split("/", 1)[1]
                mapping[model_name] = model_meta

            logger.info(
                f"Successfully created mapping for {len(all_models)} models, generated {len(mapping)} mapping relationships"
            )
            return mapping

        except Exception as e:
            logger.error(f"Failed to initialize MTEB model mapping: {e}")
            return mapping

    @staticmethod
    def _build_task_mapping() -> Dict[str, Any]:
        """Build mapping from task names to TaskResult name"""
        task_mapping = {}

        if not MTEB_AVAILABLE:
            logger.warning(
                "mteb library not available, skipping tasks mapping initialization"
            )
            return task_mapping

        try:
            logger.info("Fetching all task information from MTEB...")
            all_retrieval_tasks = mteb.get_tasks(task_types=["Retrieval"])

            for task_meta in all_retrieval_tasks:
                task_result_name = type(task_meta).__name__

                task_mapping[task_result_name] = task_meta

            logger.info(
                f"Successfully created mapping for {len(all_retrieval_tasks)} tasks, generated {len(task_mapping)} mapping relationships"
            )
            return task_mapping

        except Exception as e:
            logger.error(f"Failed to initialize MTEB tasks mapping: {e}")
            return task_mapping

    def get_model_meta(self, model_name: str) -> Optional[Any]:
        """Get model metadata"""
        if not MTEB_AVAILABLE:
            return None

        if model_name in self.model_mapping:
            model_meta = self.model_mapping[model_name]
            logger.info(f"  Retrieved model metadata: {model_meta.name}")
            return model_meta

        logger.warning(f"  No mapping found for model '{model_name}'")
        return None

    def get_vendor_and_meta(self, model_name: str) -> tuple[str, Optional[Any]]:
        """Get vendor and model metadata"""
        if model_name in self.model_mapping:
            model_meta = self.model_mapping[model_name]
            vendor = model_meta.name.split("/", 1)[0]
            return vendor, model_meta

        return self.organization, None

    def create_fallback_model_meta(
            self, model_name: str, vendor: str
    ) -> Dict[str, Any]:
        """Create fallback model_meta.json"""
        return {
            "name": f"{vendor}/{model_name}",
            "revision": "external",
            "release_date": datetime.now().strftime("%Y-%m-%d"),
            "languages": ["en"],
            "n_parameters": None,
            "memory_usage": None,
            "max_tokens": None,
            "embed_dim": None,
            "license": None,
            "open_weights": None,
            "public_training_data": None,
            "public_training_code": None,
            "framework": [],
            "reference": None,
            "similarity_fn_name": "cosine",
            "use_instructions": None,
            "training_datasets": None,
            "adapted_from": None,
            "superseded_by": None,
            "loader": None,
        }

    def create_model_directory_and_meta(self, model_name: str) -> bool:
        """Create model directory and model_meta.json"""
        try:
            vendor, model_meta = self.get_vendor_and_meta(model_name)

            if model_meta is None:
                # fallback case, create dictionary format model_meta
                model_meta_to_write = self.create_fallback_model_meta(
                    model_name, vendor
                )
                revision = "external"
                logger.info(f"  Using fallback to create model_meta: {model_name}")
            else:
                # directly use MTEB model_meta object
                model_meta_to_write = model_meta
                revision = model_meta.revision
                logger.info(f"  Using MTEB model_meta: {model_meta.name}")

            model_dir_name = (
                model_name.replace(" ", "-").replace("(", "").replace(")", "")
            )
            model_dir = self.results_dir / f"{vendor}__{model_dir_name}"
            external_dir = model_dir / revision
            external_dir.mkdir(parents=True, exist_ok=True)

            # Create model_meta.json
            meta_file = external_dir / "model_meta.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(
                    dict(model_meta_to_write),
                    f,
                    indent=4,
                    ensure_ascii=False,
                    default=str,
                )

            logger.info(f"  Created model directory: {vendor}__{model_dir_name}/{revision}")
            return True

        except Exception as e:
            logger.error(f"  Failed to create model directory {model_name}: {e}")
            return False

    def create_task_result_json(
            self, model_result: Dict[str, Any], dataset_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create task result JSON"""
        dataset_hash = hashlib.md5(dataset_result["dataset_name"].encode()).hexdigest()[
                       :8
                       ]
        task_name, task_meta = self._generate_task_name(dataset_result["dataset_name"])
        if task_meta:

            splits = task_meta.eval_splits[0] if task_meta.eval_splits else "test"
            hf_subsets = task_meta.hf_subsets[0] if task_meta.hf_subsets else "default"
        else:
            splits = "test"
            hf_subsets = "default"

        # Determine main score
        main_score = model_result.get(
            "ndcg_at_10",
            model_result.get("ndcg_at_5", model_result.get("map_at_10", 0.0)),
        )

        languages = getattr(task_meta, "eval_langs", ["eng-Latn"])
        if dataset_result["dataset_name"].startswith("CUREv1"):
            hf_subsets = dataset_result["dataset_name"].rsplit("_", maxsplit=1)[1]
            languages = _LANGUAGES.get(hf_subsets, ["eng-Latn"])
        score_obj = {"hf_subset": hf_subsets, "languages": languages, "main_score": main_score}

        # Add retrieval metrics
        for k in [1, 3, 5, 10, 20, 50, 100]:
            for metric in ["ndcg_at", "map_at", "recall_at", "precision_at"]:
                key = f"{metric}_{k}"
                if key in model_result:
                    score_obj[key] = model_result[key]

        return {
            "dataset_revision": dataset_hash,
            "task_name": task_name,
            "evaluation_time": 0,
            "kg_co2_emissions": 0,
            "mteb_version": "1.0.0",
            "scores": {splits: [score_obj]},
        }

    def _generate_task_name(self, dataset_name: str) -> Tuple[str, Any]:
        """Generate task name automatically by adding 'Retrieval' suffix"""
        # Remove leading underscores
        clean_name = dataset_name.lstrip("_")

        # If the name is already in proper format (no underscores), just add Retrieval
        if "_" not in clean_name:
            task_result_name = f"{clean_name}"
        else:
            # Split by underscores and convert to camelCase, preserving original capitalization patterns
            words = clean_name.split("_")
            result_words = []

            for word in words:
                if word:  # Skip empty strings from consecutive underscores
                    # Preserve the original capitalization pattern of each word
                    result_words.append(word)

            # Join words and capitalize first letter of each word except the first
            if result_words:
                camel_case = result_words[0]
                for word in result_words[1:]:
                    # Capitalize first letter while preserving the rest
                    camel_case += word[0].upper() + word[1:] if len(word) > 0 else ""
            else:
                camel_case = clean_name
            task_result_name = camel_case
        task_meta = self.task_mapping.get(task_result_name)
        if task_meta:
            return task_meta.metadata.name, task_meta
        # if not fund task_name then try name+Retrieval
        task_meta = self.task_mapping.get(f"{task_result_name}Retrieval")
        if task_meta:
            return task_meta.metadata.name, task_meta
        # Deal with three special cases
        if task_result_name.upper() == "APPS".upper():
            return "AppsRetrieval", self.task_mapping.get("AppsRetrieval")
        if task_result_name.upper().startswith("CUREV1"):
            task_meta = self.task_mapping.get("CUREv1Retrieval")

            return "CUREv1", task_meta
        if task_result_name == "ChatDoctorHealthCareMagic":
            return "ChatDoctorRetrieval", self.task_mapping.get("ChatDoctorRetrieval")
        logger.warning(f"  Failed to find task_name:{task_result_name} in MTEB ")
        return task_result_name, None

    def create_task_files_for_model(
            self, model_name: str, dataset_results: List[Dict], closed_dataset: bool = False
    ) -> int:
        """Create task files for model"""
        # First check if this model has any results
        has_results = False
        valid_results = []

        for dataset_result in dataset_results:
            # Look for this model's results
            model_result = None
            for result in dataset_result.get("results", []):
                if result.get("model_name") == model_name:
                    model_result = result
                    break

            if model_result:
                has_results = True
                valid_results.append((dataset_result, model_result))

        # If this model has no results, don't create directory
        if not has_results:
            logger.warning(
                f"  Model {model_name} has no results on any dataset, skipping directory creation"
            )
            return 0

        # Only create directory when model has results
        if not self.create_model_directory_and_meta(model_name):
            return 0

        vendor, model_meta = self.get_vendor_and_meta(model_name)

        # Get correct revision
        if model_meta is None:
            revision = "external"
        else:
            revision = model_meta.revision

        model_dir_name = model_name.replace(" ", "-").replace("(", "").replace(")", "")
        model_dir = self.results_dir / f"{vendor}__{model_dir_name}"
        external_dir = model_dir / revision

        success_count = 0
        for dataset_result, model_result in valid_results:
            if not closed_dataset and dataset_result["dataset_name"].startswith("ClosedDataset"):
                continue
            # Create task result JSON
            task_name, task_meta = self._generate_task_name(dataset_result["dataset_name"])
            task_result = self.create_task_result_json(model_result, dataset_result)

            # Write to file
            task_file = external_dir / f"{task_name}.json"
            try:
                if task_name == "CUREv1" and task_file.exists():
                    with open(task_file, "r", encoding="utf-8") as f:
                        task_result_exist = json.load(f)
                        scores = task_result_exist["scores"]
                        eval_splits = task_meta.eval_splits[0] if task_meta.eval_splits else "test"
                        split_scores = scores[eval_splits]
                        score_obj = task_result.get("scores", {}).get(eval_splits, [])
                        split_scores.append(score_obj)
                        task_result = task_result_exist

                with open(task_file, "w", encoding="utf-8") as f:
                    json.dump(task_result, f, indent=4, ensure_ascii=False)
                success_count += 1
                logger.info(f"  Created task file: {task_name}.json")
            except Exception as e:
                logger.error(f"  Failed to create task file {task_name}.json: {e}")

        logger.info(f"  Model {model_name}: Created {success_count} task files")
        return success_count

    def load_json_files(self, result_file: str, dataset_file: str) -> tuple:
        """Load JSON files"""
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                results_data = json.load(f)
            logger.info(f"Successfully loaded results for {len(results_data)} datasets")

            with open(dataset_file, "r", encoding="utf-8") as f:
                datasets_data = json.load(f)
            logger.info(f"Successfully loaded {len(datasets_data)} dataset categories")

            return results_data, datasets_data

        except FileNotFoundError as e:
            logger.error(f"File not found - {e}")
            return [], []
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed - {e}")
            return [], []

    def create_all_task_files(
            self, result_file: str = "result.json", dataset_file: str = "dataset.json", closed_dateset: bool = False
    ) -> None:
        """Create all task files"""
        results_data, datasets_data = self.load_json_files(result_file, dataset_file)
        if not results_data:
            return

        # Collect all model names
        all_models = set()
        for dataset_result in results_data:
            for model_result in dataset_result.get("results", []):
                model_name = model_result.get("model_name")
                if model_name:
                    all_models.add(model_name)

        logger.info(f"\nFound {len(all_models)} models")
        logger.info(f"Found {len(results_data)} datasets")

        if MTEB_AVAILABLE:
            logger.info(
                "mteb library is available, will use mteb to get accurate model metadata"
            )
        else:
            logger.warning(
                "mteb library not available, will use fallback method to create model_meta.json"
            )

        self.results_dir.mkdir(parents=True, exist_ok=True)

        total_files_created = 0
        models_processed = 0

        logger.info(f"\nStarting to process {len(all_models)} models...")

        for i, model_name in enumerate(sorted(all_models), 1):
            logger.info(f"[{i}/{len(all_models)}] Processing model: {model_name}")

            files_created = self.create_task_files_for_model(model_name, results_data, closed_dateset)
            total_files_created += files_created
            models_processed += 1

            logger.info(f"    Created {files_created} task files\n")

        logger.info("\nCompletion Statistics:")
        logger.info(f"   Models processed: {models_processed}")
        logger.info(f"   Total task files: {total_files_created}")

        if total_files_created > 0:
            logger.info(f"\nMTEB format results created in: {self.results_dir}")


def main():
    """Main function"""
    # Configure loguru for better-looking output, default level INFO
    logger.remove()  # Remove default sink
    logger.add(sys.stderr, level="INFO")

    parser = argparse.ArgumentParser(
        description="Create MTEB format result files from RTEB evaluation data",
        epilog="""
Examples:
  %(prog)s                              # Use default result.json and dataset.json
  %(prog)s -r custom_results.json       # Use custom result file
  %(prog)s -r results.json -d data.json # Use custom result and dataset files

Requirements:
  - Latest mteb library: pip install --upgrade mteb
  - loguru library for enhanced logging: pip install loguru
  - result.json: Evaluation results for models and datasets
  - dataset.json: Dataset metadata and information

The script generates MTEB-compatible directory structure in results_rteb/
with properly formatted task result files for each model and dataset combination.
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--result-file",
        "-r",
        default="result.json",
        metavar="FILE",
        help="Path to result.json file containing evaluation results (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset-file",
        "-d",
        default="dataset.json",
        metavar="FILE",
        help="Path to dataset.json file containing dataset metadata (default: %(default)s)",
    )

    args = parser.parse_args()

    logger.info(">> Starting RTEB to MTEB format conversion...")
    logger.info("=" * 60)

    # Check if required files exist
    if not Path(args.result_file).exists():
        logger.error(f"Result file '{args.result_file}' not found!")
        logger.info(
            "   Please ensure the result file exists or specify a different path with -r"
        )
        return 1

    if not Path(args.dataset_file).exists():
        logger.error(f"Dataset file '{args.dataset_file}' not found!")
        logger.info(
            "   Please ensure the dataset file exists or specify a different path with -d"
        )
        return 1

    # MTEB version recommendation
    if MTEB_AVAILABLE:
        try:
            logger.info(f"Using mteb version: {mteb.__version__}")
            logger.info(
                "TIP: For latest model metadata, ensure mteb is up to date: pip install --upgrade mteb"
            )
        except AttributeError:
            logger.info("mteb library available")
    else:
        logger.warning("mteb library not found - will use fallback mode")
        logger.info("TIP: Install mteb for better model metadata: pip install mteb")

    logger.info("=" * 60)

    creator = RTEBTaskResultCreator()
    creator.create_all_task_files(
        result_file=args.result_file, dataset_file=args.dataset_file
    )

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
