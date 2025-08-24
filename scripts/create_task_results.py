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
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib

try:
    import mteb

    MTEB_AVAILABLE = True
except ImportError:
    MTEB_AVAILABLE = False
    print(
        "WARNING: mteb library not installed, will use fallback method to create model_meta.json"
    )


class RTEBTaskResultCreator:
    """RTEB Task Result Creator"""

    def __init__(self, organization: str = "RTEB"):
        self.organization = organization
        self.base_dir = Path("results_rteb")
        self.results_dir = self.base_dir / "results"
        # Build mapping from model names to ModelMeta
        self.model_mapping: Dict[str, Any] = self._build_model_mapping()
        # Build mapping from task names to TaskResult name
        self.task_mapping: Dict[str, str] = self._build_task_mapping()

    @staticmethod
    def _build_model_mapping() -> Dict[str, Any]:
        """Build mapping from model names to ModelMeta"""
        mapping = {}

        if not MTEB_AVAILABLE:
            print(
                "WARNING: mteb library not available, skipping model mapping initialization"
            )
            return mapping

        try:
            print("Fetching all model information from MTEB...")
            all_models = mteb.get_model_metas()

            for model_meta in all_models:
                full_name = dict(model_meta).get("name", "")
                if "/" not in full_name:
                    continue

                # Extract simplified model name (remove organization name)
                model_name = full_name.split("/", 1)[1]
                mapping[model_name] = model_meta

            print(
                f"Successfully created mapping for {len(all_models)} models, generated {len(mapping)} mapping relationships"
            )
            return mapping

        except Exception as e:
            print(f"Failed to initialize MTEB model mapping: {e}")
            return mapping

    @staticmethod
    def _build_task_mapping() -> Dict[str, str]:
        """Build mapping from task names to TaskResult name"""
        task_mapping = {}

        if not MTEB_AVAILABLE:
            print(
                "WARNING: mteb library not available, skipping tasks mapping initialization"
            )
            return task_mapping

        try:
            print("Fetching all task information from MTEB...")
            all_retrieval_tasks = mteb.get_tasks(task_types=["Retrieval"])

            for task_meta in all_retrieval_tasks:
                task_result_name = type(task_meta).__name__
                task_name = task_meta.metadata.name

                task_mapping[task_result_name] = task_name

            print(
                f"Successfully created mapping for {len(all_retrieval_tasks)} tasks, generated {len(task_mapping)} mapping relationships"
            )
            return task_mapping

        except Exception as e:
            print(f"Failed to initialize MTEB tasks mapping: {e}")
            return task_mapping

    def get_model_meta(self, model_name: str) -> Optional[Any]:
        """Get model metadata"""
        if not MTEB_AVAILABLE:
            return None

        if model_name in self.model_mapping:
            model_meta = self.model_mapping[model_name]
            print(f"  Retrieved model metadata: {model_meta.name}")
            return model_meta

        print(f"  WARNING: No mapping found for model '{model_name}'")
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
                print(f"  Using fallback to create model_meta: {model_name}")
            else:
                # directly use MTEB model_meta object
                model_meta_to_write = model_meta
                revision = model_meta.revision
                print(f"  Using MTEB model_meta: {model_meta.name}")

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

            print(f"  Created model directory: {vendor}__{model_dir_name}/{revision}")
            return True

        except Exception as e:
            print(f"  Failed to create model directory {model_name}: {e}")
            return False

    def create_task_result_json(
            self, model_result: Dict[str, Any], dataset_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create task result JSON"""
        dataset_hash = hashlib.md5(dataset_result["dataset_name"].encode()).hexdigest()[
                       :8
                       ]
        task_name = self._generate_task_name(dataset_result["dataset_name"])

        # Determine main score
        main_score = model_result.get(
            "ndcg_at_10",
            model_result.get("ndcg_at_5", model_result.get("map_at_10", 0.0)),
        )

        score_obj = {"hf_subset": "test", "languages": ["en"], "main_score": main_score}

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
            "scores": {"test": [score_obj]},
        }

    def _generate_task_name(self, dataset_name: str) -> str:
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
        task_name = self.task_mapping.get(task_result_name)
        if task_name:
            return task_name
        # if not fund task_name then try name+Retrieval
        task_name = self.task_mapping.get(f"{task_result_name}Retrieval")
        if task_name:
            return task_name
        # Deal with two special cases
        if task_result_name.upper() == "APPS".upper():
            return "AppsRetrieval"
        if task_result_name.upper().startswith("CUREV1"):
            return "CUREv1"
        print(f"  Failed to find task_name:{task_result_name} in MTEB ")
        return task_result_name

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
            print(
                f"  WARNING: Model {model_name} has no results on any dataset, skipping directory creation"
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
            task_result = self.create_task_result_json(model_result, dataset_result)
            task_name = self._generate_task_name(dataset_result["dataset_name"])

            # Write to file
            task_file = external_dir / f"{task_name}.json"
            try:
                with open(task_file, "w", encoding="utf-8") as f:
                    json.dump(task_result, f, indent=4, ensure_ascii=False)
                success_count += 1
                print(f"  Created task file: {task_name}.json")
            except Exception as e:
                print(f"  Failed to create task file {task_name}.json: {e}")

        print(f"  Model {model_name}: Created {success_count} task files")
        return success_count

    def load_json_files(self, result_file: str, dataset_file: str) -> tuple:
        """Load JSON files"""
        try:
            with open(result_file, "r", encoding="utf-8") as f:
                results_data = json.load(f)
            print(f"Successfully loaded results for {len(results_data)} datasets")

            with open(dataset_file, "r", encoding="utf-8") as f:
                datasets_data = json.load(f)
            print(f"Successfully loaded {len(datasets_data)} dataset categories")

            return results_data, datasets_data

        except FileNotFoundError as e:
            print(f"ERROR: File not found - {e}")
            return [], []
        except json.JSONDecodeError as e:
            print(f"ERROR: JSON parsing failed - {e}")
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

        print(f"\nFound {len(all_models)} models")
        print(f"Found {len(results_data)} datasets")

        if MTEB_AVAILABLE:
            print(
                "mteb library is available, will use mteb to get accurate model metadata"
            )
        else:
            print(
                "WARNING: mteb library not available, will use fallback method to create model_meta.json"
            )

        self.results_dir.mkdir(parents=True, exist_ok=True)

        total_files_created = 0
        models_processed = 0

        print(f"\nStarting to process {len(all_models)} models...")

        for i, model_name in enumerate(sorted(all_models), 1):
            print(f"[{i}/{len(all_models)}] Processing model: {model_name}")

            files_created = self.create_task_files_for_model(model_name, results_data, closed_dateset)
            total_files_created += files_created
            models_processed += 1

            print(f"    Created {files_created} task files\n")

        print("\nCompletion Statistics:")
        print(f"   Models processed: {models_processed}")
        print(f"   Total task files: {total_files_created}")

        if total_files_created > 0:
            print(f"\nMTEB format results created in: {self.results_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Create MTEB format result files from RTEB evaluation data",
        epilog="""
Examples:
  %(prog)s                              # Use default result.json and dataset.json
  %(prog)s -r custom_results.json       # Use custom result file
  %(prog)s -r results.json -d data.json # Use custom result and dataset files

Requirements:
  - Latest mteb library: pip install --upgrade mteb
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

    print(">> Starting RTEB to MTEB format conversion...")
    print("=" * 60)

    # Check if required files exist
    if not Path(args.result_file).exists():
        print(f"ERROR: Result file '{args.result_file}' not found!")
        print(
            "   Please ensure the result file exists or specify a different path with -r"
        )
        return 1

    if not Path(args.dataset_file).exists():
        print(f"ERROR: Dataset file '{args.dataset_file}' not found!")
        print(
            "   Please ensure the dataset file exists or specify a different path with -d"
        )
        return 1

    # MTEB version recommendation
    if MTEB_AVAILABLE:
        try:
            print(f"Using mteb version: {mteb.__version__}")
            print(
                "TIP: For latest model metadata, ensure mteb is up to date: pip install --upgrade mteb"
            )
        except AttributeError:
            print("mteb library available")
    else:
        print("WARNING: mteb library not found - will use fallback mode")
        print("TIP: Install mteb for better model metadata: pip install mteb")

    print("=" * 60)

    creator = RTEBTaskResultCreator()
    creator.create_all_task_files(
        result_file=args.result_file, dataset_file=args.dataset_file
    )

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

