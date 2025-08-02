import argparse
from collections import defaultdict
import json
import logging
import os
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy

from ebr.retrieve import run_retrieve_task
from ebr.datasets import DatasetMeta, DATASET_REGISTRY
from ebr.models import ModelMeta, MODEL_REGISTRY
from ebr.core import Encoder, Retriever


logger = logging.getLogger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def list_available_models():
    """Print a list of available models for the user."""
    print("\nAVAILABLE MODELS")
    print("===============")
    print("Use these model IDs with the --models argument (e.g., --models model_id1,model_id2)")
    print("\nFormat: MODEL_ID: ALIAS [DIMENSIONS]")
    print("-" * 80)
    
    # Group models by provider for better organization
    models_by_provider = {}
    for model_id, model_meta in MODEL_REGISTRY.items():
        provider = model_id.split('_')[0] if '_' in model_id else "Other"
        if "__" in provider:
            provider = provider.split("__")[0]
        
        if provider not in models_by_provider:
            models_by_provider[provider] = []
        
        alias = model_meta.alias if model_meta.alias else model_id
        dim_info = f"{model_meta.embd_dim}d" if model_meta.embd_dim else ""
        models_by_provider[provider].append((model_id, alias, dim_info))
    
    # Print models grouped by provider
    for provider, models in sorted(models_by_provider.items()):
        print(f"\n{provider.upper()} Models:")
        for model_id, alias, dim_info in sorted(models):
            print(f"  - {model_id}: {alias} [{dim_info}]")


def list_available_tasks():
    """Print a list of available tasks/datasets for the user."""
    print("\nAVAILABLE TASKS/DATASETS")
    print("=====================")
    print("Use these task names with the --tasks argument (e.g., --tasks task1,task2)")
    print("\nFormat: TASK_NAME: [TIER] (GROUPS)")
    print("-" * 80)
    
    # Group datasets by their groups for better organization
    tasks_by_group = {}
    for task_id, task_meta in DATASET_REGISTRY.items():
        groups = list(task_meta.groups.keys())
        group_key = ", ".join(sorted(groups)) if groups else "Ungrouped"
        
        if group_key not in tasks_by_group:
            tasks_by_group[group_key] = []
        
        # Get tier information
        tier_info = f"Tier {task_meta.tier}"
        if task_meta.tier == 0:
            tier_info += " (fully open)"
        elif task_meta.tier == 1:
            tier_info += " (docs & queries open)"
        elif task_meta.tier == 2:
            tier_info += " (only docs open)"
        elif task_meta.tier == 3:
            tier_info += " (fully held out)"
        
        tasks_by_group[group_key].append((task_id, tier_info, groups))
    
    # Print tasks grouped by their groups
    for group, tasks in sorted(tasks_by_group.items()):
        print(f"\n{group} Tasks:")
        for task_id, tier_info, groups in sorted(tasks):
            groups_str = f"({', '.join(groups)})" if groups else ""
            print(f"  - {task_id}: [{tier_info}] {groups_str}")
            
            # Add reference if available
            if DATASET_REGISTRY[task_id].reference:
                print(f"    Reference: {DATASET_REGISTRY[task_id].reference}")
    
    print("\nNote: Tasks are grouped by their category. The tier indicates data availability.")

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Evaluation
    parser.add_argument(
        "--gpus", type=int, default=0, help="Number of gpus used for encoding.")
    parser.add_argument(
        "--cpus", type=int, default=1, help="Number of cpus used for computation (this is only for models that are not using gpus).")
    parser.add_argument(
        "--bf16", action="store_true", help="`Use bf16 precision.")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for encoding.")
    parser.add_argument(
        "--embd_batch_size", type=int, default=1024, help="Batch size for computing similarity of embeddings.")
    parser.add_argument(
        "--embd_in_memory_threshold", type=int, default=200000,
        help="Embeddings will be stored in memory if the amount is below this threshold.")

    # Model
    parser.add_argument(
        "--models", type=str, default=None, 
        help="Comma-separated list of model IDs to evaluate (e.g., 'text-embedding-3-small_float32_512d,bge-m3_float32_1024d'). "
             "Use --list-models to see all available model IDs. If not specified, all models will be evaluated.")
    parser.add_argument(
        "--list-models", action="store_true",
        help="List all available models with their IDs and aliases, then exit. Use these IDs with the --models argument.")
    parser.add_argument(
        "--list-tasks", action="store_true",
        help="List all available tasks/datasets with their details, then exit. Use these names with the --tasks argument.")
    #parser.add_argument(
    #    "--model_name", type=str, default=None, help="Model name or path.")
    #parser.add_argument(
    #    "--embd_dtype", type=str, default="float", help="Embedding type. Options: float32, int8, binary.")
    #parser.add_argument(
    #    "--embd_dim", type=int, default=None, help="Embedding dimension.")
    #parser.add_argument(
    #    "--max_length", type=int, default=None, help="Maximum length of model input.")

    # Data
    parser.add_argument(
        "--data_path", type=str, default="data/", help="Path of the dataset, must be specified for custom tasks.")
    parser.add_argument(
        "--tasks", type=str, default=None, help="Comma-separated list of task names to evaluate (e.g., 'task1,task2'). Use --list-tasks to see all available task names. If not specified, all tasks will be evaluated.")
    parser.add_argument(
        "--data_type", default="eval", choices=["eval", "train", "chunk", "merge"], help="Dataset type.")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for dataloader.")
    
    # Output
    parser.add_argument(
        "--save_path", type=str, default="output/", help="Path to save the output.")
    parser.add_argument(
        "--load_embds", action="store_true", help="Whether to load the computed embeddings.")
    parser.add_argument(
        "--keep_embds", action="store_true", help="Keep embedding files after retrieval (default: delete them).")
    parser.add_argument(
        "--save_prediction", action="store_true", help="Whether to save the predictions.")
    parser.add_argument(
        "--topk", type=int, default=100, help="Number of top documents per query.")
    parser.add_argument(
        "--overwrite", action="store_true", help="Whether to overwrite the results.")
    parser.add_argument(
        "--offload-model", action="store_true", 
        help="Offload model after encoding to save memory during retrieval phase.")
    
    args = parser.parse_args()
    return args


def _get_complete_models(
    output_dir: str = "output",
    results_dir: str = "results",
    show_report: bool = True
) -> set[str]:
    """Get models that have results for all datasets."""
    from collections import defaultdict
    
    # Count total datasets in registry
    total_datasets = len(DATASET_REGISTRY)
    all_datasets = set(DATASET_REGISTRY.keys())
    
    # Track which datasets each model has results for - start with all possible models
    model_datasets = defaultdict(set)
    
    # First, check existing results.json to get baseline coverage
    results_file = Path(results_dir) / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            results_data = json.load(f)
            
        for dataset_entry in results_data:
            dataset_name = dataset_entry.get("dataset_name")
            if dataset_name in DATASET_REGISTRY:
                for result in dataset_entry.get("results", []):
                    if "model_name" in result:
                        model_name = result["model_name"]
                        model_datasets[model_name].add(dataset_name)
    
    # Then, update with any new results from output directory
    output_path = Path(output_dir)
    if output_path.exists() and output_path.is_dir():
        for dataset_output_dir in output_path.iterdir():
            # Skip if the dataset is not in the registry
            if dataset_output_dir.name not in DATASET_REGISTRY:
                continue
                
            dataset_name = dataset_output_dir.name
            for one_result in dataset_output_dir.iterdir():
                eval_file = one_result / "retrieve_eval.json"
                if eval_file.exists():
                    with open(eval_file) as f:
                        result_data = json.load(f)
                        if "model_name" in result_data:
                            model_name = result_data["model_name"]
                            model_datasets[model_name].add(dataset_name)
    
    # Separate complete and incomplete models
    complete_models = {model for model, datasets in model_datasets.items() 
                      if len(datasets) >= total_datasets}
    
    incomplete_models = {model: datasets for model, datasets in model_datasets.items() 
                        if len(datasets) < total_datasets}
    
    # Generate detailed report using print to ensure visibility
    if show_report:
        print("=" * 80)
        print("MODEL COMPLETENESS REPORT")
        print("=" * 80)
        print(f"Total datasets in registry: {total_datasets}")
        print(f"Complete models (will be included): {len(complete_models)}")
        print(f"Incomplete models (will be excluded): {len(incomplete_models)}")
        print("")
        
        if complete_models:
            print("COMPLETE MODELS:")
            for model in sorted(complete_models):
                print(f"  ✓ {model} ({len(model_datasets[model])}/{total_datasets} datasets)")
            print("")
        
        if incomplete_models:
            print("INCOMPLETE MODELS (EXCLUDED):")
            for model in sorted(incomplete_models.keys()):
                model_dataset_set = incomplete_models[model]
                missing_datasets = all_datasets - model_dataset_set
                print(f"  ✗ {model} ({len(model_dataset_set)}/{total_datasets} datasets)")
                print(f"    Missing datasets ({len(missing_datasets)}):")
                for missing in sorted(missing_datasets):
                    print(f"      - {missing}")
                print("")
        
        print("=" * 80)
    
    return complete_models


def _dump_model_meta(
    results_dir: str = "results",
    model_registry: dict[str, ModelMeta] = MODEL_REGISTRY,
    output_dir: str = "output"
):
    print("Generating models.json with completeness filtering...")
    
    # Get models with complete results (this will print the detailed report)
    complete_models = _get_complete_models(output_dir, results_dir, show_report=True)
    
    # Filter model registry to only include complete models
    complete_model_metas = []
    for meta in model_registry.values():
        # Check if this model has complete results
        model_alias = meta.alias if meta.alias else meta.model_name
        if meta.model_name in complete_models or model_alias in complete_models:
            complete_model_metas.append(meta.model_dump())
    
    print(f"✓ Writing {len(complete_model_metas)} complete models to {results_dir}/models.json")
    
    with open(Path(results_dir) / "models.json", "w") as f:
        f.write(json.dumps(complete_model_metas, indent=4))

def _dump_dataset_info(
    results_dir: str = "results",
    dataset_registry: dict[str, DatasetMeta] = DATASET_REGISTRY,
):
    group_data = defaultdict(list)
    for dataset_meta in dataset_registry.values():
        for group_name in dataset_meta.groups.keys():
            leaderboard = dataset_meta.loader.LEADERBOARD
            group_data[(leaderboard, group_name)].append(dataset_meta.dataset_name)

    groups = []
    for (leaderboard, group_name), datasets in group_data.items():
        groups.append({"name": group_name, "datasets": datasets, "leaderboard": leaderboard})
    with open(Path(results_dir) / "datasets.json", "w") as f:
        f.write(json.dumps(groups, indent=4))


def _compile_results(
    results_dir: str = "results",
    output_dir: str = "output"
):
    print("Generating results.json with completeness filtering...")
    
    # Get models with complete results (report will be shown later)
    complete_models = _get_complete_models(output_dir, results_dir, show_report=False)
    
    results = []
    for dataset_output_dir in Path(output_dir).iterdir():
        # Skip if the dataset is not in the registry
        if dataset_output_dir.name not in DATASET_REGISTRY:
            continue

        dataset_results = []
        for one_result in dataset_output_dir.iterdir():
            eval_file = one_result / "retrieve_eval.json"
            if eval_file.exists():
                with open(eval_file) as f:
                    result_data = json.load(f)
                    # Only include results from models with complete coverage
                    model_name = result_data.get("model_name")
                    model_alias = result_data.get("alias", model_name)
                    
                    if model_name in complete_models or model_alias in complete_models:
                        dataset_results.append(result_data)

        # Only include dataset if it has results from complete models
        if dataset_results:
            results.append({
                **DATASET_REGISTRY[dataset_output_dir.name].model_dump(),
                "results": dataset_results,
                "is_closed": DATASET_REGISTRY[dataset_output_dir.name].tier != 0
            })
    
    print(f"✓ Writing results for {len(results)} datasets with complete model coverage to {results_dir}/results.json")

    with open(Path(results_dir) / "results.json", "w") as f:
        f.write(json.dumps(results, indent=4))


def main(args: argparse.Namespace):

    _dump_dataset_info()

    if args.gpus:
        trainer = pl.Trainer(
            strategy=DDPStrategy(find_unused_parameters=False),
            accelerator="gpu",
            devices=args.gpus,
            precision="bf16" if args.bf16 else "32",
        )
    else:
        trainer = pl.Trainer(
            strategy=DDPStrategy(),
            accelerator="cpu",
            devices=args.cpus,
        )

    if not trainer.is_global_zero:
        logging.basicConfig(level=logging.ERROR)

    # Filter models based on the --models argument
    models_to_evaluate = MODEL_REGISTRY
    if args.models:
        model_ids = [model_id.strip() for model_id in args.models.split(',')]
        models_to_evaluate = {model_id: MODEL_REGISTRY[model_id] for model_id in model_ids if model_id in MODEL_REGISTRY}
        if not models_to_evaluate:
            logger.error(f"No valid models found in the provided list: {args.models}")
            logger.info(f"Available models: {list(MODEL_REGISTRY.keys())}")
            return
        
        if trainer.is_global_zero:
            trainer.print(f"Evaluating {len(models_to_evaluate)} models: {list(models_to_evaluate.keys())}")
    
    # Evaluate each model on the specified datasets
    for model_id, model_meta in models_to_evaluate.items():
        if trainer.is_global_zero:
            trainer.print(f"Evaluating model: {model_id}")

        # Determine device based on GPU/CPU arguments
        device = "cuda" if args.gpus > 0 else "cpu"
        
        encoder = Encoder(
            model_meta.load_model(device=device),
            load_embds=args.load_embds
        )
        retriever = Retriever(
            topk=args.topk,
            similarity=model_meta.similarity,
            save_prediction=args.save_prediction
        )

        # Filter datasets based on the --tasks argument
        datasets_to_evaluate = DATASET_REGISTRY
        if args.tasks:
            task_names = [task_name.strip() for task_name in args.tasks.split(',')]
            datasets_to_evaluate = {task_name: DATASET_REGISTRY[task_name] for task_name in task_names if task_name in DATASET_REGISTRY}
            if not datasets_to_evaluate:
                logger.error(f"No valid tasks found in the provided list: {args.tasks}")
                logger.info(f"Available tasks: {list(DATASET_REGISTRY.keys())}")
                return

            if trainer.is_global_zero:
                trainer.print(f"Evaluating on {len(datasets_to_evaluate)} tasks: {list(datasets_to_evaluate.keys())}")

        eval_results = {}
        for dataset_name, dataset_meta in datasets_to_evaluate.items():
            if trainer.is_global_zero:
                trainer.print(f"Evaluating {model_meta.model_name} on {dataset_meta.dataset_name}")

            result = run_retrieve_task(dataset_meta, trainer, encoder, retriever, args)
            eval_results[dataset_meta.dataset_name] = result
    
        metric = "ndcg_at_10"

        # Print the results
        if trainer.is_global_zero:
            trainer.print("=" * 40)
            trainer.print(args.save_path)
            trainer.print("=" * 40)
            for task in eval_results.keys():
                if metric in eval_results[task]:
                    trainer.print(f"{task:<32}{eval_results[task][metric]:.4f}")

    _compile_results()
    _dump_model_meta(output_dir=args.save_path)


if __name__ == "__main__":
    args = get_args()

    # Handle listing flags
    if args.list_models or args.list_tasks:
        if args.list_models:
            list_available_models()
            # Add a separator if both flags are passed
            if args.list_tasks:
                print("\n" + "=" * 80 + "\n")
        if args.list_tasks:
            list_available_tasks()
    else:
        main(args)
