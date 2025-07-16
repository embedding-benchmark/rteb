import argparse
from pathlib import Path
import os
import json

from beir.retrieval.evaluation import EvaluateRetrieval
import pytorch_lightning as pl
from termcolor import colored

from ebr.core import Encoder
from ebr.core.data import RetrieveDataModule
from ebr.core.meta import DatasetMeta
from ebr.utils.memory import force_garbage_collection


CORPUS_EMBD_FILENAME = "corpus_embds.jsonl"
QUERIES_EMBD_FILENAME = "queries_embds.jsonl"
RETRIEVE_EVAL_FILENAME = "retrieve_eval.json"
RETRIEVE_PRED_FILENAME = "retrieve_pred.json"


def run_retrieve_evaluation(relevance, prediction):
    if len(relevance) != len(prediction):
        raise RuntimeError("Prediction and ground truth have different sizes.")
    
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        relevance, prediction, k_values=[1,3,5,10,20,50,100], ignore_identical_ids=False
    )
    scores = {
        **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
        **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
        **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
        **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
    }
    return scores


def run_retrieve_task(
    dataset_meta: DatasetMeta,
    trainer: pl.Trainer,
    encoder: Encoder,
    retriever: pl.LightningModule,
    args: argparse.Namespace
):
    dataset_name = dataset_meta.dataset_name

    task_save_path = Path(args.save_path) / dataset_name / encoder.model._id
    task_save_path.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        eval_file = task_save_path / RETRIEVE_EVAL_FILENAME
        pred_file = task_save_path / RETRIEVE_PRED_FILENAME
        if eval_file.exists():
            with open(str(eval_file)) as f:
                scores = json.load(f)
            return scores
        else:
            if pred_file.exists():
                return

    # DataModule manages the datasets
    dataset_kwargs = {
        "query_instruct": encoder.model.query_instruct,
        "corpus_instruct": encoder.model.corpus_instruct
    }
    collator_kwargs = {}

    dm = RetrieveDataModule(
        data_path=args.data_path,
        dataset_name=dataset_name,
        batch_size=args.batch_size,
        embd_batch_size=args.embd_batch_size,
        num_workers=args.num_workers,
        dataset_kwargs=dataset_kwargs,
        collator_kwargs=collator_kwargs,
    )
    if trainer.is_global_zero:
        dm.prepare_data()
        trainer.print("Queries size:", len(dm.dataset.queries))
        trainer.print("Corpus size:", len(dm.dataset.corpus))
    
    trainer.strategy.barrier()

    if len(dm.dataset.queries) < trainer.num_devices or len(dm.dataset.corpus) < trainer.num_devices:
        trainer.print(colored("Skipping the task due to too few queries / documents.", "red"))
        return {}

    if len(dm.dataset.queries) >= 1e6:
        trainer.print(colored("Skipping the task due to too many queries.", "red"))
        return {}

    if dataset_name == "bm25":
        # Build the index from corpus
        retriever.build_index(dm.dataset.corpus)
        # Compute the scores for queries
        retriever.save_file = os.path.join(task_save_path, RETRIEVE_PRED_FILENAME)
        trainer.predict(model=retriever, dataloaders=dm.queries_dataloader())
    
    else:
        # Step 1: Encode queries (always save to disk, never keep in memory)
        trainer.print(colored("Encode queries", "yellow"))
        encoder.is_query = True
        encoder.expected_dataset_size = len(dm.dataset.queries)
        encoder.save_file = os.path.join(task_save_path, QUERIES_EMBD_FILENAME)
        
        # Check if encoding is needed (handles both missing files and incomplete files)
        need_encoding = True
        if args.load_embds:
            # Load existing embeddings to check completeness (safe to call before trainer attachment)
            encoder.load_existing_embeddings(trainer.num_devices)
            if hasattr(encoder, 'local_existing_ids') and len(encoder.local_existing_ids) == encoder.expected_dataset_size:
                trainer.print(f"All {len(encoder.local_existing_ids)} query embeddings already exist, skipping encoding")
                need_encoding = False
            elif hasattr(encoder, 'local_existing_ids') and len(encoder.local_existing_ids) > 0:
                missing_count = encoder.expected_dataset_size - len(encoder.local_existing_ids)
                trainer.print(f"Resuming query encoding: {len(encoder.local_existing_ids)}/{encoder.expected_dataset_size} exist, generating {missing_count} missing embeddings")
        
        if need_encoding:
            if not args.load_embds:
                trainer.print(f"Encoding queries to disk...")
            # Use filtered dataloader to exclude already processed embeddings when resuming
            exclude_ids = getattr(encoder, 'local_existing_ids', None) if args.load_embds else None
            trainer.predict(model=encoder, dataloaders=dm.queries_dataloader(exclude_ids=exclude_ids))
        queries_embds_files = encoder.get_embd_files(trainer.num_devices)
        
        # Force garbage collection after query encoding
        force_garbage_collection()
        
        # Step 2: Encode corpus (always save to disk, never keep in memory)
        trainer.print(colored("Encode corpus", "yellow"))
        encoder.is_query = False
        encoder.expected_dataset_size = len(dm.dataset.corpus)
        encoder.save_file = os.path.join(task_save_path, CORPUS_EMBD_FILENAME)
        
        # Check if encoding is needed (handles both missing files and incomplete files)
        need_encoding = True
        if args.load_embds:
            # Load existing embeddings to check completeness (safe to call before trainer attachment)
            encoder.load_existing_embeddings(trainer.num_devices)
            if hasattr(encoder, 'local_existing_ids') and len(encoder.local_existing_ids) == encoder.expected_dataset_size:
                trainer.print(f"All {len(encoder.local_existing_ids)} corpus embeddings already exist, skipping encoding")
                need_encoding = False
            elif hasattr(encoder, 'local_existing_ids') and len(encoder.local_existing_ids) > 0:
                missing_count = encoder.expected_dataset_size - len(encoder.local_existing_ids)
                trainer.print(f"Resuming corpus encoding: {len(encoder.local_existing_ids)}/{encoder.expected_dataset_size} exist, generating {missing_count} missing embeddings")
        
        if need_encoding:
            if not args.load_embds:
                trainer.print(f"Encoding corpus to disk...")
            # Use filtered dataloader to exclude already processed embeddings when resuming
            exclude_ids = getattr(encoder, 'local_existing_ids', None) if args.load_embds else None
            trainer.predict(model=encoder, dataloaders=dm.corpus_dataloader(exclude_ids=exclude_ids))
        corpus_embds_files = encoder.get_embd_files(trainer.num_devices)
        
        # Force garbage collection after corpus encoding
        force_garbage_collection()

        # Step 3: Always offload model to save memory before retrieval
        trainer.print(colored("Offloading model to save memory before retrieval...", "yellow"))
        encoder.offload_model()
        
        # Additional garbage collection after model offloading
        force_garbage_collection()

        # Step 4: Load embeddings from disk for retrieval
        trainer.print(colored("Loading embeddings from disk for retrieval...", "yellow"))
        dm.set_queries_embds(queries_embds_files=queries_embds_files)
        dm.set_corpus_embds(corpus_embds_files=corpus_embds_files)

        # Step 5: Run retrieval
        trainer.print(colored("Retrieve", "yellow"))
        retriever.corpus_embd_dataloader = dm.corpus_embd_dataloader()
        retriever.in_memory = False  # Always use disk-based retrieval for memory efficiency
        retriever.save_file = os.path.join(task_save_path, RETRIEVE_PRED_FILENAME)
        trainer.predict(model=retriever, dataloaders=dm.queries_embd_dataloader())
        
        # Clear embedding datasets from data module and force garbage collection
        dm.queries_embd_ds = None
        dm.corpus_embd_ds = None
        if hasattr(dm, 'queries_embds'):
            dm.queries_embds = None
        if hasattr(dm, 'corpus_embds'):
            dm.corpus_embds = None
        force_garbage_collection()
        trainer.print("Embedding datasets cleared from memory")
        
        # Step 6: Cleanup embedding files after successful retrieval (unless --keep_embds)
        if not args.keep_embds and trainer.is_global_zero:
            trainer.print(colored("Cleaning up embedding files...", "yellow"))
            for file in queries_embds_files + corpus_embds_files:
                if os.path.exists(file):
                    os.remove(file)
                    trainer.print(f"Removed: {file}")

    # Run evaluation
    if trainer.is_global_zero:
        scores = run_retrieve_evaluation(dm.dataset.relevance, retriever.prediction)
        trainer.print("-" * 40)
        trainer.print("Dataset:", colored(f"{dataset_name}", "red"))
        
        trainer.print("Model:", colored(f"{encoder.model.model_name}", "red"))
        model_identifier = encoder.model.alias or encoder.model.model_name
        model_embd_dim = encoder.model.embd_dim
        model_embd_dtype = encoder.model.embd_dtype
        
        trainer.print("Save path:", colored(task_save_path, "yellow"))
        trainer.print("Retrieval evaluation:")
        trainer.print(scores)
        scores |= {
            "model_name": model_identifier,
            "embd_dim": model_embd_dim,
            "embd_dtype": model_embd_dtype
        }
        with open(os.path.join(task_save_path, RETRIEVE_EVAL_FILENAME), "w") as f:
            json.dump(scores, f)
        trainer.print(os.path.join(task_save_path, RETRIEVE_EVAL_FILENAME))
        return scores

    return
