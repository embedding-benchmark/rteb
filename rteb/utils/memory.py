"""Memory management utilities for efficient cleanup and garbage collection."""

import gc
import logging

import torch

logger = logging.getLogger(__name__)


def force_garbage_collection():
    """Force garbage collection and clear CUDA cache for memory efficiency.
    
    This function performs comprehensive memory cleanup by:
    1. Running Python's garbage collector to free unreferenced objects
    2. Clearing PyTorch's CUDA memory cache to free GPU memory fragments
    
    Should be called after major memory drops like model offloading,
    embedding cleanup, or between processing phases.
    """
    # Force Python garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    logger.debug("Garbage collection and CUDA cache clearing completed")