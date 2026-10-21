"""PyTorch 2.7 compatibility for the project's Transformers/DeepSpeed versions."""
import numpy as np
import torch


class CheckpointCompatibilityMixin:
    def _load_rng_state(self, checkpoint):
        # Transformers 4.46 stores a NumPy uint32 RNG array in this file.
        from importlib import import_module
        numpy_core = 'numpy._core' if int(np.__version__.split('.')[0]) >= 2 else 'numpy.core'
        reconstruct = import_module(numpy_core + '.multiarray')._reconstruct
        allowed = [
            (reconstruct, 'numpy.core.multiarray._reconstruct'),
            (reconstruct, 'numpy._core.multiarray._reconstruct'),
            np.ndarray, np.dtype, type(np.dtype('uint32')),
        ]
        with torch.serialization.safe_globals(allowed):
            return super()._load_rng_state(checkpoint)


def install_deepspeed_checkpoint_compatibility():
    """Keep full-state loading local to DeepSpeed's checkpoint engine module.

    DS 0.15 checkpoints contain optimizer Python objects, not just weight tensors.
    This proxy does not modify torch.load in the rest of the process.
    """
    import deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine as engine

    class CheckpointTorch:
        def __getattr__(self, name):
            return getattr(torch, name)

        def load(self, *args, **kwargs):
            kwargs['weights_only'] = False
            return torch.load(*args, **kwargs)

    engine.torch = CheckpointTorch()
