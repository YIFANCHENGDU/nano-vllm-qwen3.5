"""Layer-wise weight loader with optional async prefetching.

The loader reads ``layer_XXXX.pt`` files produced by
:mod:`nanovllm.utils.save_layers` and hands weight dictionaries to the
caller one layer at a time.  An optional background thread pre-loads the
*next* layer into CPU RAM while the *current* layer is executing on the GPU,
hiding most of the disk-to-RAM latency.
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor

import torch


class LayerWeightLoader:
    """On-demand loader for per-layer weight files with optional prefetch.

    Typical usage::

        loader = LayerWeightLoader(layer_dir, num_layers, device="cuda")
        for i in range(num_layers):
            loader.schedule_prefetch(i + 1)   # background-load next layer
            weights = loader.get(i)           # blocks until weights are ready
            # … run forward pass …
            loader.release(weights)           # free GPU memory

    Args:
        layer_dir:  Directory containing ``layer_XXXX.pt`` files.
        num_layers: Total number of decoder layers in the model.
        device:     Target device for returned tensors (e.g. ``'cuda'``).
        prefetch:   Enable background prefetching of the next layer into CPU
                    RAM while the current layer runs on the GPU.  Requires
                    roughly 2× per-layer CPU RAM but reduces latency by
                    overlapping I/O and compute.
    """

    def __init__(
        self,
        layer_dir: str,
        num_layers: int,
        device: str = "cuda",
        prefetch: bool = True,
    ) -> None:
        self.layer_dir = layer_dir
        self.num_layers = num_layers
        self.device = device
        self.prefetch = prefetch
        # layer_idx → Future[dict[str, Tensor]]
        self._futures: dict[int, Future] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="layer_prefetch"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _layer_path(self, layer_idx: int) -> str:
        return os.path.join(self.layer_dir, f"layer_{layer_idx:04d}.pt")

    def _load_from_disk(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Load a layer file from disk to CPU RAM (blocking)."""
        return torch.load(
            self._layer_path(layer_idx),
            map_location="cpu",
            weights_only=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def schedule_prefetch(self, layer_idx: int) -> None:
        """Asynchronously start loading ``layer_idx`` into CPU RAM.

        Safe to call even when ``layer_idx`` is out of range or already
        prefetching — duplicate or invalid calls are silently ignored.
        """
        if not self.prefetch:
            return
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        if layer_idx not in self._futures:
            self._futures[layer_idx] = self._executor.submit(
                self._load_from_disk, layer_idx
            )

    def get(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Return the weights for ``layer_idx`` on :attr:`device`.

        If a prefetch was scheduled the method waits for the background thread
        to finish (very fast if prefetch completed during previous layer
        execution); otherwise the load is performed synchronously.

        The returned tensors are moved to :attr:`device` via non-blocking
        host-to-device copies.
        """
        if layer_idx in self._futures:
            weights_cpu = self._futures.pop(layer_idx).result()
        else:
            weights_cpu = self._load_from_disk(layer_idx)
        return {
            k: v.to(self.device, non_blocking=True)
            for k, v in weights_cpu.items()
        }

    def release(self, weights: dict[str, torch.Tensor]) -> None:
        """Delete GPU tensors and encourage the CUDA allocator to free them.

        Callers should discard the ``weights`` reference after this call.
        """
        weights.clear()
        if self.device != "cpu":
            torch.cuda.empty_cache()

    def __del__(self) -> None:
        self._executor.shutdown(wait=False)
