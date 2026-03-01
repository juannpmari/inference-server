"""GPU ↔ CPU memory transfer handlers.

Provides an ABC with two implementations:
- MockGPUTransferHandler: simulates transfers with asyncio.sleep (no GPU needed)
- CuPyGPUTransferHandler: real CUDA transfers via CuPy pinned memory
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import NamedTuple

logger = logging.getLogger(__name__)


class TransferStatus(NamedTuple):
    success: bool
    message: str


class GPUTransferHandler(ABC):
    """Abstract base for HBM ↔ DRAM transfer implementations."""

    @abstractmethod
    async def copy_hbm_to_dram(self, hbm_addr: int, cpu_addr: int, size: int) -> TransferStatus:
        """Copy data from GPU HBM to CPU DRAM."""

    @abstractmethod
    async def copy_dram_to_hbm(self, cpu_addr: int, hbm_addr: int, size: int) -> TransferStatus:
        """Copy data from CPU DRAM to GPU HBM."""


class MockGPUTransferHandler(GPUTransferHandler):
    """Simulates GPU transfers with a short sleep. No GPU required."""

    async def copy_hbm_to_dram(self, hbm_addr: int, cpu_addr: int, size: int) -> TransferStatus:
        await asyncio.sleep(0.001)
        return TransferStatus(True, f"Mock offloaded {size} bytes to CPU: {hex(cpu_addr)}")

    async def copy_dram_to_hbm(self, cpu_addr: int, hbm_addr: int, size: int) -> TransferStatus:
        await asyncio.sleep(0.001)
        return TransferStatus(True, f"Mock loaded {size} bytes to GPU: {hex(hbm_addr)}")


class CuPyGPUTransferHandler(GPUTransferHandler):
    """Real CUDA transfers using CuPy with pinned memory and async streams."""

    def __init__(self):
        try:
            import cupy as cp
            self._cp = cp
            self._stream = cp.cuda.Stream(non_blocking=True)
            logger.info("CuPyGPUTransferHandler initialized with CUDA stream")
        except ImportError:
            raise ImportError(
                "CuPy is required for real GPU transfers. "
                "Install with: pip install cupy-cuda12x (adjust for your CUDA version)"
            )

    async def copy_hbm_to_dram(self, hbm_addr: int, cpu_addr: int, size: int) -> TransferStatus:
        """Device-to-Host async copy via CuPy."""
        cp = self._cp
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._sync_d2h, hbm_addr, cpu_addr, size)
            return TransferStatus(True, f"Offloaded {size} bytes to CPU: {hex(cpu_addr)}")
        except Exception as e:
            return TransferStatus(False, f"D2H transfer failed: {e}")

    async def copy_dram_to_hbm(self, cpu_addr: int, hbm_addr: int, size: int) -> TransferStatus:
        """Host-to-Device async copy via CuPy."""
        cp = self._cp
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._sync_h2d, cpu_addr, hbm_addr, size)
            return TransferStatus(True, f"Loaded {size} bytes to GPU: {hex(hbm_addr)}")
        except Exception as e:
            return TransferStatus(False, f"H2D transfer failed: {e}")

    def _sync_d2h(self, hbm_addr: int, cpu_addr: int, size: int) -> None:
        """Synchronous device-to-host copy (runs in thread executor)."""
        cp = self._cp
        with self._stream:
            src = cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(hbm_addr, size, None), 0)
            dst = cp.cuda.PinnedMemoryPointer(cp.cuda.UnownedMemory(cpu_addr, size, None), 0)
            dst.copy_from_device_async(src, size, self._stream)
            self._stream.synchronize()

    def _sync_h2d(self, cpu_addr: int, hbm_addr: int, size: int) -> None:
        """Synchronous host-to-device copy (runs in thread executor)."""
        cp = self._cp
        with self._stream:
            src = cp.cuda.PinnedMemoryPointer(cp.cuda.UnownedMemory(cpu_addr, size, None), 0)
            dst = cp.cuda.MemoryPointer(cp.cuda.UnownedMemory(hbm_addr, size, None), 0)
            dst.copy_from_host_async(src, size, self._stream)
            self._stream.synchronize()


def create_transfer_handler(mock: bool = True) -> GPUTransferHandler:
    """Factory: returns Mock or CuPy handler based on environment."""
    if mock:
        return MockGPUTransferHandler()
    return CuPyGPUTransferHandler()
