from src.data.model import BenchmarkItem, BenchmarkTask

# BenchmarkStore imported lazily to avoid pulling in `datasets` when not needed
def __getattr__(name):
    if name == "BenchmarkStore":
        from src.data.store import BenchmarkStore
        return BenchmarkStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["BenchmarkItem", "BenchmarkTask", "BenchmarkStore"]
