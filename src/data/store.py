"""Unified interface for loading, filtering, and accessing benchmark tasks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from src.data.loaders import load_ds1000, load_humaneval, load_mbpp
from src.data.model import BenchmarkTask


class BenchmarkStore:
    """Unified access to HumanEval, MBPP, and DS-1000 tasks.

    Usage:
        store = BenchmarkStore()                  # loads nothing yet
        store.load("humaneval")                   # load one source
        store.load_all()                          # load all three

        task = store.get("HumanEval/0")           # by exact ID
        tasks = store.filter(source="humaneval")  # by source
        tasks = store.filter(library="Pandas")    # DS-1000 library
    """

    _LOADERS = {
        "humaneval": load_humaneval,
        "mbpp": load_mbpp,
        "ds1000": load_ds1000,
    }

    def __init__(self):
        self._tasks: dict[str, BenchmarkTask] = {}
        self._loaded_sources: set[str] = set()

    def load(self, source: str, **kwargs) -> int:
        """Load tasks from a single source. Returns number of tasks loaded."""
        if source not in self._LOADERS:
            raise ValueError(
                f"Unknown source '{source}'. Choose from: {list(self._LOADERS)}"
            )
        loader = self._LOADERS[source]
        tasks = loader(**kwargs)
        for t in tasks:
            self._tasks[t.task_id] = t
        self._loaded_sources.add(source)
        return len(tasks)

    def load_all(self) -> dict[str, int]:
        """Load all benchmark sources. Returns {source: count} mapping."""
        return {source: self.load(source) for source in self._LOADERS}

    def get(self, task_id: str) -> BenchmarkTask:
        """Get a single task by its exact ID. Raises KeyError if not found."""
        return self._tasks[task_id]

    def filter(
        self,
        source: Optional[str] = None,
        library: Optional[str] = None,
    ) -> list[BenchmarkTask]:
        """Filter tasks by source and/or library."""
        tasks = list(self._tasks.values())
        if source:
            tasks = [t for t in tasks if t.source == source]
        if library:
            tasks = [t for t in tasks if t.library == library]
        return tasks

    def all_tasks(self) -> list[BenchmarkTask]:
        """Return all loaded tasks."""
        return list(self._tasks.values())

    @property
    def loaded_sources(self) -> set[str]:
        return set(self._loaded_sources)

    def __len__(self) -> int:
        return len(self._tasks)

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, data_dir: str | Path) -> dict[str, int]:
        """Save all loaded tasks to JSONL files, one per source.

        Creates:
            data_dir/humaneval.jsonl
            data_dir/mbpp.jsonl
            data_dir/ds1000.jsonl

        Returns {source: count} written.
        """
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

        counts: dict[str, int] = {}
        # Group tasks by source
        by_source: dict[str, list[BenchmarkTask]] = {}
        for t in self._tasks.values():
            by_source.setdefault(t.source, []).append(t)

        for source, tasks in by_source.items():
            path = data_dir / f"{source}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for t in tasks:
                    f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
            counts[source] = len(tasks)

        return counts

    @classmethod
    def load_local(cls, data_dir: str | Path) -> BenchmarkStore:
        """Load tasks from previously saved JSONL files.

        Reads all *.jsonl files in data_dir.
        """
        store = cls()
        data_dir = Path(data_dir)
        for path in sorted(data_dir.glob("*.jsonl")):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    task = BenchmarkTask.from_dict(json.loads(line))
                    store._tasks[task.task_id] = task
                    store._loaded_sources.add(task.source)
        return store

    def __repr__(self) -> str:
        counts = {}
        for t in self._tasks.values():
            counts[t.source] = counts.get(t.source, 0) + 1
        parts = [f"{s}={n}" for s, n in sorted(counts.items())]
        return f"BenchmarkStore({', '.join(parts)}, total={len(self)})"
