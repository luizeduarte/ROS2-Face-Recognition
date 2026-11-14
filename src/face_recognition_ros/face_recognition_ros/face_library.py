"""Open-set face identification by cosine similarity against enrolled embeddings.

This module has no ROS dependency so it can be unit-tested and reused on its own.

Why not a classifier (e.g. an SVM)? A closed-set classifier always picks one of the
known identities, so it cannot say "this is nobody I know", and it needs at least two
identities and a retrain every time someone is enrolled. Comparing L2-normalised
embeddings with a similarity threshold handles unknown faces, works from the first
enrolled person, and enrolling is just appending a row.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

UNKNOWN = 'unknown'


@dataclass(frozen=True)
class Match:
    label: str
    score: float

    @property
    def is_known(self) -> bool:
        return self.label != UNKNOWN


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class FaceLibrary:
    """A set of labelled face embeddings with nearest-identity lookup.

    Each identity may have several embeddings (different poses / lighting); a query is
    scored against every one and the best score per identity wins.
    """

    def __init__(self, threshold: float = 0.363, dim: int | None = None):
        # 0.363 is the cosine threshold recommended for OpenCV's SFace model on LFW.
        self.threshold = threshold
        self._dim = dim
        self._embeddings = np.empty((0, dim or 0), dtype=np.float32)
        self._labels: list[str] = []

    def __len__(self) -> int:
        return len(self._labels)

    @property
    def identities(self) -> list[str]:
        return sorted(set(self._labels))

    def add(self, label: str, embedding: np.ndarray) -> None:
        label = label.strip()
        if not label or label == UNKNOWN:
            raise ValueError(f'invalid label: {label!r}')
        vec = _normalize(embedding)
        if vec.shape[0] != 1:
            raise ValueError('add() expects a single embedding')
        if self._dim is None:
            self._dim = vec.shape[1]
            self._embeddings = np.empty((0, self._dim), dtype=np.float32)
        elif vec.shape[1] != self._dim:
            raise ValueError(f'embedding has dim {vec.shape[1]}, library expects {self._dim}')
        self._embeddings = np.vstack([self._embeddings, vec])
        self._labels.append(label)

    def remove(self, label: str) -> int:
        """Remove every embedding of ``label``; returns how many were removed."""
        keep = [i for i, lbl in enumerate(self._labels) if lbl != label]
        removed = len(self._labels) - len(keep)
        self._embeddings = self._embeddings[keep]
        self._labels = [self._labels[i] for i in keep]
        return removed

    def identify(self, embedding: np.ndarray) -> Match:
        return self.identify_batch(embedding)[0]

    def identify_batch(self, embeddings: np.ndarray) -> list[Match]:
        queries = _normalize(embeddings)
        if not self._labels:
            return [Match(UNKNOWN, 0.0) for _ in range(queries.shape[0])]
        if queries.shape[1] != self._dim:
            raise ValueError(f'embedding has dim {queries.shape[1]}, library expects {self._dim}')
        scores = queries @ self._embeddings.T  # cosine similarity, shape (queries, enrolled)
        best = scores.argmax(axis=1)
        matches = []
        for row, col in enumerate(best):
            score = float(scores[row, col])
            label = self._labels[col] if score >= self.threshold else UNKNOWN
            matches.append(Match(label, score))
        return matches

    def save(self, path: str | Path) -> None:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        # np.savez appends .npz unless the name already ends with it; write to a temp
        # file then rename so a crash never leaves a half-written library behind.
        tmp = path.with_name(path.name + '.tmp.npz')
        np.savez(tmp, embeddings=self._embeddings, labels=np.array(self._labels, dtype=str))
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path, threshold: float = 0.363) -> FaceLibrary:
        with np.load(Path(path).expanduser(), allow_pickle=False) as data:
            embeddings = data['embeddings'].astype(np.float32)
            labels = [str(lbl) for lbl in data['labels']]
        lib = cls(threshold=threshold, dim=embeddings.shape[1] if len(labels) else None)
        for label, emb in zip(labels, embeddings):
            lib.add(label, emb)
        return lib

    @classmethod
    def load_or_create(cls, path: str | Path, threshold: float = 0.363) -> FaceLibrary:
        path = Path(path).expanduser()
        return cls.load(path, threshold) if path.exists() else cls(threshold=threshold)
