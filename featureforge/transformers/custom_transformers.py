"""Custom transformer extension points for future package users."""

from __future__ import annotations

from sklearn.base import BaseEstimator, TransformerMixin


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """A tiny pass-through transformer useful in tests and pipeline assembly."""

    def fit(self, X, y=None):  # noqa: N803 - sklearn uses X/y names.
        return self

    def transform(self, X):  # noqa: N803 - sklearn uses X/y names.
        return X
