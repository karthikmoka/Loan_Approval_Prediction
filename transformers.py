from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class IQRClipper(BaseEstimator, TransformerMixin):
    """
    Custom transformer for handling outliers
    using the IQR (Inter Quartile Range) method.

    Values outside the lower and upper limits
    are clipped instead of removing rows.
    """

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

        for column in X.columns:

            values = pd.to_numeric(
                X[column],
                errors="coerce"
            )

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)

            iqr = q3 - q1

            # Skip if IQR is zero
            if iqr == 0:
                continue

            lower = q1 - self.factor * iqr
            upper = q3 + self.factor * iqr

            self.lower_bounds_[column] = lower
            self.upper_bounds_[column] = upper

        return self

    def transform(self, X):

        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)

        X = X.copy()

        for column in X.columns:

            if column in self.lower_bounds_:

                X[column] = pd.to_numeric(
                    X[column],
                    errors="coerce"
                ).clip(
                    lower=self.lower_bounds_[column],
                    upper=self.upper_bounds_[column]
                )

        return X