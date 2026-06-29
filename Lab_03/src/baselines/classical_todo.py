"""PCA + scikit-learn baseline — E1."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


class ClassicalBaseline:
    """Dimensionality reduction with PCA followed by classical estimators.

    A single StandardScaler + PCA pipeline is shared between the gender
    classifier and the age regressor to avoid computing two decompositions on
    the same input. Both estimators receive the same low-dimensional
    representation.

    The baseline is intentionally kept simple so that ablations (PCA
    components, classifier choice) are easy to isolate with a single change.

    Args:
        n_components: Number of PCA components kept after dimensionality
            reduction. Must be smaller than min(n_samples, n_features).
        gender_clf: ``"logistic"`` for LogisticRegression or
            ``"naive_bayes"`` for GaussianNB.
        age_reg: ``"ridge"`` for Ridge regression.
    """

    _GENDER_CLASSIFIERS: dict = {
        "logistic": lambda: LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "naive_bayes": lambda: GaussianNB(),
    }
    _AGE_REGRESSORS: dict = {
        "ridge": lambda: Ridge(alpha=1.0),
    }

    # Maximum components to consider when n_components is a variance threshold.
    # PCA(n_components=0.99) with svd_solver='full' over 150K features takes hours;
    # instead we fit randomized SVD up to this cap and find the variance cutoff.
    _MAX_RANDOMIZED_COMPONENTS = 500

    def __init__(
        self,
        n_components: int | float = 0.99,
        gender_clf: str = "logistic",
        age_reg: str = "ridge",
    ) -> None:
        if gender_clf not in self._GENDER_CLASSIFIERS:
            raise ValueError(
                f"gender_clf debe ser uno de {list(self._GENDER_CLASSIFIERS)}."
            )
        if age_reg not in self._AGE_REGRESSORS:
            raise ValueError(
                f"age_reg debe ser uno de {list(self._AGE_REGRESSORS)}."
            )

        self.n_components = n_components
        self.scaler = StandardScaler()

        if isinstance(n_components, float):
            # Two-pass strategy: randomized SVD up to _MAX_RANDOMIZED_COMPONENTS,
            # then truncate to the minimum components that meet the variance target.
            # This avoids the O(n_features^2 * n_samples) cost of full SVD.
            self._variance_threshold: float | None = n_components
            self.pca = PCA(
                n_components=self._MAX_RANDOMIZED_COMPONENTS,
                svd_solver="randomized",
                random_state=42,
            )
        else:
            self._variance_threshold = None
            self.pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)

        self.gender_clf = self._GENDER_CLASSIFIERS[gender_clf]()
        self.age_reg = self._AGE_REGRESSORS[age_reg]()

    def fit(
        self,
        X: np.ndarray,
        y_gender: np.ndarray,
        y_age: np.ndarray,
    ) -> None:
        """Fit scaler, PCA and both task estimators on flattened training images."""
        X_scaled = self.scaler.fit_transform(X)
        X_pca_full = self.pca.fit_transform(X_scaled)

        if self._variance_threshold is not None:
            # Find minimum k where cumulative explained variance >= threshold.
            cumvar = np.cumsum(self.pca.explained_variance_ratio_)
            k = int(np.searchsorted(cumvar, self._variance_threshold) + 1)
            k = min(k, X_pca_full.shape[1])
            self._n_components_used = k
            X_pca = X_pca_full[:, :k]
        else:
            self._n_components_used = X_pca_full.shape[1]
            X_pca = X_pca_full

        self.gender_clf.fit(X_pca, y_gender)
        self.age_reg.fit(X_pca, y_age)

    def predict(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (gender_predictions, age_predictions) for a feature matrix."""
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        if self._variance_threshold is not None:
            X_pca = X_pca[:, : self._n_components_used]
        return self.gender_clf.predict(X_pca), self.age_reg.predict(X_pca)

    def fit_from_loader(self, loader: DataLoader) -> None:
        """Extract features from a DataLoader and call ``fit``."""
        X, y_gender, y_age = _extract_arrays(loader)
        self.fit(X, y_gender, y_age)

    def predict_from_loader(
        self, loader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract features from a DataLoader and return predictions."""
        X, _, _ = _extract_arrays(loader)
        return self.predict(X)

    def targets_from_loader(
        self, loader: DataLoader
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ground-truth (gender, age) arrays from a DataLoader."""
        _, y_gender, y_age = _extract_arrays(loader)
        return y_gender, y_age


def _extract_arrays(
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten image tensors and collect labels from a DataLoader."""
    images_list, genders_list, ages_list = [], [], []
    for images, genders, ages in loader:
        images_list.append(images.view(images.size(0), -1).numpy())
        genders_list.append(genders.numpy())
        ages_list.append(ages.numpy())
    if not images_list:
        raise RuntimeError("El DataLoader está vacío.")
    return (
        np.concatenate(images_list),
        np.concatenate(genders_list),
        np.concatenate(ages_list),
    )
