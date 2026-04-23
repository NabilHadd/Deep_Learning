

import numpy as np
import math
from collections import defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import validate_data, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class NaiveBayesClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, alpha=1.0, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior

    def fit(self, X, y):
        X, y = validate_data(self, X, y)

        self.classes_ = unique_labels(y)
        self.class_counts_ = defaultdict(int)
        self.feature_counts_ = defaultdict(lambda: defaultdict(int))
        self.total_ = len(y)
        # valores posibles por feature (mejor suavizado)
        self.feature_values_ = defaultdict(set)

        for xi, label in zip(X, y):
            self.class_counts_[label] += 1
            for i, val in enumerate(xi):
                self.feature_counts_[label][(i, val)] += 1
                self.feature_values_[i].add(val)

        return self

    def _log_prior(self, c):
        if self.fit_prior:
            return math.log(self.class_counts_[c] / self.total_)
        return math.log(1 / len(self.classes_))

    def _predict_one_log(self, xi):
        scores = []

        for c in self.classes_:
            logp = self._log_prior(c)

            for i, val in enumerate(xi):
                freq = self.feature_counts_[c][(i, val)] + self.alpha
                total = self.class_counts_[c] + self.alpha * len(self.feature_values_[i])
                logp += math.log(freq / total)

            scores.append(logp)

        return scores

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        preds = []
        for xi in X:
            scores = self._predict_one_log(xi)
            preds.append(self.classes_[np.argmax(scores)])

        return np.array(preds)

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        probas = []

        for xi in X:
            log_scores = self._predict_one_log(xi)

            max_log = max(log_scores)
            exp_scores = np.exp(np.array(log_scores) - max_log)
            probs = exp_scores / exp_scores.sum()

            probas.append(probs)

        return np.array(probas)