from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class MSC(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.reference_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        corrected = []
        for spectrum in X:
            fit = np.polyfit(self.reference_, spectrum, 1)
            corrected.append((spectrum - fit[1]) / fit[0])
        return np.array(corrected)

class SNV(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):

        self.mean_ = np.mean(X)
        self.std_ = np.std(X)
        return self

    def transform(self, X):
        return (X - self.mean_) / self.std_

    def inverse_transform(self, X_snv):
        return X_snv * self.std_ + self.mean_