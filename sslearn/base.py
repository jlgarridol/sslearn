from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import MetaEstimatorMixin

class Ensemble(ABC, MetaEstimatorMixin):

    @abstractmethod
    def predict_proba(self, X, **kwards):
        pass

    def predict(self, X, **kwards):
        """Predict the classes of X.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Array representing the data.
        Returns
        -------
        y : ndarray of shape (n_samples,)
            Array with predicted labels.
        """
        predicted_probabilitiy = self.predict_proba(X, **kwards)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)),
                                  axis=0)