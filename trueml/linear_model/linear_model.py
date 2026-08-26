from __future__ import annotations

import numpy as np

from trueml.history import History
from trueml.tensor import Tensor


class LinearModel:
    def __init__(self, n_features, lr=0.01, history=True, uops=False):
        self.lr = lr
        self.uops = uops

        self.weights = Tensor(np.random.random(n_features), uops=self.uops)
        self.bias = Tensor(0.0, uops=self.uops)
        self.history = History() if history else None

    def forward(self, X_train):
        if not isinstance(X_train, Tensor):
            X_train = Tensor(X_train, uops=self.uops)

        self.X_train = X_train
        return self.X_train @ self.weights + self.bias

    def backward(self, dL_dy_pred):
        """
        ŷ = Xw + b

        dŷ/dw = X
        dŷ/db = 1

        Loss = L(y_true, ŷ)

        dL/dw = dL/dŷ · dŷ/dw
              = X.T @ dL/dŷ

        dL/db = sum(dL/dŷ)
        """

        # dw = ∂L/∂w
        # db = ∂L/∂b
        dw = self.X_train.T @ dL_dy_pred
        db = np.sum(dL_dy_pred)

        self.weights -= self.lr * dw
        self.bias -= self.lr * db

    def predict(self, X_test):
        return X_test @ self.weights + self.bias
