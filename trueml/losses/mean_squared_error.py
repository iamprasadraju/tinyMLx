from __future__ import annotations

from trueml.tensor import Tensor


class MSELoss:
    def __init__(self, uops=False):
        self.uops = uops

    def __call__(self, y_true, y_pred):
        y_true = Tensor.ensure(y_true, uops=self.uops)
        y_pred = Tensor.ensure(y_pred, uops=self.uops)

        diff = y_pred - y_true

        return (diff * diff).mean()

    def grad(self, y_true, y_pred):
        y_true = Tensor.ensure(y_true, uops=self.uops)
        y_pred = Tensor.ensure(y_pred, uops=self.uops)

        n = y_true.size

        return (2 / n) * (y_pred - y_true)
