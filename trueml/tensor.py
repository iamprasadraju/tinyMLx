from __future__ import annotations

import numpy as np

from trueml._uops import UOp


class Tensor:
    def __init__(self, arr, uops=True):
        self.arr = np.asarray(arr)

        self.shape = self.arr.shape
        self.ndim = self.arr.ndim
        self.dtype = self.arr.dtype
        self.uops = uops

        self.uop = None

        if uops:
            self.uop = UOp(
                op="LOAD",
                dtype=self.dtype,
                shape=self.shape,
                ndim=self.ndim,
            )

    @classmethod
    def ensure(cls, value, uops=False):
        if isinstance(value, cls):
            return value

        return cls(value, uops=uops)

    @classmethod
    def _make(cls, arr, op, *srcs, arg=None):
        arr = np.asarray(arr)

        if not all(src.uops for src in srcs):
            return cls(arr, uops=False)

        uop = UOp(
            op=op,
            dtype=arr.dtype,
            shape=arr.shape,
            ndim=arr.ndim,
            src=tuple(src.uop for src in srcs),
            arg=arg,
        )

        return cls(arr, uops=True)._replace_uop(uop)

    def _replace_uop(self, uop):
        self.uop = uop
        return self

    @classmethod
    def _constant(cls, value):
        tensor = cls(value, uops=True)

        tensor.uop = UOp(
            op="CONST",
            dtype=tensor.dtype,
            shape=tensor.shape,
            ndim=tensor.ndim,
            value=tensor.arr,
        )

        return tensor

    @property
    def size(self):
        return self.arr.size

    @property
    def metadata(self):
        return {
            "shape": self.shape,
            "ndim": self.ndim,
            "dtype": self.dtype,
        }

    def __add__(self, other):
        other = Tensor.ensure(other, uops=self.uops)

        return Tensor._make(
            self.arr + other.arr,
            "ADD",
            self,
            other,
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = Tensor.ensure(other, uops=self.uops)

        return Tensor._make(
            self.arr - other.arr,
            "SUB",
            self,
            other,
        )

    def __rsub__(self, other):
        other = Tensor.ensure(other, uops=self.uops)

        return Tensor._make(
            other.arr - self.arr,
            "SUB",
            other,
            self,
        )

    def __mul__(self, other):
        other = Tensor.ensure(other, uops=self.uops)

        return Tensor._make(
            self.arr * other.arr,
            "MUL",
            self,
            other,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __matmul__(self, other):
        other = Tensor.ensure(other, uops=self.uops)

        return Tensor._make(
            self.arr @ other.arr,
            "MATMUL",
            self,
            other,
        )

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        if out is not None:
            raise NotImplementedError("out is not supported")

        return Tensor._make(
            np.sum(
                self.arr,
                axis=axis,
                dtype=dtype,
                keepdims=keepdims,
            ),
            "SUM",
            self,
            arg={
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
            },
        )

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        if out is not None:
            raise NotImplementedError("out is not supported")

        return Tensor._make(
            np.mean(
                self.arr,
                axis=axis,
                dtype=dtype,
                keepdims=keepdims,
            ),
            "MEAN",
            self,
            arg={
                "axis": axis,
                "dtype": dtype,
                "keepdims": keepdims,
            },
        )

    @property
    def T(self):
        return Tensor._make(
            self.arr.T,
            "TRANSPOSE",
            self,
        )

    def __repr__(self):
        return (
            f"<Tensor {self.ndim}D shape={self.shape} dtype={self.dtype}>\n{self.arr}"
        )
