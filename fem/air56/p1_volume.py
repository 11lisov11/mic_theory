"""Preassembled P1 element gradients and sparse scatter for repeated nonlinear FEM solves."""

import numpy as np
from scipy.sparse import csc_matrix
from fem.air56.loaded_section import material


class P1Volume:
    def __init__(self, field):
        b = field.b
        self.n = len(field.free)
        idx = np.full(b.N, self.n)
        idx[field.free] = np.arange(self.n)
        self.id = idx[b.element_dofs]
        self.gx, self.gy = np.stack([basis[0].grad[:, :, 0] for basis in b.basis], axis=1)
        self.area = b.dx.sum(axis=1)
        self.steel = field.steel
        row = np.broadcast_to(self.id[:, None, :], (3, 3, self.id.shape[1]))
        col = row.swapaxes(0, 1)
        kept = (row < self.n) & (col < self.n)
        self.parts = np.nonzero(kept)[2]
        i, j, k = np.nonzero(kept)
        self.xx = self.gx[i, k] * self.gx[j, k] * self.area[k]
        self.xy = (self.gx[i, k] * self.gy[j, k] + self.gy[i, k] * self.gx[j, k]) * self.area[k]
        self.yy = self.gy[i, k] * self.gy[j, k] * self.area[k]
        keys, self.scatter = np.unique(row[kept] + self.n * col[kept], return_inverse=True)
        self.rows = (keys % self.n).astype("int32")
        self.ptr = np.r_[0, np.cumsum(np.bincount(keys // self.n, minlength=self.n))].astype("int32")

    def field(self, a):
        v = np.r_[a, 0.0][self.id]
        gx = np.sum(self.gx * v, axis=0)
        gy = np.sum(self.gy * v, axis=0)
        nu, t = material(gx, gy, self.steel)
        values = (self.gx * gx + self.gy * gy) * (nu * self.area)
        return np.bincount(self.id.ravel(), weights=values.ravel(), minlength=self.n + 1)[: self.n], t

    def matrix(self, t):
        k = self.parts
        data = np.bincount(
            self.scatter, weights=self.xx * t[0][k] + self.xy * t[1][k] + self.yy * t[2][k]
        )
        return csc_matrix((data, self.rows, self.ptr), shape=(self.n, self.n))
