import numpy as np
from scipy.spatial.distance import cdist
from typing import List


class AbsKernel:
    def fit(self, data: np.ndarray, **kwargs) -> None:
        raise NotImplementedError

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinearKernel(AbsKernel):

    def __init__(self):
        super(LinearKernel, self).__init__()

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        return data1 @ data2.T


class BinaryKernel(AbsKernel):

    def __init__(self):
        super(BinaryKernel, self).__init__()

    def fit(self, data: np.ndarray, **kwargs):
        pass

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        res = data1.dot(data2.T)
        res += (1 - data1).dot(1 - data2.T)
        return res


class GaussianKernel(AbsKernel):
    sigma: np.float

    def __init__(self):
        super(GaussianKernel, self).__init__()

    def fit(self, data: np.ndarray, **kwargs) -> None:
        dists = cdist(data, data, 'sqeuclidean')
        self.sigma = np.median(dists)

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        dists = cdist(data1, data2, 'sqeuclidean')
        dists = dists / self.sigma
        return np.exp(-dists)


class CoordinateWiseProductKernel(AbsKernel):

    def __init__(self, kernel_list: List[AbsKernel], dim_list: List[int]):
        super(CoordinateWiseProductKernel, self).__init__()
        self.kernel_list = kernel_list,
        self.dims = [0] + list(np.cumsum(dim_list))

    def fit(self, data: np.ndarray, **kwargs) -> None:
        for i in range(len(self.kernel_list)):
            self.kernel_list[i].fit(data[:, self.dims[i]:self.dims[i + 1]])

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        final_mat = np.ones(data1.shape[0], data2.shape[0])
        for i in range(len(self.kernel_list)):
            data1_sub = data1[:, self.dims[i]:self.dims[i + 1]]
            data2_sub = data2[:, self.dims[i]:self.dims[i + 1]]
            final_mat = final_mat * self.kernel_list[i].cal_kernel_mat(data1_sub, data2_sub)

        return final_mat
