# ADA
# Project: Illustration for Expectation Maximization algorithm
# Description:
# Author: Alexander Ponamarev
# Created Date: 4/19/18
# IQVIA. All Rights Reserved.
import numpy as np
from functools import reduce


def generate_data(clusters: tuple, num_points: int = 20):
    """
    Generates points according to a random distribution based on cluster mean and std
    :param clusters: Mean and std for each cluster ((mean1, std1), (mean2, std2),..)
    :param num_points: Number of points to be generated for each cluster
    :return: data points
    :rtype: tuple
    """
    # Check input values
    assert len(clusters)>0, "At lease one cluster should be provided"
    clusters = tuple(filter(lambda c: len(c) == 2, clusters))
    data_points: tuple = tuple(map(lambda c: np.random.normal(loc=c[0], scale=c[1], size=num_points), clusters))
    return data_points

def gaussian_probability(x: float, mean: float, std: float):
    two_sigma_squared = 2.0 * std**2
    p = np.exp(-(x-mean)**2 / two_sigma_squared) / np.sqrt(np.pi*two_sigma_squared)
    return p

def expectation_maximization(data: np.ndarray, n_clusters: int) -> tuple:
    clusters = tuple(np.random.randn(2) for _ in range(n_clusters))
    n: int = 0
    while n<5:
        # [n, n_clusters]
        P_x_c = np.array(tuple(map(lambda c: gaussian_probability(data, c[0], c[1]), clusters))).T
        # [1, n_clusters]
        P_c = np.average(P_x_c, axis=0)
        # [n, 1]
        P_x = np.reshape(np.sum(P_x_c * P_c, axis=1), [-1,1])
        P_c_given_x = P_x_c * P_c / P_x

        # m = b1x1+b2x2..+..bnxn / b1 + b2 + .. + bn
        means = np.dot(data, P_c_given_x) / P_c.T
        # S = b1(x1-m1)^2 + .. + bn(xn-mn)^2 / b1 + b2 + .. + bn

if __name__ == '__main__':
    orig_clusters = ((0.5, 2), (-3.0, 0.5))
    data = generate_data(orig_clusters, 20)
    flat_data = np.reshape(np.array(data), -1)
    expectation_maximization(flat_data, 2)
