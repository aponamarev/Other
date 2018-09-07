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
    :param clusters: Mean and std for each cluster ((mean1, var1), (mean2, var2),..)
    :param num_points: Number of points to be generated for each cluster
    :return: data points
    :rtype: tuple
    """
    # Check input values
    assert len(clusters)>0, "At lease one cluster should be provided"
    clusters = tuple(filter(lambda c: len(c) == 2, clusters))
    data_points: tuple = tuple(map(lambda c: np.random.normal(loc=c[0], scale=c[1]**0.5, size=num_points), clusters))
    return data_points

def gaussian_probability(x: float, mean: float, var: float):
    var2 = 2.0 * var
    p = np.exp(-np.square(x-mean) / var2) / np.sqrt(np.pi*var2)
    return p

def expectation_maximization(data: np.ndarray, n_clusters: int) -> tuple:
    clusters = tuple( (np.random.rand(), np.random.rand()**2) for _ in range(n_clusters))
    print("0: c1m: {:.2f}, c1v: {:.2f}, c2m: {:.2f}, c2v: {:.2f}".format(clusters[0][0], clusters[0][1], clusters[1][0],
                                                                         clusters[1][1]))
    n: int = 0
    # [1, n_clusters]
    P_c = np.reshape([1 / n_clusters] * n_clusters, [1, -1])
    while n<20:
        # [n, n_clusters]
        P_x_c = np.concatenate(tuple(map(lambda c: gaussian_probability(data, c[0], c[1]), clusters)), axis=-1)
        # [n, 1]
        P_c_given_x = P_x_c * P_c / np.sum(P_x_c * P_c, axis=1, keepdims=True)

        # m = b1x1+b2x2..+..bnxn / b1 + b2 + .. + bn
        means = np.dot(data.T, P_c_given_x) / np.sum(P_c_given_x, axis=0, keepdims=True)
        # S = b1(x1-m1)^2 + .. + bn(xn-mn)^2 / b1 + b2 + .. + bn
        var = np.sum(P_c_given_x * np.square(data-means), 0, keepdims=True) / np.sum(P_c_given_x, axis=0, keepdims=True)
        clusters = np.concatenate([means, var], axis=0).T
        print("{}: c1m: {:.2f}, c1v: {:.2f}, c2m: {:.2f}, c2v: {:.2f}".format(n, clusters[0][0], clusters[0][1],
                                                                              clusters[1][0], clusters[1][1]))
        n += 1

if __name__ == '__main__':
    orig_clusters = ((0.5, 2.0), (-3.0, 0.5))
    print("c1m: {}, c1v: {}, c2m: {}, c2v: {}".format(orig_clusters[0][0],orig_clusters[0][1],orig_clusters[1][0],
                                                      orig_clusters[1][1]))
    data = generate_data(orig_clusters, 100)
    flat_data = np.reshape(np.array(data), [-1,1])
    expectation_maximization(flat_data, 2)
