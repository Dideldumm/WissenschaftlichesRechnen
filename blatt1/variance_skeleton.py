#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


def compute_variance_1(data: np.ndarray) -> np.float64:
    """
    Compute variance of given data array using basic 2-pass algorithm

    Exceptions: ValueError when data is not 1-dimensional;

    return : variance
    """
    n = len(data)
    expectation = __compute_expectation(data)
    variance = sum([pow(x - expectation, 2) for x in data]) / n
    return variance


def compute_variance_2(data: np.ndarray) -> np.float64:
    """
    Compute variance of given data array using 1-pass algorithm

    Exceptions: ValueError when data is not 1-dimensional;

    return : variance
    """
    n = len(data)
    variance = sum([pow(x - __compute_expectation(data), 2) for x in data]) / n
    return variance


def compute_variance_2_kahan(data: np.ndarray) -> np.float64:
    """
    Compute variance of given data array using 1-pass algorithm and Kahan's algorithm for
    the summation

    Exceptions: ValueError when data is not 1-dimensional;

    return : variance
    """
    n = len(data)
    expectation = __compute_expectation(data)
    variance = np.divide(__add_kahan(np.array([pow(x - expectation, 2) for x in data])), n)
    return variance


def __add_kahan(data: np.ndarray) -> np.float64:
    s = data[0]
    c = 0
    for xj in data[1:]:
        y = xj - c
        t = s + y
        c = (t - s) - y
        s = t
    return s


def load_temperature_data(filename: str) -> np.ndarray:
    """
    Load temperature data from given file name

    return : data (as double precision np.array)
    """

    data = np.fromfile(filename, dtype=float)

    return data


def compute_variances() -> np.ndarray:
    """
    Compute variances with compute_variance_1(), compute_variance_2(), compute_variance_2_Kahan()
    for different latitudes.

    return : variances for different algorithms (rows) and different latitudes (columns)
    """

    vars = np.zeros((3, 5))

    # TODO

    return vars


def compute_rel_errs() -> np.ndarray:
    """
    Compute relative errors for variances obtained in compute_variances()

    return : relative errors for different algorithms (rows) and different latitudes (columns)
    """

    # TODO

    return np.array([1., 1., 1.])


def compute_mean_temperatures_yearly() -> np.ndarray:
    """
    Compute mean temperature of data per latitude

    return: yearly mean temperatures for different latitudes (rows) and years (columns)
    """

    num_lats = 5
    num_years = 30

    mean_temps = np.zeros((num_lats, num_years))

    return mean_temps


def __compute_expectation(data: np.ndarray) -> np.float64:
    """
    Compute expectation of data
    """
    expectation = np.divide(__add_kahan(np.array([x for x in data])), len(data))
    return expectation


def main():
    data = load_temperature_data("./temperature_locs.dat")
    print(data)
    print(compute_variance_1(data))
    print(compute_variance_2(data))
    print(compute_variance_2_kahan(data))


if __name__ == '__main__':
    main()
