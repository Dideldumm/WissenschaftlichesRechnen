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
    minuend = 0
    subtrahend = 0
    for xi in data:
        minuend += np.square(xi)
        subtrahend += xi
    return np.divide(minuend - np.divide(np.square(subtrahend), n), n)


def compute_variance_2_kahan(data: np.ndarray) -> np.float64:
    """
    Compute variance of given data array using 1-pass algorithm and Kahan's algorithm for
    the summation

    Exceptions: ValueError when data is not 1-dimensional;

    return : variance
    """
    n = len(data)
    minuend = np.square(data[0])
    c_m = 0
    subtrahend = data[0]
    c_s = 0
    for x in data[1:]:
        x_m = np.square(x)
        y_m = x_m - c_m
        t_m = minuend + y_m
        c_m = (t_m - minuend) - y_m
        minuend = t_m

        x_s = x
        y_s = x_s - c_s
        t_s = subtrahend + y_s
        c_s = (t_s - subtrahend) - y_s
        subtrahend = t_s

    return np.divide(minuend - np.divide(np.square(subtrahend), n), n)


def load_temperature_data(filename: str) -> np.ndarray:
    """
    Load temperature data from given file name

    return : data (as double precision np.array)
    """

    data = np.fromfile(filename, dtype=float)

    return data


def compute_variances(data_split: np.ndarray) -> np.ndarray:
    """
    Compute variances with compute_variance_1(), compute_variance_2(), compute_variance_2_Kahan()
    for different latitudes.

    return : variances for different algorithms (rows) and different latitudes (columns)
    """

    vars = np.zeros((3, 5))

    # first: variance1
    for j in range(5):
        vars[0, j] = compute_variance_1(data_split[j])
        vars[1, j] = compute_variance_2(data_split[j])
        vars[2, j] = compute_variance_2_kahan(data_split[j])

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
    expectation = np.divide(sum(np.array([x for x in data])), len(data))
    return expectation


def main():
    data = load_temperature_data("./temperature_locs.dat")
    data_split = np.array_split(data, len(data) / 5)


if __name__ == '__main__':
    main()
