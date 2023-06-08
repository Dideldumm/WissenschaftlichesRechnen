import numpy as np
import matplotlib.pyplot as plt
import datetime

import unittest
import tomograph
from main import compute_tomograph, gaussian_elimination, back_substitution, compute_cholesky


class Tests(unittest.TestCase):
    def test_gaussian_elimination(self):
        A = np.random.randn(4, 4)
        x = np.random.rand(4)
        b = np.dot(A, x)
        A_elim, b_elim = gaussian_elimination(A, b)
        self.assertTrue(np.allclose(np.linalg.solve(A_elim, b_elim), x))  # Check if system is still solvable
        self.assertTrue(np.allclose(A_elim, np.triu(A_elim)))  # Check if matrix is upper triangular

    def test_back_substitution(self):
        A = np.random.randn(4, 4)
        b = np.random.rand(4)
        A_elim, b_elim = gaussian_elimination(A, b)
        x = back_substitution(A_elim, b_elim)
        self.assertTrue(np.allclose(np.linalg.solve(A_elim, b_elim), x))  # Check if system is solved correctly
        self.assertTrue(np.allclose(A_elim, np.triu(A_elim)))  # Check if matrix is upper triangular

    def test_cholesky_decomposition(self):
        A = np.random.randn(4, 4)
        try:
            L = compute_cholesky(A)
        except ValueError:
            L = np.zeros(1)

        try:
            lsg = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            lsg = np.zeros(1)
        self.assertTrue(np.allclose(lsg, L))  # Check if L is solved correctly

    def test_solve_cholesky(self):
        pass
        # TODO

    def test_compute_tomograph(self):
        t = datetime.datetime.now()
        print("Start time: " + str(t.hour) + ":" + str(t.minute) + ":" + str(t.second))

        # Compute tomographic image
        n_shots = 64  # 128
        n_rays = 64  # 128
        n_grid = 32  # 64
        tim = compute_tomograph(n_shots, n_rays, n_grid)

        t = datetime.datetime.now()
        print("End time: " + str(t.hour) + ":" + str(t.minute) + ":" + str(t.second))

        # Visualize image
        plt.imshow(tim, cmap='gist_yarg', extent=[-1.0, 1.0, -1.0, 1.0],
                   origin='lower', interpolation='nearest')
        plt.gca().set_xticks([-1, 0, 1])
        plt.gca().set_yticks([-1, 0, 1])
        plt.gca().set_title('%dx%d' % (n_grid, n_grid))

        plt.show()


if __name__ == '__main__':
    unittest.main()
