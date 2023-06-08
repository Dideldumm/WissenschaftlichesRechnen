import numpy as np
import tomograph


####################################################################################################
# Exercise 1: Gaussian elimination

def gaussian_elimination(A: np.ndarray, b: np.ndarray, use_pivoting: bool = True) -> (np.ndarray, np.ndarray):
    """
    Gaussian Elimination of Ax=b with or without pivoting.

    Arguments:
    A : matrix, representing left side of equation system of size: (m,m)
    b : vector, representing right hand side of size: (m, )
    use_pivoting : flag if pivoting should be used

    Return:
    A : reduced result matrix in row echelon form (type: np.ndarray, size: (m,m))
    b : result vector in row echelon form (type: np.ndarray, size: (m, ))

    Raised Exceptions:
    ValueError: if matrix and vector sizes are incompatible, matrix is not square or pivoting is disabled but necessary

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """
    # Create copies of input matrix and vector to leave them unmodified
    A: np.ndarray = A.copy()
    b: np.ndarray = b.copy()

    # Test if shape of matrix and vector is compatible and raise ValueError if not
    if not __is_matrix_compatible_to_vector(A, b):
        raise ValueError

    m, _ = A.shape

    # Perform gaussian elimination
    for i in range(m):
        if use_pivoting:
            max_row = np.argmax(abs(A[i:, i])) + i
            A[[i, max_row], :] = A[[max_row, i], :]
            b[[i, max_row]] = b[[max_row, i]]
            # https://stackoverflow.com/questions/54069863/swap-two-rows-in-a-numpy-array-in-python

        for j in range(i + 1, m):
            if A[i, i] == 0:
                raise ValueError
            f = A[j, i] / A[i, i]
            A[j, i:] -= f * A[i, i:]
            b[j] -= f * b[i]

    return A, b


def back_substitution(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Back substitution for the solution of a linear system in row echelon form.

    Arguments:
    A : matrix in row echelon representing linear system
    b : vector, representing right hand side

    Return:
    x : solution of the linear system

    Raised Exceptions:
    ValueError: if matrix/vector sizes are incompatible or no/infinite solutions exist

    Side Effects:
    -

    Forbidden:
    - numpy.linalg.*
    """

    # Test if shape of matrix and vector is compatible and raise ValueError if not
    if not __is_matrix_compatible_to_vector(A, b):
        raise ValueError

    if not np.allclose(A, np.triu(A)):
        raise ValueError

    # Initialize solution vector with proper size
    m, _ = A.shape
    x = np.zeros(m)

    # Run backsubstitution and fill solution vector, raise ValueError if no/infinite solutions exist
    for i in range(m - 1, -1, -1):
        if A[i, i] == 0:
            raise ValueError
        x[i] = (b[i] - np.dot(A[i, (i + 1):], x[(i + 1):])) / A[i, i]

    return x


def __is_matrix_compatible_to_vector(A: np.ndarray, b: np.ndarray) -> bool:
    m, n = A.shape
    l, = b.shape
    return m == n == l


####################################################################################################
# Exercise 2: Cholesky decomposition

def compute_cholesky(M: np.ndarray) -> np.ndarray:
    """
    Compute Cholesky decomposition of a matrix

    Arguments:
    M : matrix, symmetric and positive (semi-)definite

    Raised Exceptions:
    ValueError: L is not symmetric and psd

    Return:
    L :  Cholesky factor of M

    Forbidden:
    - numpy.linalg.*
    """

    # check for symmetry and raise an exception of type ValueError
    (n, m) = M.shape
    if not n == m:
        raise ValueError

    # build the factorization and raise a ValueError in case of a non-positive definite input matrix

    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                tmp = M[i, i] - np.sum(L[i, :j] ** 2)
                if tmp < 0:
                    raise ValueError
                L[i, j] = np.sqrt(tmp)
            else:
                L[i, j] = (M[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]

    return L


def solve_cholesky(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the system L L^T x = b where L is a lower triangular matrix

    Arguments:
    L : matrix representing the Cholesky factor
    b : right hand side of the linear system

    Raised Exceptions:
    ValueError: sizes of L, b do not match
    ValueError: L is not lower triangular matrix

    Return:
    x : solution of the linear system

    Forbidden:
    - numpy.linalg.*
    """

    # Check the input for validity, raising a ValueError if this is not the case
    (n, m) = L.shape
    if not n == m:
        raise ValueError

    if not np.allclose(L, np.tril(L)):
        raise ValueError

    # Solve the system by forward- and backsubstitution
    LT = L.transpose()
    y = back_substitution(LT, b)
    x = np.zeros(m)
    for i in range(m):
        if L[i, i] == 0:
            raise ValueError
        x[i] = (y[i] - np.dot(L[i, :i], x[:i])) / L[i, i]

    return x


####################################################################################################
# Exercise 3: Tomography

def setup_system_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> (np.ndarray, np.ndarray):
    """
    Set up the linear system describing the tomographic reconstruction

    Arguments:
    n_shots  : number of different directions
    n_rays   : number of parallel rays
    n_grid   : number of cells of grid in each direction

    Return:
    A : system matrix
    v : measured intensities

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    -
    """

    # TODO: Initialize system matrix with proper size
    A = np.zeros((1, 1))
    # TODO: Initialize sinogram for measurements for each shot in one row
    S = np.zeros((1, 1))

    # TODO: Iterate over equispaced angles, take measurements, and update system matrix and sinogram
    theta = 0
    # Take a measurement from direction theta. Return values are
    # ints : measured intensities for parallel rays (ndarray)
    # idx  : indices of rays (ndarray)
    # idx_isect : indices of intersected cells (ndarray)
    # dt : lengths of segments in intersected cells (ndarray)
    ints, idxs, idxs_isects, dt = tomograph.take_measurement(n_grid, n_rays, theta)

    # TODO: Convert per shot measurements in sinogram to a 1D np.array so that columns
    # in the sinogram become consecutive elements in the array
    v = np.zeros(n_shots * n_rays)

    return [A, v]


def compute_tomograph(n_shots: np.int64, n_rays: np.int64, n_grid: np.int64) -> np.ndarray:
    """
    Compute tomographic image

    Arguments:
    n_shots  : number of shots from different directions
    n_rays   : number of parallel rays
    r_theta  : number of cells in the grid in each direction

    Return:
    tim : tomographic image

    Raised Exceptions:
    -

    Side Effects:
    -

    Forbidden:
    """

    # Setup the system describing the image reconstruction
    [L, v] = setup_system_tomograph(n_shots, n_rays, n_grid)

    # TODO: Solve for tomographic image using your Cholesky solver
    # (alternatively use Numpy's Cholesky implementation)

    # TODO: Convert solution of linear system to 2D image
    tim = np.zeros((n_grid, n_grid))

    return tim


if __name__ == '__main__':
    # Compute tomographic image
    n_shots = 32  # 128
    n_rays = 32  # 128
    n_grid = 16  # 64
    tim = compute_tomograph(n_shots, n_rays, n_grid)

    print("All requested functions for the assignment have to be implemented in this file and uploaded to the "
          "server for the grading.\nTo test your implemented functions you can "
          "implement/run tests in the file tests.py (> python3 -v test.py [Tests.<test_function>]).")
