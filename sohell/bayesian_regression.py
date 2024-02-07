import logging

import numpy as np
from tqdm import tqdm

from labeling.cycle_finder import CycleData
from typing import Callable

from labeling.utils import load_parameters

logger = logging.getLogger(__name__)


def get_target_values(cycles: list[CycleData], labels_dir: str) -> np.ndarray:
    labels = np.empty(len(cycles))
    for i, cycle in enumerate(cycles):
        parameters = load_parameters(labels_dir, cycle)
        average_sohc = parameters.get_average_sohc()
        labels[i] = average_sohc
    return labels


def get_basis_function_values(cycles: list[CycleData], basis_function: Callable[[CycleData], float]) -> np.ndarray:
    function_values = np.empty((len(cycles), 1))
    for i, cycle in tqdm(enumerate(cycles), total=len(cycles), desc="Extracting design matrix column for cycles"):
        function_value = basis_function(cycle)
        function_values[i] = function_value
    return function_values


def posterior(Phi, t, alpha, beta, return_inverse=False):
    """
    Compute mean and covariance matrix of the posterior distribution.
    Assume a zero-mean isotropic Gaussian prior over the weights,
    governed merely by alpha^(-1) * I as the standard deviation matrix (See Bishop, eq. (3.52)).
    """
    S_N_inv = alpha * np.eye(Phi.shape[1]) + beta * Phi.T.dot(Phi)  # Bishop, eq. (3.54)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N.dot(Phi.T).dot(t)  # Bishop, eq. (3.53)

    if return_inverse:
        return m_N, S_N, S_N_inv
    else:
        return m_N, S_N


def posterior_predictive(Phi_test, m_N, S_N, beta):
    y = Phi_test.dot(m_N)  # Bishop, eq. (3.58), mean of the normal distribution over t
    # Only compute variances (diagonal elements of covariance matrix)
    y_var = 1 / beta + np.sum(Phi_test.dot(S_N) * Phi_test, axis=1)  # Bishop, eq. (3.59)

    return y.ravel(), y_var


def log_marginal_likelihood(Phi, t, alpha, beta):
    N, M = Phi.shape

    m_N, _, S_N_inv = posterior(Phi, t, alpha, beta, return_inverse=True)

    E_D = beta * np.sum((t - Phi.dot(m_N)) ** 2)  # Bishop, eq. (3.82), left summand
    E_W = alpha * np.sum(m_N**2)  # Bishop, eq. (3.82), right summand

    # Bishop, eq. (3.86)
    score = M * np.log(alpha) + N * np.log(beta) - E_D - E_W - np.log(np.linalg.det(S_N_inv)) - N * np.log(2 * np.pi)

    return 0.5 * score


# Bishop, p. 168, 169
def fit(Phi, t, alpha_0=1e-5, beta_0=1e-5, max_iter=200, rtol=1e-5, verbose=True):

    N, M = Phi.shape

    eigenvalues_0 = np.linalg.eigvalsh(Phi.T.dot(Phi))

    beta = beta_0
    alpha = alpha_0

    for i in range(max_iter):
        beta_prev = beta
        alpha_prev = alpha

        eigenvalues = eigenvalues_0 * beta  # Bishop, eq. (3.87)

        m_N, S_N = posterior(Phi, t, alpha, beta)

        gamma = np.sum(eigenvalues / (eigenvalues + alpha))  # Bishop, eq. (3.91)
        alpha = gamma / np.sum(m_N**2)  # Bishop, eq. (3.92)

        beta_inv = 1 / (N - gamma) * np.sum((t - Phi.dot(m_N)) ** 2)  # Bishop, eq. (3.95)
        beta = 1 / beta_inv

        if np.isclose(alpha_prev, alpha, rtol=rtol) and np.isclose(beta_prev, beta, rtol=rtol):
            if verbose:
                logger.info(f"Convergence after {i + 1} iterations.")
            return alpha, beta, m_N, S_N

    if verbose:
        logger.info(f"Stopped after {max_iter} iterations.")
    return alpha, beta, m_N, S_N


def expand(X: list[CycleData], basis_functions: list[Callable[[CycleData], float]]):
    Phi = np.zeros((len(X), len(basis_functions) + 1))
    # The zero-th basis function is always phi_0(x) = 1
    Phi[:, 0] = np.ones((len(X),))
    for j, basis_function in enumerate(basis_functions):
        Phi[:, j + 1] = get_basis_function_values(X, basis_function).ravel()
    return Phi
