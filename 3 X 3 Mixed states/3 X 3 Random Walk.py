import numpy as np
import time
from scipy.linalg import eigvals, expm
import math
import hashlib


def generate_unique_complex_matrix(rows, cols, hash_set):
    """Generate a unique random complex matrix based on a hash set."""
    while True:
        random_matrix = np.random.randn(rows, cols)
        matrix_hash = hashlib.sha256(random_matrix.tobytes()).hexdigest()
        if matrix_hash not in hash_set:
            hash_set.add(matrix_hash)
            return random_matrix


def semi_unitary_matrix(m, n):
    """Generate a semi-unitary matrix using SVD."""
    random_matrix = np.random.randn(m, n)
    u, _, vh = np.linalg.svd(random_matrix, full_matrices=False)
    return (u @ vh).T if m < n else (vh @ u.T).T


def decompose_density_matrix(density_matrix, m, n, semi_unitary):
    """Decompose a density matrix into pure states and probabilities."""
    eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
    eigenvectors = eigenvectors.T
    sqrt_eigenvalues = np.sqrt(np.abs(eigenvalues))

    pure_states, probabilities = [], []
    for j in range(m):
        pure_state = np.zeros_like(eigenvectors[0], dtype=complex)
        probability = 0
        for i in range(len(eigenvalues)):
            pure_state += sqrt_eigenvalues[i] * (semi_unitary[j, i] * eigenvectors[i])
            probability += eigenvalues[i] * (np.conj(semi_unitary[j, i]) * semi_unitary[j, i])
        pure_states.append(pure_state)
        probabilities.append(probability.real)

    # Normalize pure states
    normalized_states = [state / np.sqrt(prob) for state, prob in zip(pure_states, probabilities)]
    psi_psi = [np.outer(state, np.conj(state).T) for state in normalized_states]

    return pure_states, probabilities, psi_psi


def von_entropy(rho):
    """Compute the von Neumann entropy of a reduced density matrix."""
    reduced_rho = np.array([
        [rho[0, 0] + rho[3, 3] + rho[6, 6], rho[0, 1] + rho[3, 4] + rho[6, 7], rho[0, 2] + rho[3, 5] + rho[6, 8]],
        [rho[1, 0] + rho[4, 3] + rho[7, 6], rho[1, 1] + rho[4, 4] + rho[7, 7], rho[1, 2] + rho[4, 5] + rho[7, 8]],
        [rho[2, 0] + rho[5, 3] + rho[8, 6], rho[2, 1] + rho[5, 4] + rho[8, 7], rho[2, 2] + rho[5, 5] + rho[8, 8]]
    ])
    eig_prob = eigvals(reduced_rho).real
    entropy = 0
    for element in eig_prob:
        entropy += shannon_entropy(element)
    return entropy


def shannon_entropy(x):
    """Compute the Shannon entropy of a probability."""
    return -x * math.log2(x) if x > 0 else 0


def hermitian_matrix(m):
    """Generate a random Hermitian matrix."""
    random_matrix = np.random.randn(m, m) + 1j * np.random.randn(m, m)
    return (random_matrix + random_matrix.conj().T) / 2


def acim(matrix, alpha):
    """Apply the ACIM transformation to a matrix."""
    return expm(1j * alpha * matrix)


def entanglement(probabilities, states):
    """Compute the entanglement entropy."""
    return sum(prob * von_entropy(state) for prob, state in zip(probabilities, states))


def auto_entanglement(E, semi_unitary, m, alpha, alpha_end, steps, check):
    """Optimize entanglement by adjusting the semi-unitary matrix."""
    count = 0
    n=0
    min_sum = []
    while alpha > alpha_end:
        step_results, step_matrices = [], []
        original_semi_unitary = semi_unitary
        for _ in range(steps):
            semi_unitary = semi_unitary @ acim(hermitian_matrix(9), alpha)
            step_matrices.append(semi_unitary)
            pure_states, probabilities, psi_psi = decompose_density_matrix(rho, m, 9, semi_unitary)
            E_new = entanglement(probabilities, psi_psi)
            step_results.append(E_new)

        E_new = min(step_results)
        semi_unitary = step_matrices[step_results.index(E_new)]
        if E_new < E:
            E = E_new
            n=0
            min_sum.append(semi_unitary)
        elif n == check:
            n = 0
            alpha = decrement * alpha
        else:
            semi_unitary = original_semi_unitary
            n += 1

    print(E, f"Try: {count}" )
    return E


# Main execution
start = time.time()
rho = np.array([[0.15450849719793816, -9.107298248878237e-18, -0.08589018742370494, -8.673617379884035e-18, 0.0, 0.0, -0.08589018742370495, 2.42861286636753e-17, -1.214306433183765e-17], [-9.107298248878237e-18, 0.15450849719793816, -2.2551405187698492e-17, 3.469446951953614e-18, 1.1275702593849246e-17, -0.08589018742370495, 2.2551405187698492e-17, 0.08589018742370495, -8.673617379884035e-18], [-0.08589018742370494, -2.2551405187698492e-17, 0.09549150282050275, 0.0, -0.08589018742370495, -1.0408340855860843e-17, -3.122502256758253e-17, -5.204170427930421e-18, 0.0], [-8.673617379884035e-18, 3.469446951953614e-18, 0.0, 0.15450849719793816, 0.0, 0.08589018742370495, 2.7755575615628914e-17, 0.08589018742370494, -6.938893903907228e-18], [0.0, 1.1275702593849246e-17, -0.08589018742370495, 0.0, 0.15450849719793813, 2.0816681711721685e-17, 0.08589018742370495, -3.8163916471489756e-17, -6.938893903907228e-18], [0.0, -0.08589018742370495, -1.0408340855860843e-17, 0.08589018742370495, 2.0816681711721685e-17, 0.09549150282050273, -1.3877787807814457e-17, 0.0, -6.938893903907228e-18], [-0.08589018742370495, 2.2551405187698492e-17, -3.122502256758253e-17, 2.7755575615628914e-17, 0.08589018742370495, -1.3877787807814457e-17, 0.09549150282050273, -6.938893903907228e-18, 0.0], [2.42861286636753e-17, 0.08589018742370495, -5.204170427930421e-18, 0.08589018742370494, -3.8163916471489756e-17, 0.0, -6.938893903907228e-18, 0.09549150282050273, -6.938893903907228e-18], [-1.214306433183765e-17, -8.673617379884035e-18, 0.0, -6.938893903907228e-18, -6.938893903907228e-18, -6.938893903907228e-18, 0.0, -6.938893903907228e-18, -1.5839551892327108e-12]])


# Parameters
alpha_start = 0.3
alpha_end = 0.00001
m = 9
entanglement_results = []
decrement = 0.70

for _ in range(10):
    semi_unitary = semi_unitary_matrix(m, 9)
    pure_states, probabilities, psi_psi = decompose_density_matrix(rho, m, 9, semi_unitary)
    E = entanglement(probabilities, psi_psi)
    result = auto_entanglement(E, semi_unitary, m, alpha_start, alpha_end, 100, 30)
    entanglement_results.append(result)
    with open("3_X_3_bipartite_result_3.txt", "a+") as file:
        file.write(f"Result of iteration {_}: {result}\n")

print(f"Minimum entanglement: {min(entanglement_results)}")
with open("3_X_3_bipartite_result_3.txt", "a+") as file:
    file.write(f"Minimum entanglement: {min(entanglement_results)}\n")

print(f"Execution time: {time.time() - start} seconds")
