import numpy as np
from julia.api import Julia

jl = Julia(compiled_modules=False)

from julia import Main

label_smoothening_path = "labeling/label_smoothening"

Main.eval(f'using Pkg')
Main.eval(f'Pkg.activate("{label_smoothening_path}", io=devnull)')


def smoothen_labels(labels: list[float], tau=1.0, beta=2.5, kappa=0.0):
    """Expected to be called from `src/learning/bayesian_regression`.

    Args:
        labels (list[float]): _description_
        tau (float): _description_
        beta (float): _description_
        kappa (float): _description_


    Returns:
        _type_: _description_
    """
    if labels[0] > 200 or np.allclose(np.array(labels), labels[0]):
        return labels
    if not isinstance(labels[0], float):
        labels = [float(label) for label in labels]

    Main.eval(f'include("{label_smoothening_path}/labels_factor_graph.jl")')
    result = Main.eval(f"smoothen_labels_measure_runtime({labels}, {tau}, {beta}, {kappa})")
    return result


def smoothen_labels_greater_than(labels: list[float], beta=2.5, epsilon: float | int = 0):
    """Expected to be called from `src/learning/bayesian_regression`.

    Args:
        labels (list[float]): _description_
        tau (float): _description_
        beta (float): _description_
        kappa (float): _description_


    Returns:
        _type_: _description_
    """

    Main.eval(f'include("{label_smoothening_path}/labels_factor_graph_greater_than.jl")')
    result = Main.eval(f"smoothen_labels({labels}, {beta}, {epsilon})")
    return result
