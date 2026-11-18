"Utils module for helper function with tensor operations."

from typing import Tuple, List, Dict

import torch
from torch.quasirandom import SobolEngine


def get_index_from_lookup(X: torch.Tensor, X_lookup: torch.Tensor) -> List:
    """
    Finds the index (or indices) of elements in `X` within `X_lookup`.

    Args:
        X (torch.Tensor): A tensor containing elements to look up.
        X_lookup (torch.Tensor): A tensor serving as the lookup source.

    Returns:
        List: Indices of where X is found in X_lookup.

    This function facilitates finding the position(s) of tensor(s) in a lookup tensor, useful for indexing or cross-referencing operations.
    """
    # Creates a lookup dictionary to match tensors in X_lookup to their indices
    lookup_dict = create_lookup_dict(X_lookup)

    if X.ndim == 1:
        X = X.unsqueeze(0)

    index_list = []
    for x in X:
        key = tensor_to_tuple(x)
        if key in lookup_dict:
            index_list.append(lookup_dict[key])

    return index_list


def unified_objective_function(
    X: torch.Tensor,
    X_lookup: torch.Tensor,
    Y_lookup: torch.Tensor,
) -> torch.Tensor:
    """
    Maps each tensor in X to a given index in X_lookup, and returns the value of Y_lookup[index].

    Args:
        X (torch.Tensor): A tensor containing the items for which corresponding Y_lookup values are desired
        X_lookup (torch.Tensor): A tensor where each row represents a unique key for lookup.
                                 Each row in X should have a corresponding row in X_lookup.
        Y_lookup (torch.Tensor): A tensor where each row contains values corresponding to the keys in X_lookup.
                                 The index of each row in Y_lookup should match the index of the corresponding row in X_lookup.

    Returns:
        torch.Tensor: A tensor containing the Y_lookup values corresponding to each tensor in X, in the order they
                      are found in X. If a tensor in X does not have a corresponding value (i.e., it is not found in
                      X_lookup), it is omitted from the result.
    """
    # get indices of X in X_lookup, this is a list
    indices = get_index_from_lookup(X, X_lookup)

    # get the corresponding Y values
    results = torch.stack([Y_lookup[idx] for idx in indices])

    return results


def tensor_to_tuple(t: torch.Tensor) -> Tuple:
    """
    Convert a tensor to a tuple for use as a dictionary key.
    This is required as dictionary keys are required to be immutable.
    However, tensor values can be updated so should be converted.

    Args:
        t (torch.Tensor): A PyTorch tensor.

    Returns:
        tuple: A tuple representation of the input tensor.
    """
    return tuple(t.tolist())


def create_lookup_dict(X_lookup: torch.Tensor) -> Dict:
    """
    Create a lookup dictionary from an iterable of tensors to their indices.

    Args:
        X_lookup [torch.Tensor]: An iterable tensor where each row is used as lookup values.

    Returns:
        dict: A dictionary mapping tensor tuples to their respective indices.
    """
    return {tensor_to_tuple(x): i for i, x in enumerate(X_lookup)}


def draw_sobol_samples(n_samples: int, feature_matrix: torch.Tensor) -> torch.Tensor:
    """
    Draw Sobol samples from a pre-defined, normalized (0,1) feature space and finds the closest points in the feature matrix.

    The feature matrix is pre-normalized to ensure same scale as Sobol.

    Args:
        n_samples (int): Number of Sobol samples to draw.
        feature_matrix (torch.Tensor): A tensor representing the feature space, normalized to (0,1), where each row
                                       is a feature vector.

    Returns:
        torch.Tensor: The closest points in the feature matrix corresponding to each Sobol sample.

    Note:
        This function's reproducibility is influenced by the global random seed setting and scrambling of Sobol sequences.
    """

    num_features = feature_matrix.size(1)
    sobol_sampler = SobolEngine(dimension=num_features, scramble=True)

    # Generate Sobol samples in unit hypercube
    sobol_samples = sobol_sampler.draw(n_samples).to(feature_matrix)

    # Find closest points in the tensor for each Sobol sample
    selected_indices = torch.cdist(sobol_samples, feature_matrix).argmin(dim=1)
    selected_configurations = feature_matrix[selected_indices]

    return selected_configurations
