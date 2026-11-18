"Wrapper for different acquisition functions"

import gc
from typing import List, Dict

from tqdm import tqdm
import torch
import numpy as np
from botorch.models.model import Model
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.samplers import SobolQMCNormalSampler

from plateopt.utils import get_index_from_lookup
from plateopt.scoring import calculate_qnei_scores

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


def optimise_qnei_and_get_suggestions(
    model: Model,
    train_x: torch.Tensor,
    search_matrix: torch.Tensor,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    """
    Basic implementation of qNEI. Used for baseline comparison
    against templated qNEI.
    """

    # empty tensor to store new candidates
    X_next = torch.empty((0, search_matrix.size(1)), **tkwargs)
    evaluation_matrix = search_matrix.detach().clone()

    sampler = SobolQMCNormalSampler(num_samples=512, seed=seed)

    for _ in tqdm(range(batch_size), desc="Optimizing batch"):
        acq_func = qNoisyExpectedImprovement(
            model=model,
            X_baseline=train_x,
            sampler=sampler,
            prune_baseline=True,
            X_pending=X_next,
        )

        # Evaluate acquisition function over the discrete search matrix
        candidate, _ = optimize_acqf_discrete(
            acq_function=acq_func, choices=evaluation_matrix, q=1
        )

        X_next = torch.cat((X_next, candidate), dim=0)

        # get index of best candidate from the current evaluation matrix
        best_index = get_index_from_lookup(candidate, evaluation_matrix)
        best_index = best_index[0]

        # remove best candidate from the current evaluation matrix
        evaluation_matrix = torch.cat(
            [evaluation_matrix[:best_index], evaluation_matrix[best_index + 1 :]]
        )

        # Explicit memory cleanup after generation of each batch point
        gc.collect()
        torch.cuda.empty_cache()

    return X_next


def select_top_temperature(
    model: Model,
    train_x: torch.Tensor,
    search_matrix: torch.Tensor,
    seed: int,
    feature_index: int,
):
    "Uses average qnei value to select the top temperature for plate"
    qnei_scores = calculate_qnei_scores(
        model=model,
        train_x=train_x,
        search_matrix=search_matrix,
        n_samples=512,
        seed=seed,
    )  # already returned as np

    temp_values = (
        search_matrix[:, feature_index].detach().cpu().numpy()
    )  # extract temperatures from search matrix

    top_temperature = get_top_temperatures(
        temp_values=temp_values,
        acqf_values=qnei_scores,
        n_temperatures=1,
        aggregation_strategy="mean",
    )[0]  # take only the first value

    # Filter the search matrix to include only the rows with the top temperature
    promising_temperature_experiments = search_matrix[search_matrix[:, feature_index] == top_temperature]

    return promising_temperature_experiments.to(train_x.device)


def nested_qnei(
    model: Model,
    train_x: torch.Tensor,
    search_matrix: torch.Tensor,
    batch_size: int,
    seed: int,
    feature_index: int,
):
    "Runs nested qnei for plates where temperature is restricted"
    filtered_search_matrix = select_top_temperature(
        model=model,
        train_x=train_x,
        search_matrix=search_matrix,
        seed=seed,
        feature_index=feature_index,
    )

    new_x = optimise_qnei_and_get_suggestions(
        model=model,
        train_x=train_x,
        search_matrix=filtered_search_matrix,
        batch_size=batch_size,
        seed=seed,
    )

    return new_x


def get_top_temperatures(
    temp_values: np.array,
    acqf_values: np.array,
    n_temperatures: int,
    aggregation_strategy: str,
) -> List:
    """
    From computed acquisition function values for all points in the search matrix,
    aggregate into dictionaries and obtain the top_n_temperatures based on the
    acqusition function values or their experiments.

    Note:
        This assumes that maximising acquisition function value is the desired goal.

    Args:
        temp_values (np.array): List of temperatures for all reactions in search space
        acqf_values (np.array): List of acquisition function values for all reactions
                                in the search space
        n_temperatures (int): Number of unique temperatures to return
        aggregation_strategy (str): Whether to return temperatures sorted by max, mean, or min values.

    Output:
        List: List of top temperatures based on aggregation strategy.
    """

    temp_dict = {}
    for temp, hvi_value in zip(temp_values, acqf_values):
        if temp not in temp_dict:
            temp_dict[temp] = {"values": []}
        temp_dict[temp]["values"].append(hvi_value)

    aggregated_dict = aggregate_acqf_values(temp_dict, aggregation_strategy)
    sorted_temperatures = sorted(
        aggregated_dict.items(), key=lambda x: x[1], reverse=True
    )

    # get the top performing temperatures
    top_temperatures = [temp for temp, _ in sorted_temperatures[:n_temperatures]]

    return top_temperatures


def aggregate_acqf_values(temp_dict: Dict, aggregation_strategy: str = "mean") -> Dict:
    """
    Aggregates the acquisition function values for each temperature based on the specified strategy.

    Args:
        temp_dict (dict): Dictionary containing temperature-wise acquisition function values.
        strategy (str): Strategy to use for aggregation. Options: 'mean', 'max', 'min'.

    Returns:
        Dict: Dictionary containing the aggregated acquisition function value for each temperature.
    """
    if aggregation_strategy not in ["mean", "max", "min"]:
        raise ValueError(f"Invalid aggregation strategy: {aggregation_strategy}")

    aggregated_dict = {}

    for temp in temp_dict:
        values = temp_dict[temp]["values"]

        if aggregation_strategy == "mean":
            aggregated_value = sum(values) / len(values)

        elif aggregation_strategy == "max":
            aggregated_value = max(values)

        elif aggregation_strategy == "min":
            aggregated_value = min(values)

        aggregated_dict[temp] = aggregated_value

    return aggregated_dict
