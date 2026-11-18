"Module to score reaction conditions"

import torch
import numpy as np
from tqdm import tqdm

from botorch.models.model import Model
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim.optimize import _split_batch_eval_acqf


def score_reactions(
        gp_model: Model,
        train_x: torch.Tensor,
        x_space: torch.Tensor,
        seed: int,
        method: str,
) -> np.ndarray:
    """
    Score reaction conditions using the provided scoring method.

    Args:
        gp_model (Model): Trained GP model.
        train_x (torch.Tensor): train_x tensor for the iteration
        x_space (torch.Tensor): x feature tensor of the search space
        seed (int): seed to use for MCSamplers if needed
        method (str): Method to use for scoring x_space.

    Returns:
        np.ndarray: The qNEI scores for each point in the search matrix.
    """
    method = method.lower()

    if method == 'qnei':
        scores = calculate_qnei_scores(
            model=gp_model,
            train_x=train_x,
            search_matrix=x_space,
            n_samples=512,
            seed=seed
        )
    elif method == 'ucb':
        scores = calculate_ucb_scores(
            model=gp_model,
            search_matrix=x_space,
            beta=1.0
        )
    elif method == 'qpo':
        scores = calculate_qpo_approx_scores(
            model=gp_model,
            search_matrix=x_space
        )
    else:
        raise ValueError(f"Unknown scoring method: {method}. "
                        f"Available methods: ['qnei', 'ucb','qpo']")

    return scores


def calculate_qnei_scores(
        model: Model,
        train_x: torch.Tensor,
        search_matrix: torch.Tensor,
        n_samples: int,
        seed: int
) -> np.ndarray:
    """
    Calculate the q-Noisy Expected Improvement (qNEI) acquisition function
    scores for the search matrix.

    Args:
        model (Model): A trained Gaussian Process (GP) model.
        train_x (torch.Tensor): The input features of the training data.
        search_matrix (torch.Tensor): The input features of the search space.
        n_samples (int): The number of MC samples to use.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: The qNEI scores for each point in the search matrix.
    """

    acqf = qNoisyExpectedImprovement(
        model=model,
        X_baseline=train_x,
        sampler=SobolQMCNormalSampler(num_samples=n_samples, seed=seed),
        prune_baseline=True
    )

    with torch.no_grad():
        scores = _split_batch_eval_acqf(
            acq_function=acqf, X=search_matrix.unsqueeze(-2), max_batch_size=2048
        )

    return scores.detach().cpu().numpy()


def calculate_ucb_scores(
        model: Model,
        search_matrix: torch.Tensor,
        beta: float = 1.0
) -> np.ndarray:
    """
    Calculate Upper Confidence Bound (UCB) acquisition function scores.
    UCB = mean + beta * std

    Args:
        model (Model): A trained Gaussian Process (GP) model.
        search_matrix (torch.Tensor): The input features of the search space.
        beta (float): Trade-off parameter for UCB.

    Returns:
        np.ndarray: The UCB scores for each point in the search matrix.
    """
    with torch.no_grad():
        posterior = model.posterior(search_matrix)
        mean = posterior.mean
        std = posterior.variance.sqrt()
        scores = mean + beta * std

    return scores.detach().cpu().numpy().reshape(-1)


def calculate_qpo_approx_scores(
        model: Model,
        search_matrix: torch.Tensor,
        M_samples: int = 1000,
        chunk_size: int = 500,
) -> np.ndarray:
    """
    Calculate approximate qPO acquisitiom function scores.
    Approximation to qPO that chunks the search space to handle large
    candidate sets. For each sample, finds the best candidate within each chunk,
    then determines the global winner among chunk winners.
    Args:
        model: Trained Gaussian Process model.
        search_matrix: Input features of candidate points, shape [n_candidates, n_features].
        M_samples: Number of qPO samples to draw. Default: 1000.
        chunk_size: Number of candidates per chunk. Default: 500.
    Returns:
        Array of probability scores for each candidate. Higher scores indicate
        higher probability of being the global optimum.
    """
    model.eval()

    n_candidates = len(search_matrix)
    vote_counts = np.zeros(n_candidates)  # Track votes for each candidate

    chunk_boundaries = list(range(0, n_candidates, chunk_size))
    if chunk_boundaries[-1] != n_candidates:
        chunk_boundaries.append(n_candidates)

    n_chunks = len(chunk_boundaries) - 1
    print(f"Processing {n_candidates} candidates in {n_chunks} chunks of ~{chunk_size}")
    print(f"Running {M_samples} qPO samples...")

    with torch.no_grad():
        # Main loop: Run M independent samples
        for sample_idx in tqdm(range(M_samples)):
            # STEP 1: Process each chunk independently
            chunk_winners = []          # Store actual winner tensors
            chunk_winner_indices = []   # Store global indices of winners

            for chunk_idx in range(n_chunks):
                # Extract current chunk from search space
                start_idx = chunk_boundaries[chunk_idx]
                end_idx = chunk_boundaries[chunk_idx + 1]
                chunk = search_matrix[start_idx:end_idx]  # Shape: [chunk_size, n_features]

                # Sample from this chunk's posterior distribution
                # This gives us one function sample evaluated at all chunk points
                posterior = model.posterior(chunk)
                chunk_sample = posterior.sample()  # Shape: [1, chunk_size] or [chunk_size]

                # Flatten sample if needed (remove singleton dimensions)
                if chunk_sample.dim() > 1:
                    chunk_sample = chunk_sample.squeeze()  # Shape: [chunk_size]

                # Find the winner within this chunk (highest sampled value)
                local_winner_idx = torch.argmax(chunk_sample).item()
                global_winner_idx = start_idx + local_winner_idx

                # Store the winning candidate and its global index
                chunk_winners.append(chunk[local_winner_idx])
                chunk_winner_indices.append(global_winner_idx)

            # STEP 2: Final competition among chunk winners
            if len(chunk_winners) == 1:
                # Special case: only one chunk, so winner is obvious
                global_winner_idx = chunk_winner_indices[0]
            else:
                # Stack all chunk winners into a tensor for joint sampling
                chunk_winners_tensor = torch.stack(chunk_winners)  # Shape: [n_chunks, n_features]
                # Sample from joint posterior of chunk winners
                # This preserves correlations between the chunk winners
                winners_posterior = model.posterior(chunk_winners_tensor)
                winners_sample = winners_posterior.sample()  # Shape: [1, n_chunks] or [n_chunks]

                # Flatten sample if needed
                if winners_sample.dim() > 1:
                    winners_sample = winners_sample.squeeze()  # Shape: [n_chunks]

                # Find the overall winner among chunk winners
                winner_among_winners_idx = torch.argmax(winners_sample).item()
                global_winner_idx = chunk_winner_indices[winner_among_winners_idx]

            # STEP 3: Record vote for this sample's winner
            vote_counts[global_winner_idx] += 1

    # STEP 4: Convert votes to probabilities and select final batch
    probs = vote_counts / M_samples  # Probability each candidate is optimal
    return probs
