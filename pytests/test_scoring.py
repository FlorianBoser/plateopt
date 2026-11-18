import pytest
import torch
import numpy as np
from unittest.mock import MagicMock

from plateopt.scoring import calculate_qpo_approx_scores

def test_qpo_approx_scores_logic():
    """
    Tests the core logic of calculate_qpo_approx_scores.

    This test uses a mock model to control the outcome of posterior sampling,
    ensuring that the chunking, winner-selection, and voting mechanism
    works as designed.
    """
    # 1. ARRANGE: Set up the test scenario
    # ----------------------------------------------------
    n_candidates = 10
    chunk_size = 3  # Will create chunks of size [3, 3, 3, 1]
    M_samples = 4   # We will run 4 simulated "tournaments"

    # A simple search matrix with 10 candidates
    search_matrix = torch.arange(n_candidates, dtype=torch.float).unsqueeze(-1)

    # Define the "script" for our mock model's samples to follow.
    # We will force specific candidates to win each stage.
    # --- The regional winners will always be candidates #1, #4, #8, #9 ---
    chunk_1_sample = torch.tensor([1.0, 5.0, 2.0])  # Winner is local index 1 (global 1)
    chunk_2_sample = torch.tensor([3.0, 4.0, 1.0])  # Winner is local index 1 (global 4)
    chunk_3_sample = torch.tensor([2.0, 3.0, 5.0])  # Winner is local index 2 (global 8)
    chunk_4_sample = torch.tensor([10.0])          # Winner is local index 0 (global 9)

    # --- In the "finals", we will make different candidates win each time ---
    final_round_sample_1 = torch.tensor([1., 2., 5., 3.]) # Winner is the 3rd one in the list (candidate #8)
    final_round_sample_2 = torch.tensor([2., 1., 5., 4.]) # Winner is the 3rd one again (candidate #8)
    final_round_sample_3 = torch.tensor([3., 2., 1., 5.]) # Winner is the 4th one (candidate #9)
    final_round_sample_4 = torch.tensor([5., 1., 2., 3.]) # Winner is the 1st one (candidate #1)

    # This is the full sequence of return values our mock sampler will produce.
    # It cycles through all chunks, then a final, for each of the 4 M_samples.
    mock_sample_sequence = [
        # M_sample 1
        chunk_1_sample, chunk_2_sample, chunk_3_sample, chunk_4_sample, final_round_sample_1,
        # M_sample 2
        chunk_1_sample, chunk_2_sample, chunk_3_sample, chunk_4_sample, final_round_sample_2,
        # M_sample 3
        chunk_1_sample, chunk_2_sample, chunk_3_sample, chunk_4_sample, final_round_sample_3,
        # M_sample 4
        chunk_1_sample, chunk_2_sample, chunk_3_sample, chunk_4_sample, final_round_sample_4,
    ]

    # Create the mock model
    mock_model = MagicMock()
    mock_posterior = MagicMock()
    # Configure the mock to return values from our sequence on each call to .sample()
    mock_posterior.sample.side_effect = mock_sample_sequence
    mock_model.posterior.return_value = mock_posterior

    # 2. ACT: Run the function with the mock model and data
    # ----------------------------------------------------
    probs = calculate_qpo_approx_scores(
        model=mock_model,
        search_matrix=search_matrix,
        M_samples=M_samples,
        chunk_size=chunk_size,
    )

    # 3. ASSERT: Check if the outcome is as expected
    # ----------------------------------------------------
    # Based on our "script", the votes should be:
    # - Candidate #8 won 2 times
    # - Candidate #9 won 1 time
    # - Candidate #1 won 1 time
    # Total votes = 4 (which is M_samples)
    expected_probs = np.zeros(n_candidates)
    expected_probs[8] = 2 / M_samples  # 0.5
    expected_probs[9] = 1 / M_samples  # 0.25
    expected_probs[1] = 1 / M_samples  # 0.25

    assert isinstance(probs, np.ndarray)
    assert probs.shape == (n_candidates,)
    # Use numpy's testing utility for robust array comparison
    np.testing.assert_array_equal(probs, expected_probs)
    # Sanity check: probabilities should sum to 1
    assert np.isclose(np.sum(probs), 1.0)