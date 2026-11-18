"Test misc functions in acqf.py"

import pytest
import torch
import numpy as np


@pytest.fixture
def sample_data():
    """Create sample search matrix and parameters"""
    search_matrix = torch.tensor([
        [1.0, 50.0, 3.0],
        [2.0, 80.0, 4.0], 
        [3.0, 50.0, 5.0],
        [4.0, 80.0, 6.0]
    ])

    feature_index = 1  # temperature column
    top_temperature = 80.0
    top_temperatures = [80.0]

    return search_matrix, feature_index, top_temperature, top_temperatures


def test_temperature_filtering_equivalence(sample_data):
    """Test that both filtering methods give same result"""
    search_matrix, feature_index, top_temperature, top_temperatures = sample_data

    # Method 1: List comprehension + stack
    result1 = torch.stack([
        experiment 
        for experiment in search_matrix 
        if experiment[feature_index] in top_temperatures
    ])

    # Method 2: Boolean indexing
    result2 = search_matrix[search_matrix[:, feature_index] == top_temperature]

    # Test equality
    assert torch.equal(result1, result2)

    # Test content
    expected_result = torch.tensor([
        [2.0, 80.0, 4.0],
        [4.0, 80.0, 6.0]
    ])
    assert torch.equal(result1, expected_result)
    assert torch.equal(result2, expected_result)


def test_no_matching_temperatures(sample_data):
    """Test behavior when no temperatures match"""
    search_matrix, feature_index, _, _ = sample_data

    top_temperature = 100.0
    top_temperatures = [100.0]

    # Method 1
    matching_experiments = [
        experiment 
        for experiment in search_matrix 
        if experiment[feature_index] in top_temperatures
    ]

    if matching_experiments:  # Only stack if there are matches
        result1 = torch.stack(matching_experiments)
    else:
        result1 = torch.tensor([])
 
    # Method 2
    result2 = search_matrix[search_matrix[:, feature_index] == top_temperature]

    assert result1.numel() == 0
    assert result2.numel() == 0


def test_multiple_temperatures():
    """Test behavior when multiple temperatures are provided (should be different)"""
    search_matrix = torch.tensor([
        [1.0, 50.0, 3.0],
        [2.0, 80.0, 4.0],
        [3.0, 50.0, 5.0],
        [4.0, 80.0, 6.0]
    ])

    feature_index = 1
    top_temperatures = [50.0, 80.0]
    top_temperature = 80.0

    # Method 1 should return all rows
    result1 = torch.stack([
        experiment 
        for experiment in search_matrix 
        if experiment[feature_index] in top_temperatures
    ])

    # Method 2 should return only 80.0 rows
    result2 = search_matrix[search_matrix[:, feature_index] == top_temperature]

    assert not torch.equal(result1, result2)
    assert len(result1) == 4  # All rows
    assert len(result2) == 2  # Only 80.0 rows
