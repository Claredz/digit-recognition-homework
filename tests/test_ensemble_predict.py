import torch

from src.ensemble_predict import average_probabilities, search_specialist_generalist_weight, weighted_probability_fusion


def test_average_probabilities_preserves_shape_and_normalization():
    first = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
    second = torch.tensor([[0.6, 0.4], [0.5, 0.5]])

    averaged = average_probabilities([first, second])

    assert averaged.shape == (2, 2)
    assert torch.allclose(averaged.sum(dim=1), torch.ones(2))


def test_weighted_probability_fusion_shape_and_extremes():
    specialist = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    generalist = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    fused = weighted_probability_fusion(specialist, generalist, 0.7)

    assert fused.shape == specialist.shape
    assert torch.allclose(fused.sum(dim=1), torch.ones(2))
    assert torch.allclose(weighted_probability_fusion(specialist, generalist, 1.0), specialist)
    assert torch.allclose(weighted_probability_fusion(specialist, generalist, 0.0), generalist)


def test_specialist_generalist_weight_search_returns_best_weight():
    labels = torch.tensor([0, 1])
    specialist = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
    generalist = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    result = search_specialist_generalist_weight(specialist, generalist, labels, weights=[0.0, 0.5, 1.0])

    assert result["best_weight"] == 1.0
    assert result["best_accuracy"] == 1.0
