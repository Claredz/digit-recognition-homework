import pytest
import torch

from src.model import MediumCNN, SmallCNN, build_model, normalize_model_name


def test_small_cnn_returns_logits_for_ten_classes():
    model = SmallCNN(num_classes=10, in_channels=1)
    batch = torch.randn(4, 1, 28, 28)

    logits = model(batch)

    assert logits.shape == (4, 10)


def test_medium_cnn_returns_logits_for_ten_classes():
    model = MediumCNN(num_classes=10, in_channels=1, dropout=0.3)
    batch = torch.randn(4, 1, 28, 28)

    logits = model(batch)

    assert logits.shape == (4, 10)


def test_medium_cnn_state_dict_matches_submission_checkpoint_layout():
    model = MediumCNN(num_classes=10, in_channels=1, dropout=0.2167)
    state_keys = set(model.state_dict())

    assert "features.14.weight" in state_keys
    assert "features.15.running_mean" in state_keys
    assert "classifier.2.weight" in state_keys
    assert model.state_dict()["classifier.2.weight"].shape == (10, 128)


def test_build_model_accepts_aliases():
    assert isinstance(build_model("small"), SmallCNN)
    assert isinstance(build_model("medium_cnn"), MediumCNN)
    assert normalize_model_name("medium-cnn") == "medium_cnn"
    assert normalize_model_name("convnext") == "convnext_micro"


@pytest.mark.parametrize(
    "model_name",
    ["preact_resnet_tiny", "wide_resnet_tiny", "convnext_micro", "convstem_vit", "mobilenetv3_28"],
)
def test_heterogeneous_models_return_logits_for_ten_classes(model_name):
    model = build_model(model_name, num_classes=10, in_channels=1, dropout=0.1)
    batch = torch.randn(2, 1, 28, 28)

    logits = model(batch)

    assert logits.shape == (2, 10)


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError):
        build_model("unknown")


def test_small_cnn_backward_pass_populates_gradients():
    model = SmallCNN(num_classes=10, in_channels=1)
    batch = torch.randn(2, 1, 28, 28)
    labels = torch.tensor([0, 1])

    loss = torch.nn.CrossEntropyLoss()(model(batch), labels)
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.parameters())
