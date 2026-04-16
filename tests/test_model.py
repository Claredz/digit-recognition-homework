import torch

from src.model import SmallCNN


def test_small_cnn_returns_logits_for_ten_classes():
    model = SmallCNN(num_classes=10, in_channels=1)
    batch = torch.randn(4, 1, 28, 28)

    logits = model(batch)

    assert logits.shape == (4, 10)



def test_small_cnn_backward_pass_populates_gradients():
    model = SmallCNN(num_classes=10, in_channels=1)
    batch = torch.randn(2, 1, 28, 28)
    labels = torch.tensor([0, 1])

    loss = torch.nn.CrossEntropyLoss()(model(batch), labels)
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.parameters())
