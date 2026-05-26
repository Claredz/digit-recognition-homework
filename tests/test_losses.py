import pytest
import torch

from src.losses import AntiClass1MarginLoss, build_criterion


class TestAntiClass1MarginLoss:
    @pytest.fixture
    def logits(self):
        return torch.randn(4, 10)

    @pytest.fixture
    def labels_without_target(self):
        return torch.tensor([0, 2, 3, 4])

    @pytest.fixture
    def labels_all_target(self):
        return torch.tensor([1, 1, 1, 1])

    def test_disabled_equals_plain_ce(self, logits, labels_without_target):
        plain = torch.nn.CrossEntropyLoss(label_smoothing=0.0)(logits, labels_without_target)
        anti = build_criterion(label_smoothing=0.0, anti_class1_loss_config={"enabled": False})(logits, labels_without_target)
        assert torch.allclose(plain, anti, atol=1e-6), f"plain={plain.item():.6f} anti={anti.item():.6f}"

    def test_enabled_adds_penalty(self, logits, labels_without_target):
        plain = torch.nn.CrossEntropyLoss()(logits, labels_without_target)
        anti = AntiClass1MarginLoss(lam=0.5, margin=0.2, target_class=1)(logits, labels_without_target)
        assert anti.item() > plain.item(), f"penalty should increase loss: plain={plain.item():.4f} anti={anti.item():.4f}"

    def test_all_target_class_zero_penalty(self, logits, labels_all_target):
        plain = torch.nn.CrossEntropyLoss()(logits, labels_all_target)
        anti: AntiClass1MarginLoss = build_criterion(label_smoothing=0.0, anti_class1_loss_config={"enabled": True, "lambda": 10.0, "margin": 100.0, "target_class": 1})
        result = anti(logits, labels_all_target)
        assert torch.allclose(plain, result, atol=1e-6), f"all-target batch should be penalty-free: plain={plain.item():.6f} anti={result.item():.6f}"

    def test_mixup_composition(self):
        logits = torch.randn(4, 10, requires_grad=True)
        labels_a = torch.tensor([0, 2])
        labels_b = torch.tensor([3, 4])
        lam = 0.3
        criterion = AntiClass1MarginLoss(lam=0.05, margin=0.2, target_class=1)
        loss_a = criterion(logits[:2], labels_a)
        loss_b = criterion(logits[:2], labels_b)
        composed = lam * loss_a + (1.0 - lam) * loss_b
        assert composed.grad_fn is not None
        assert composed.item() > 0


class TestBuildCriterion:
    def test_none_config_returns_plain_ce(self):
        crit = build_criterion(label_smoothing=0.1, anti_class1_loss_config=None)
        assert isinstance(crit, torch.nn.CrossEntropyLoss)

    def test_disabled_config_returns_plain_ce(self):
        crit = build_criterion(label_smoothing=0.1, anti_class1_loss_config={"enabled": False})
        assert isinstance(crit, torch.nn.CrossEntropyLoss)

    def test_enabled_returns_anti_loss(self):
        crit = build_criterion(label_smoothing=0.02, anti_class1_loss_config={"enabled": True, "lambda": 0.05, "margin": 0.2, "target_class": 1})
        assert isinstance(crit, AntiClass1MarginLoss)
