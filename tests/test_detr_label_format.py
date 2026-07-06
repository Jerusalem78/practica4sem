from src.training.detr_train import build_model


def test_detr_config_has_background_label_slot():
    model = build_model(num_classes=3)
    assert model.config.num_labels == 4
