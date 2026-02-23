from __future__ import annotations

from src.training.train_deeplabv3 import TrainConfig, train


def main() -> None:
    cfg = TrainConfig(
        epochs=3,
        batch_size=2,
        image_size=256,
        lr=1e-3,
    )
    train(cfg)


if __name__ == "__main__":
    main()