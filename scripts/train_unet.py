from __future__ import annotations

from src.training.train_unet import TrainConfig, train


def main() -> None:
    cfg = TrainConfig(
        epochs=1,
        batch_size=2,
        image_size=352,
        lr=1e-3,
        base=32,
    )
    train(cfg)


if __name__ == "__main__":
    main()