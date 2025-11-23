# scripts/run_bernoulli_MNIST.py
from lightning.pytorch.cli import LightningCLI
from src.models.jsa import JSA
from src.data.mnist import MNISTDataModule
from hydra.utils import instantiate


def main():
    LightningCLI(JSA, MNISTDataModule, save_config_overwrite=True)


if __name__ == "__main__":
    main()
