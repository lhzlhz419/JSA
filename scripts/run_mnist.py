# scripts/run_mnist.py
from lightning.pytorch.cli import LightningCLI
import torch
torch.set_float32_matmul_precision("medium")

from dotenv import load_dotenv
load_dotenv() # Load environment variables from a .env file if present


def main():
    LightningCLI(run=True)


if __name__ == "__main__":
    main()
