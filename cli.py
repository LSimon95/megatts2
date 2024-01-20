# main.py
from lightning.pytorch.cli import LightningCLI

from models.trainer import MegaGANTrainer
from modules.datamodule import test



# def cli_main():
#     cli = LightningCLI(MegaGANTrainer, TTSDataModule)

if __name__ == "__main__":
    test()
    # note: it is good practice to implement the CLI in a function and call it in the main if block