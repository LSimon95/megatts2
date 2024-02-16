# main.py
from lightning.pytorch.cli import LightningCLI

from models.trainer import MegaGANTrainer, MegaPLMTrainer, MegaADMTrainer
from modules.datamodule import TTSDataModule, test
from modules.feat_extractor import test



def cli_main():
    cli = LightningCLI(MegaGANTrainer, TTSDataModule)

if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block