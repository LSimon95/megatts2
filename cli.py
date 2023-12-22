# main.py
from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


def cli_main():
    cli = LightningCLI(DemoModel, BoringDataModule)
    # note: don't call fit!!


from modules.datamodule import test

if __name__ == "__main__":
    test()
    # note: it is good practice to implement the CLI in a function and call it in the main if block