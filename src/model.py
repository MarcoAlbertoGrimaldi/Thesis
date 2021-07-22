import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl


class Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.
        
