from __future__ import division

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


classes = load_classes(opt.class_path)

# Set data config path
data_config_path = "config/coco.data"
data_config = parse_data_config(opt.data_config_path)
train_path = data_config["train"]



# Initiate model
model_config_path = "config/yolov3.cfg"
weights_path = "config/my_weights.weights"
model = Darknet(model_config_path)
model.load_weights(weights_path)
model = model.cuda()


dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path), batch_size=1, shuffle=False, num_workers=opt.n_cpu
)


