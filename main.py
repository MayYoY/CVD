from dataset import utils, vipl_hr
from configs import running
from model import train_test

import pandas as pd
from tqdm.auto import tqdm
import argparse
import torch
import os


train_test.fixSeed(42)
train_test.train_test("./saved", running.TrainConfig, running.TestConfig)
