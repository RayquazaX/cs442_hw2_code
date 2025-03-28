#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import COMPAS
from models import FairNet, CFairNet
from utils import conditional_errors
from utils import get_logger

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Name used to save the log file.", type=str, default="compas")
parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)
parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient of the adversarial classification loss",
                    type=float, default=1.0)
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=50)
parser.add_argument("-r", "--lr", type=float, help="Learning rate of optimization", default=1.0)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=512)
parser.add_argument("-m", "--model", help="Which model to run: [fair|cfair-eo]", type=str,
                    default="mlp")
# Compile and configure all the model parameters.
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = get_logger(args.name)

# Set random number seed.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
dtype = np.float32

logger.info("Propublica COMPAS data set, target attribute: recidivism classification, sensitive attribute: race")
# Load COMPAS dataset.
time_start = time.time()
compas = pd.read_csv("data/propublica.csv").values
logger.debug("Shape of COMPAS dataset: {}".format(compas.shape))
# Random shuffle and then partition by 70/30.
num_classes = 2
num_groups = 2
num_insts = compas.shape[0]
logger.info("Total number of instances in the COMPAS data: {}".format(num_insts))
# Random shuffle the dataset.
indices = np.arange(num_insts)
np.random.shuffle(indices)
compas = compas[indices]
# Partition the dataset into train and test split.
ratio = 0.7
num_train = int(num_insts * ratio)
compas_train = COMPAS(compas[:num_train, :])
compas_test = COMPAS(compas[num_train:, :])

train_loader = DataLoader(compas_train, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(compas_test, batch_size=args.batch_size, shuffle=False)
input_dim = compas_train.xdim
time_end = time.time()
logger.info("Time used to load all the data sets: {} seconds.".format(time_end - time_start))

# Pre-compute the statistics in the training set.
idx = compas_train.attrs == 0
train_base_0, train_base_1 = np.mean(compas_train.labels[idx]), np.mean(compas_train.labels[~idx])
train_y_1 = np.mean(compas_train.labels)
# For reweighing purpose.
if args.model == "cfair-eo":
    reweight_target_tensor = torch.tensor([1.0, 1.0]).float().to(device)
reweight_attr_0_tensor = torch.tensor([1.0 / (1.0 - train_base_0), 1.0 / train_base_0]).float().to(device)
reweight_attr_1_tensor = torch.tensor([1.0 / (1.0 - train_base_1), 1.0 / train_base_1]).float().to(device)
reweight_attr_tensors = [reweight_attr_0_tensor, reweight_attr_1_tensor]

# Pre-compute the statistics in the test set.
target_insts = torch.from_numpy(compas_test.insts).to(device)
target_labels = compas_test.labels
target_attrs = compas_test.attrs
test_idx = target_attrs == 0
conditional_idx = target_labels == 0
base_0, base_1 = np.mean(target_labels[test_idx]), np.mean(target_labels[~test_idx])
label_marginal = np.mean(target_labels)
cls_error, error_0, error_1 = 0.0, 0.0, 0.0
# Pr(Pred = 1 | A = 0, 1)
pred_0, pred_1 = 0.0, 0.0
# Pr(Pred = 1 | A = 0, 1, Y = 0, 1)
cond_00, cond_01, cond_10, cond_11 = 0.0, 0.0, 0.0, 0.0

# Configs.
configs = {"num_classes": num_classes, "num_groups": num_groups, "num_epochs": args.epoch,
           "batch_size": args.batch_size, "lr": args.lr, "mu": args.mu, "input_dim": input_dim,
           "hidden_layers": [10], "adversary_layers": [10]}
num_epochs = configs["num_epochs"]
batch_size = configs["batch_size"]
lr = configs["lr"]

if args.model == "fair":
    # Training with FairNet to show the debiased results.
    logger.info("Experiment with FairNet: {} adversarial debiasing:".format(args.model))
    logger.info("Hyperparameter setting = {}.".format(configs))
    time_start = time.time()
    net = FairNet(configs).to(device)
    logger.info("Model architecture: {}".format(net))
    optimizer = optim.Adadelta(net.parameters(), lr=lr)
    mu = args.mu
    net.train()
    for t in range(num_epochs):
        running_loss, running_adv_loss = 0.0, 0.0
        for xs, ys, attrs in train_loader:
            xs, ys, attrs = xs.to(device), ys.to(device), attrs.to(device)
            optimizer.zero_grad()
            ypreds, apreds = net(xs)
            # Compute both the prediction loss and the adversarial loss.
            loss = F.nll_loss(ypreds, ys)
            adv_loss = F.nll_loss(apreds, attrs)
            running_loss += loss.item()
            running_adv_loss += adv_loss.item()
            loss += mu * adv_loss
            loss.backward()
            optimizer.step()
        logger.info("Iteration {}, loss value = {}, adv_loss value = {}".format(t, running_loss, running_adv_loss))
    time_end = time.time()
    logger.info("Time used for training = {} seconds.".format(time_end - time_start))
    net.eval()
    preds_labels = torch.max(net.inference(target_insts), 1)[1].cpu().numpy()
    cls_error, error_0, error_1 = conditional_errors(preds_labels, target_labels, target_attrs)
    pred_0, pred_1 = np.mean(preds_labels[test_idx]), np.mean(preds_labels[~test_idx])
    cond_00 = np.mean(preds_labels[np.logical_and(test_idx, conditional_idx)])
    cond_10 = np.mean(preds_labels[np.logical_and(~test_idx, conditional_idx)])
    cond_01 = np.mean(preds_labels[np.logical_and(test_idx, ~conditional_idx)])
    cond_11 = np.mean(preds_labels[np.logical_and(~test_idx, ~conditional_idx)])

elif args.model == "cfair-eo":
    # Training with CFairNet to show the debiased results.
    logger.info("Experiment with CFairNet: {} adversarial debiasing:".format(args.model))
    logger.info("Hyperparameter setting = {}.".format(configs))
    time_start = time.time()
    net = CFairNet(configs).to(device)
    logger.info("Model architecture: {}".format(net))
    optimizer = optim.Adadelta(net.parameters(), lr=lr)
    mu = args.mu
    net.train()
    for t in range(num_epochs):
        running_loss, running_adv_loss = 0.0, 0.0
        for xs, ys, attrs in train_loader:
            xs, ys, attrs = xs.to(device), ys.to(device), attrs.to(device)
            optimizer.zero_grad()
            ypreds, apreds = net(xs, ys)
            # Compute both the prediction loss and the adversarial loss. Note that in CFairNet, both are conditional
            # losses.
            loss = F.nll_loss(ypreds, ys, weight=reweight_target_tensor)
            adv_loss = torch.mean(torch.stack([F.nll_loss(apreds[j], attrs[ys == j], weight=reweight_attr_tensors[j]) for j in range(num_classes)]))
            running_loss += loss.item()
            running_adv_loss += adv_loss.item()
            loss += mu * adv_loss
            loss.backward()
            optimizer.step()
        logger.info("Iteration {}, loss value = {}, adv_loss value = {}".format(t, running_loss, running_adv_loss))
    time_end = time.time()
    logger.info("Time used for training = {} seconds.".format(time_end - time_start))
    net.eval()
    preds_labels = torch.max(net.inference(target_insts), 1)[1].cpu().numpy()
    cls_error, error_0, error_1 = conditional_errors(preds_labels, target_labels, target_attrs)
    pred_0, pred_1 = np.mean(preds_labels[test_idx]), np.mean(preds_labels[~test_idx])
    cond_00 = np.mean(preds_labels[np.logical_and(test_idx, conditional_idx)])
    cond_10 = np.mean(preds_labels[np.logical_and(~test_idx, conditional_idx)])
    cond_01 = np.mean(preds_labels[np.logical_and(test_idx, ~conditional_idx)])
    cond_11 = np.mean(preds_labels[np.logical_and(~test_idx, ~conditional_idx)])
else:
    raise NotImplementedError("{} not supported.".format(args.model))

# Print out all the statistics.
logger.info("The global marginal label distribution of Y = 1: {}".format(label_marginal))
logger.info("Overall predicted error = {}, Err|A=0 = {}, Err|A=1 = {}".format(cls_error, error_0, error_1))
logger.info("Statistical Parity Gap: |Pred=1|A=0 - Pred=1|A=1| = {}".format(np.abs(pred_0 - pred_1)))
logger.info("Equalized Odds Gap Y = 0: |Pred = 1|A = 0, Y = 0 - Pred = 1|A = 1, Y = 0| = {}".format(np.abs(cond_00 - cond_10)))
logger.info("Equalized Odds Gap Y = 1: |Pred = 1|A = 0, Y = 1 - Pred = 1|A = 1, Y = 1| = {}".format(np.abs(cond_01 - cond_11)))

# Save all the results.
out_file = "compas_{}_{}.npz".format(args.model, args.mu)
np.savez(out_file, prediction=preds_labels, truth=target_labels, attribute=target_attrs)
