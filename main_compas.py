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
import matplotlib.pyplot as plt

from dataset import COMPAS
from models import FairNet, CFairNet
from utils import conditional_errors
from utils import get_logger

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Name used to save the log file.", type=str, default="compas")
parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)
# We override the mu parameter in the loop.
parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient of the adversarial classification loss",
                    type=float, default=1.0)
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=50)
parser.add_argument("-r", "--lr", type=float, help="Learning rate of optimization", default=1.0)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=512)
parser.add_argument("-m", "--model", help="Which model to run: [fair|cfair-eo]", type=str,
                    default="mlp")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = get_logger(args.name)

# Set random seed.
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
indices = np.arange(num_insts)
np.random.shuffle(indices)
compas = compas[indices]
ratio = 0.7
num_train = int(num_insts * ratio)
compas_train = COMPAS(compas[:num_train, :])
compas_test = COMPAS(compas[num_train:, :])
train_loader = DataLoader(compas_train, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(compas_test, batch_size=args.batch_size, shuffle=False)
input_dim = compas_train.xdim
time_end = time.time()
logger.info("Time used to load all the data sets: {} seconds.".format(time_end - time_start))

# Pre-compute statistics in the training set.
idx = compas_train.attrs == 0
train_base_0, train_base_1 = np.mean(compas_train.labels[idx]), np.mean(compas_train.labels[~idx])
train_y_1 = np.mean(compas_train.labels)
if args.model == "cfair-eo":
    reweight_target_tensor = torch.tensor([1.0, 1.0]).float().to(device)
reweight_attr_0_tensor = torch.tensor([1.0 / (1.0 - train_base_0), 1.0 / train_base_0]).float().to(device)
reweight_attr_1_tensor = torch.tensor([1.0 / (1.0 - train_base_1), 1.0 / train_base_1]).float().to(device)
reweight_attr_tensors = [reweight_attr_0_tensor, reweight_attr_1_tensor]

# Pre-compute statistics in the test set.
target_insts = torch.from_numpy(compas_test.insts).to(device)
target_labels = compas_test.labels
target_attrs = compas_test.attrs
test_idx = target_attrs == 0
conditional_idx = target_labels == 0
base_0, base_1 = np.mean(target_labels[test_idx]), np.mean(target_labels[~test_idx])
label_marginal = np.mean(target_labels)

# Configs.
configs = {"num_classes": num_classes, "num_groups": num_groups, "num_epochs": args.epoch,
           "batch_size": args.batch_size, "lr": args.lr, "mu": args.mu, "input_dim": input_dim,
           "hidden_layers": [10], "adversary_layers": [10]}
num_epochs = configs["num_epochs"]
batch_size = configs["batch_size"]
lr = configs["lr"]

# Lists to store metrics for different mu values.
mu_list = [0.1, 1, 10]
overall_errors = []
stat_parity_gaps = []
equalized_odds_gaps = []

for mu_val in mu_list:
    logger.info("Starting experiment with adversarial loss weight mu = {}".format(mu_val))
    args.mu = mu_val
    configs["mu"] = mu_val

    # Reset random seeds for each run.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.model == "fair":
        logger.info("Training with FairNet (mu = {})".format(mu_val))
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
                loss = F.nll_loss(ypreds, ys)
                adv_loss = F.nll_loss(apreds, attrs)
                running_loss += loss.item()
                running_adv_loss += adv_loss.item()
                loss += mu * adv_loss
                loss.backward()
                optimizer.step()
            logger.info("Iteration {}, loss = {}, adv_loss = {}".format(t, running_loss, running_adv_loss))
    elif args.model == "cfair-eo":
        logger.info("Training with CFairNet (mu = {})".format(mu_val))
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
                loss = F.nll_loss(ypreds, ys, weight=reweight_target_tensor)
                adv_loss = torch.mean(torch.stack([F.nll_loss(apreds[j], attrs[ys == j], weight=reweight_attr_tensors[j]) for j in range(num_classes)]))
                running_loss += loss.item()
                running_adv_loss += adv_loss.item()
                loss += mu * adv_loss
                loss.backward()
                optimizer.step()
            logger.info("Iteration {}, loss = {}, adv_loss = {}".format(t, running_loss, running_adv_loss))
    else:
        raise NotImplementedError("{} not supported.".format(args.model))

    # Evaluation.
    net.eval()
    preds_labels = torch.max(net.inference(target_insts), 1)[1].cpu().numpy()
    cls_error, error_0, error_1 = conditional_errors(preds_labels, target_labels, target_attrs)
    test_idx_bool = target_attrs == 0
    pred_0 = np.mean(preds_labels[test_idx_bool])
    pred_1 = np.mean(preds_labels[~test_idx_bool])
    cond_00 = np.mean(preds_labels[np.logical_and(test_idx_bool, conditional_idx)])
    cond_10 = np.mean(preds_labels[np.logical_and(~test_idx_bool, conditional_idx)])
    cond_01 = np.mean(preds_labels[np.logical_and(test_idx_bool, ~conditional_idx)])
    cond_11 = np.mean(preds_labels[np.logical_and(~test_idx_bool, ~conditional_idx)])
    
    stat_gap = np.abs(pred_0 - pred_1)
    eq_odds_gap = 0.5 * (np.abs(cond_00 - cond_10) + np.abs(cond_01 - cond_11))
    
    overall_errors.append(cls_error)
    stat_parity_gaps.append(stat_gap)
    equalized_odds_gaps.append(eq_odds_gap)
    
    logger.info("For mu = {}: Overall error = {}, Statistical Parity Gap = {}, Equalized Odds Gap = {}"
                .format(mu_val, cls_error, stat_gap, eq_odds_gap))
    
    out_file = "compas_{}_{}.npz".format(args.model, mu_val)
    np.savez(out_file, prediction=preds_labels, truth=target_labels, attribute=target_attrs)

# Plot combined metrics.
plt.figure(figsize=(8,6))
plt.plot(mu_list, overall_errors, marker='o', linestyle='-', color='blue', label='Overall Error')
plt.plot(mu_list, stat_parity_gaps, marker='s', linestyle='--', color='green', label='Statistical Parity Gap')
plt.plot(mu_list, equalized_odds_gaps, marker='^', linestyle='-.', color='red', label='Equalized Odds Gap')
plt.xlabel("Adversarial Loss Weight (mu)")
plt.ylabel("Metric Value")
plt.title("Metrics vs. Adversarial Loss Weight (COMPAS Dataset)")
plt.xticks(mu_list)
plt.legend()
plt.grid(True)
plt.savefig("combined_metrics_compas.png")
plt.show()
