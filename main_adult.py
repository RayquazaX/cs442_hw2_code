#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import AdultDataset
from models import FairNet, CFairNet
from utils import conditional_errors
from utils import get_logger

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Name used to save the log file.", type=str, default="adult")
parser.add_argument("-s", "--seed", help="Random seed.", type=int, default=42)
# We will override the mu parameter in the loop.
parser.add_argument("-u", "--mu", help="Hyperparameter of the coefficient of the adversarial classification loss",
                    type=float, default=10.0)
parser.add_argument("-e", "--epoch", help="Number of training epochs", type=int, default=100)
parser.add_argument("-r", "--lr", type=float, help="Learning rate of optimization", default=1.0)
parser.add_argument("-b", "--batch_size", help="Batch size during training", type=int, default=512)
parser.add_argument("-m", "--model", help="Which model to run: [fair|cfair-eo]", type=str,
                    default="fair")
parser.add_argument("-y", "--target", help="Name of the target attribute", type=str, default="income")
parser.add_argument("-p", "--private", help="Name of the sensitive/private attribute", type=str, default="sex")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8)

logger = get_logger(args.name)

# Set random seed.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
dtype = np.float32

logger.info("UCI Adult data set, target attribute: {}, sensitive attribute: {}".format(args.target, args.private))
# Load UCI Adult dataset.
time_start = time.time()
adult_train = AdultDataset(root_dir='data', phase='train', tar_attr=args.target, priv_attr=args.private)
adult_test = AdultDataset(root_dir='data', phase='test', tar_attr=args.target, priv_attr=args.private)
train_loader = DataLoader(adult_train, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(adult_test, batch_size=args.batch_size, shuffle=False)
time_end = time.time()
logger.info("Time used to load all the data sets: {} seconds.".format(time_end - time_start))
input_dim = adult_train.xdim
num_classes = 2
num_groups = 2

# Pre-compute statistics from training set.
train_target_attrs = np.argmax(adult_train.A, axis=1)
train_target_labels = np.argmax(adult_train.Y, axis=1)
train_idx = train_target_attrs == 0
train_base_0, train_base_1 = np.mean(train_target_labels[train_idx]), np.mean(train_target_labels[~train_idx])
train_y_1 = np.mean(train_target_labels)
if args.model == "cfair":
    reweight_target_tensor = torch.tensor([1.0 / (1.0 - train_y_1), 1.0 / train_y_1]).float().to(device)
elif args.model == "cfair-eo":
    reweight_target_tensor = torch.tensor([1.0, 1.0]).float().to(device)
reweight_attr_0_tensor = torch.tensor([1.0 / (1.0 - train_base_0), 1.0 / train_base_0]).float().to(device)
reweight_attr_1_tensor = torch.tensor([1.0 / (1.0 - train_base_1), 1.0 / train_base_1]).float().to(device)
reweight_attr_tensors = [reweight_attr_0_tensor, reweight_attr_1_tensor]

logger.info("Average value of A = {}".format(np.mean(train_target_attrs)))
logger.info("A: Male = 0, Female = 1")
# Pre-compute statistics from test set.
target_insts = torch.from_numpy(adult_test.X).float().to(device)
target_labels = np.argmax(adult_test.Y, axis=1)
target_attrs = np.argmax(adult_test.A, axis=1)
test_idx = target_attrs == 0
conditional_idx = target_labels == 0
base_0, base_1 = np.mean(target_labels[test_idx]), np.mean(target_labels[~test_idx])
label_marginal = np.mean(target_labels)
logger.info("Value of Base 0: {}, value of Base 1: {}".format(base_0, base_1))

# Configurations.
configs = {"num_classes": num_classes, "num_groups": num_groups, "num_epochs": args.epoch,
           "batch_size": args.batch_size, "lr": args.lr, "mu": args.mu, "input_dim": input_dim,
           "hidden_layers": [60], "adversary_layers": [50]}
num_epochs = configs["num_epochs"]
batch_size = configs["batch_size"]
lr = configs["lr"]

# Prepare lists to store metrics for different mu values.
mu_list = [0.1, 1, 10]
overall_errors = []
stat_parity_gaps = []
equalized_odds_gaps = []

for mu_val in mu_list:
    logger.info("Starting experiment with adversarial loss weight mu = {}".format(mu_val))
    args.mu = mu_val  # update mu parameter
    configs["mu"] = mu_val

    # Reset random seeds for each run (optional).
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
                adv_loss = torch.mean(torch.stack([F.nll_loss(apreds[j], attrs[ys == j],
                                                                 weight=reweight_attr_tensors[j])
                                                    for j in range(num_classes)]))
                running_loss += loss.item()
                running_adv_loss += adv_loss.item()
                loss += mu * adv_loss
                loss.backward()
                optimizer.step()
            logger.info("Iteration {}, loss = {}, adv_loss = {}".format(t, running_loss, running_adv_loss))
    else:
        raise NotImplementedError("{} not supported.".format(args.model))

    # Evaluate on test set.
    net.eval()
    preds_labels = torch.max(net.inference(target_insts), 1)[1].cpu().numpy()
    cls_error, error_0, error_1 = conditional_errors(preds_labels, target_labels, target_attrs)
    # Compute group-wise prediction averages.
    test_idx_bool = target_attrs == 0
    pred_0 = np.mean(preds_labels[test_idx_bool])
    pred_1 = np.mean(preds_labels[~test_idx_bool])
    # For equalized odds, compute conditional predictions:
    cond_00 = np.mean(preds_labels[np.logical_and(test_idx_bool, conditional_idx)])
    cond_10 = np.mean(preds_labels[np.logical_and(~test_idx_bool, conditional_idx)])
    cond_01 = np.mean(preds_labels[np.logical_and(test_idx_bool, ~conditional_idx)])
    cond_11 = np.mean(preds_labels[np.logical_and(~test_idx_bool, ~conditional_idx)])
    
    # Compute metrics.
    stat_gap = np.abs(pred_0 - pred_1)
    eq_odds_gap = 0.5 * (np.abs(cond_00 - cond_10) + np.abs(cond_01 - cond_11))
    
    overall_errors.append(cls_error)
    stat_parity_gaps.append(stat_gap)
    equalized_odds_gaps.append(eq_odds_gap)
    
    logger.info("For mu = {}: Overall error = {}, Statistical Parity Gap = {}, Equalized Odds Gap = {}"
                .format(mu_val, cls_error, stat_gap, eq_odds_gap))
    
    # Optionally, save predictions/results for each mu.
    out_file = "adult_{}_{}.npz".format(args.model, mu_val)
    np.savez(out_file, prediction=preds_labels, truth=target_labels, attribute=target_attrs)

# Plot all three metrics in one figure.
plt.figure(figsize=(8,6))
plt.plot(mu_list, overall_errors, marker='o', linestyle='-', color='blue', label='Overall Error')
plt.plot(mu_list, stat_parity_gaps, marker='s', linestyle='--', color='green', label='Statistical Parity Gap')
plt.plot(mu_list, equalized_odds_gaps, marker='^', linestyle='-.', color='red', label='Equalized Odds Gap')
plt.xlabel("Adversarial Loss Weight (mu)")
plt.ylabel("Metric Value")
plt.title("Metrics vs. Adversarial Loss Weight (Adult Dataset)")
plt.xticks(mu_list)
plt.legend()
plt.grid(True)
plt.savefig("combined_metrics_adult.png")
plt.show()
