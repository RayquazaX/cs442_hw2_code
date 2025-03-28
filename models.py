#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

logger = logging.getLogger(__name__)


class GradReverse(Function):
    """
    Implement the gradient reversal layer for the convenience of domain adaptation neural network.
    The forward part is the identity function while the backward part is the negative function.
    """

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output * -1
        return output, None


def grad_reverse(x):
    return GradReverse.apply(x)

class FairNet(nn.Module):
    """
    Multi-layer perceptron with adversarial training for fairness.
    """

    def __init__(self, configs):
        super(FairNet, self).__init__()
        self.input_dim = configs["input_dim"]
        self.num_classes = configs["num_classes"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                                      for i in range(self.num_hidden_layers)])
        # Parameter of the final softmax classification layer.
        self.softmax = nn.Linear(self.num_neurons[-1], configs["num_classes"])
        # Parameter of the adversary classification layer.
        self.num_adversaries = [self.num_neurons[-1]] + configs["adversary_layers"]
        self.num_adversaries_layers = len(configs["adversary_layers"])
        self.adversaries = nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])
                                          for i in range(self.num_adversaries_layers)])
        self.sensitive_cls = nn.Linear(self.num_adversaries[-1], 2)

    def forward(self, inputs):
        """
        The feature extractor is specified by self.hiddens.
        The label predictor is specified by self.softmax.
        The adversarial discriminator is specified by self.adversaries, followed by self.sensitive_cls.

        You need to return two things:
        1) The first thing is the log of the predicted probabilities (rather than predicted logits) from the label predictor.
        2) The second thing is the log of the predicted probabilities (rather than predicted logits) from the adversarial discriminator.

        Notice:
        For both the label predictor and the adversarial discriminator, we apply the ReLU activation on all layers
        except for the last linear layer.
        """
        # Feature extraction
        x = inputs
        for layer in self.hiddens:
            x = F.relu(layer(x))

        # Label prediction branch
        logits = self.softmax(x)
        label_logprobs = F.log_softmax(logits, dim=1)

        # Adversarial branch with gradient reversal
        adv_input = grad_reverse(x)
        for idx, layer in enumerate(self.adversaries):
            if idx < len(self.adversaries) - 1:
                adv_input = F.relu(layer(adv_input))
            else:
                adv_input = layer(adv_input)
        adv_logits = self.sensitive_cls(adv_input)
        adv_logprobs = F.log_softmax(adv_logits, dim=1)

        return label_logprobs, adv_logprobs

    def inference(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # Classification probability.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        return logprobs


class CFairNet(nn.Module):
    """
    Multi-layer perceptron with adversarial training for conditional fairness.
    """
    def __init__(self, configs):
        super(CFairNet, self).__init__()
        self.input_dim = configs["input_dim"]
        self.num_classes = configs["num_classes"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        # Parameters of hidden, fully-connected layers, feature learning component.
        self.hiddens = nn.ModuleList([nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                                      for i in range(self.num_hidden_layers)])
        # Parameter of the final softmax classification layer.
        self.softmax = nn.Linear(self.num_neurons[-1], configs["num_classes"])
        # Parameter of the conditional adversary classification layer.
        self.num_adversaries = [self.num_neurons[-1]] + configs["adversary_layers"]
        self.num_adversaries_layers = len(configs["adversary_layers"])
        # Conditional adversaries for sensitive attribute classification, one separate adversarial classifier for
        # one class label.
        self.adversaries = nn.ModuleList([nn.ModuleList([nn.Linear(self.num_adversaries[i], self.num_adversaries[i + 1])
                                                         for i in range(self.num_adversaries_layers)])
                                          for _ in range(self.num_classes)])
        self.sensitive_cls = nn.ModuleList([nn.Linear(self.num_adversaries[-1], 2) for _ in range(self.num_classes)])

    def forward(self, inputs, labels):
        """
        The feature extractor is specified by self.hiddens.
        The label predictor is specified by self.softmax.
        The adversarial discriminator is specified by self.adversaries, followed by self.sensitive_cls.

        You need to return two things:
        1) The first thing is the log of the predicted probabilities (rather than predicted logits) from the label predictor.
        2) The second thing is a list of the log of the predicted probabilities (rather than predicted logits) from the adversarial discriminator,
        where each list corresponds to one class (e.g., Y=0, Y=1, etc)

        Notice:
        For both the label predictor and the adversarial discriminator, we apply the ReLU activation on all layers except for the last linear layer.
        """
        # Feature extraction
        x = inputs
        for layer in self.hiddens:
            x = F.relu(layer(x))

        # Label prediction branch
        logits = self.softmax(x)
        label_logprobs = F.log_softmax(logits, dim=1)

        # Adversarial branch with gradient reversal
        adv_input = grad_reverse(x)

        # Compute adversarial predictions conditional on the provided labels
        adv_outputs = []
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                adv_feat = adv_input[mask]
                # Pass through the conditional adversary network for class c
                for idx, layer in enumerate(self.adversaries[c]):
                    if idx < len(self.adversaries[c]) - 1:
                        adv_feat = F.relu(layer(adv_feat))
                    else:
                        adv_feat = layer(adv_feat)
                adv_logits = self.sensitive_cls[c](adv_feat)
                adv_logprobs = F.log_softmax(adv_logits, dim=1)
                adv_outputs.append(adv_logprobs)
            else:
                # No samples for this class; append an empty tensor on the same device.
                adv_outputs.append(torch.empty(0, 2, device=inputs.device))
                
        return label_logprobs, adv_outputs

    def inference(self, inputs):
        h_relu = inputs
        for hidden in self.hiddens:
            h_relu = F.relu(hidden(h_relu))
        # Classification probabilities.
        logprobs = F.log_softmax(self.softmax(h_relu), dim=1)
        return logprobs
