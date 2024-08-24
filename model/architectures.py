from types import SimpleNamespace

import os
import pickle as pk
from functools import lru_cache

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..featurizer.protein import FOLDSEEK_MISSING_IDX
from ..utils import get_logger

logg = get_logger()

#################################
# Latent Space Distance Metrics #
#################################


class Cosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2)


class SquaredCosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2) ** 2


class Euclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0)


class SquaredEuclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0) ** 2


DISTANCE_METRICS = {
    "Cosine": Cosine,
    "SquaredCosine": SquaredCosine,
    "Euclidean": Euclidean,
    "SquaredEuclidean": SquaredEuclidean,
}

ACTIVATIONS = {"ReLU": nn.ReLU, "GELU": nn.GELU, "ELU": nn.ELU, "Sigmoid": nn.Sigmoid}

#######################
# Model Architectures #
#######################


class LogisticActivation(nn.Module):
    """
    Implementation of Generalized Sigmoid
    Applies the element-wise function:
    :math:`\\sigma(x) = \\frac{1}{1 + \\exp(-k(x-x_0))}`
    :param x0: The value of the sigmoid midpoint
    :type x0: float
    :param k: The slope of the sigmoid - trainable -  :math:`k \\geq 0`
    :type k: float
    :param train: Whether :math:`k` is a trainable parameter
    :type train: bool
    """

    def __init__(self, x0=0, k=1, train=False):
        super(LogisticActivation, self).__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]), requires_grad=False)
        self.k.requiresGrad = train

    def forward(self, x):
        """
        Applies the function to the input elementwise
        :param x: :math:`(N \\times *)` where :math:`*` means, any number of additional dimensions
        :type x: torch.Tensor
        :return: :math:`(N \\times *)`, same shape as the input
        :rtype: torch.Tensor
        """
        o = torch.clamp(
            # 1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1
            1 * (1 + torch.sigmoid(self.k * (x - self.x0))), min=0, max=1
            # 首先计算Sigmoid函数的值，然后通过torch.clamp将其限制在0到1之间，最后将结果压缩成一维（使用squeeze()）
        ).squeeze()
        return o

    def clip(self):
        """
        Restricts sigmoid slope :math:`k` to be greater than or equal to 0, if :math:`k` is trained.
        :meta private:
        """
        self.k.data.clamp_(min=0)


#######################
# Model Architectures #
#######################


class DeepCL(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
        dropout_rate=0.2  # Add a parameter for the dropout rate
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]
        self.dropout_rate = dropout_rate  # Save the dropout rate

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension),
            self.latent_activation(),
            # nn.Dropout(self.dropout_rate)  # Add dropout layer
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension),
            self.latent_activation(),
            nn.Dropout(self.dropout_rate),
            self.latent_activation(),# Add dropout layer
        )
        # self.prot_projector = nn.Sequential(
        #     nn.Linear(self.prot_emb_size, hidden_size),
        #     torch.nn.Dropout(p=0.5, inplace=False),
        #     latent_activation(),
        #     nn.Linear(hidden_size, latent_size),
        #     torch.nn.Dropout(p=0.5, inplace=False),
        #     latent_activation(),
        # )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)


    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze()


class DeepCLsigmoid(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
        dropout_rate=0.2  # Add a parameter for the dropout rate
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]
        self.dropout_rate = dropout_rate  # Save the dropout rate

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension),
            self.latent_activation(),
            # nn.Dropout(self.dropout_rate)  # Add dropout layer
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension),
            self.latent_activation(),
            nn.Dropout(self.dropout_rate),
            self.latent_activation(),# Add dropout layer
        )
        # self.prot_projector = nn.Sequential(
        #     nn.Linear(self.prot_emb_size, hidden_size),
        #     torch.nn.Dropout(p=0.5, inplace=False),
        #     latent_activation(),
        #     nn.Linear(hidden_size, latent_size),
        #     torch.nn.Dropout(p=0.5, inplace=False),
        #     latent_activation(),
        # )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)


    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()



