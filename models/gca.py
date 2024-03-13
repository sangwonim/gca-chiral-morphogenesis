import os
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List

from models.backbones import BACKBONES
from models.base_model import Model
from utils.solvers import build_lr_scheduler, build_optimizer
from utils.pad import get_shifts
from utils.sparse_tensor import SparseTensorWrapper
from utils.util import timeit
from utils.chamfer_distance import ChamferDistance
from utils.metrics import MMDCalculator


class GCA(Model, ABC):
    def __init__(self, config, writer: SummaryWriter):
        Model.__init__(self, config, writer)
        # The output sparse tensor is packed with neighboring information
        self.backbone = BACKBONES[config['backbone']['name']](config)
        # initialize the sparse tensor to choose
        SparseTensorWrapper.init_sparse_conv(config)

        self.out_dim = config['backbone']['out_channels']
        self.optimizer = build_optimizer(
            self.config['optimizer'], self.parameters()
        )
        self.lr_scheduler = build_lr_scheduler(
            self.config['lr_scheduler'], self.optimizer
        )
        self.shifts = get_shifts(
            padding=self.config['padding'],
            pad_type=self.config['pad_type'], data_dim=self.config['data_dim'],
        ).to(self.config['device'])
        self.shift_size = self.shifts.shape[0]
        self._chamfer_dist = ChamferDistance()

        if (self.config.get('task') is None) or (self.config.get('task') == 'completion'):
            self.mmd_calculator = MMDCalculator(config)
            self.mean_aligned_mmd_calculator = MMDCalculator(config)
            self.min_aligned_mmd_calculator = MMDCalculator(config)

    def transition(self, s: SparseTensorWrapper) -> SparseTensorWrapper:
        s = s.to(self.device)
        y_hat = self.forward(s)
        feat_sample = self.sample_feat(y_hat.feats)
        s_next_coord = y_hat.coords[feat_sample.bool(), :]

        # if the sampled output contains no coords
        for batch_idx in range(s.batch_size):
            if (s_next_coord[:, 0] == batch_idx).shape[0] == 0:
                if s_next_coord[:, 0].shape[0] == 0:
                    s_next_coord = torch.zeros(1, 4).int().to(s_next_coord.device)
                else:
                    s_next_coord = torch.stack([
                        s_next_coord,
                        torch.tensor([[batch_idx] + [0, ] * self.config['data_dim']]).int().to(s_next_coord.device)
                    ], dim=0)

        s_next_feat = torch.ones(s_next_coord.shape[0], 1)
        try:
            s_next = SparseTensorWrapper(
                s_next_feat, s_next_coord,
                collated=True, device=self.device
            )
        except RuntimeError:
            breakpoint()
        return s_next

    def sample_feat(self, feat: torch.tensor) -> torch.tensor:
        '''
        Takes feature of torch tensor as input and returns sampled features

        Args:
            feat: torch.tensor of shape N x param_dim
        Output:
            tensor of shape N
        '''
        sampling_scheme = self.config['sampling_scheme']
        if sampling_scheme == 'bernoulli':
            assert feat.shape[1] == 1, \
                'Expected feature shape of 1 for bernoulli, but got {}'.format(feat.shape[1])
            return torch.bernoulli(torch.sigmoid(feat.squeeze(1)))
        elif sampling_scheme == 'ml':
            assert feat.shape[1] == 1, \
                'Expected feature shape of 1 for bernoulli, but got {}'.format(feat.shape[1])
            return (torch.sigmoid(feat.squeeze(1)) > 0.5).float()
        else:
            raise ValueError

    def visualize_imgs(self, dataset, vis_indices: List, step):
        raise NotImplementedError()

    def visualize_all_imgs(self, dataset, step):
        vis_indices = list(range(len(dataset)))
        write_dir = os.path.join(
            self.config['log_dir'],
            'visualization', 'step-{}'.format(step)
        )
        os.makedirs(write_dir, exist_ok=True)
        self.visualize_imgs(
            dataset, vis_indices, step
        )

    def evaluate(self, x: SparseTensorWrapper, y: SparseTensorWrapper, step) -> float:
        raise NotImplementedError()

    def test(self, x: SparseTensorWrapper, y: SparseTensorWrapper, step) -> (float, float, float):
        raise NotImplementedError()

    def generative_test(self, x: SparseTensorWrapper, y: SparseTensorWrapper, generated_set, step):
        raise NotImplementedError()

    def visualize_debug(self, dataset, vis_indices, step):
        raise NotImplementedError()
