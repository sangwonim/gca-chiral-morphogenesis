import torch
from MinkowskiEngine import SparseTensor
from MinkowskiEngine.utils.collation import sparse_collate
from typing import List, Union


class SparseTensorWrapper:
    '''
    A wrapper class for sparse tensors.
    Used for experimenting between spconv and Minkowski engine sparse tensors.
    '''
    library = None
    spatial_shape = None

    def __init__(
            self, feats: torch.tensor, coords: torch.tensor,
            names=None, collated=False, device=None,
            coordinate_manager=None
    ):
        '''
        Args:
            if collated
                feats: torch.tensor of N x feat_dim, where N is the number of points
                coords: torch.tensor of N x (data_dim + 1), 1 for batch_size
            if not collated
                feats: list of torch.tensor of N x feat_dim, where N is the number of points
                coords: list of torch.tensor of N x (data_dim + 1), 1 for batch_size
            names: list of names of batches
            device: cpu or cuda
        Returns:
            sparse convolutional tensor for either spconv or MinkowskiEngine
        '''
        if collated:
            coords = coords[:feats.shape[0], :]
            self.batch_size = int(coords[:, 0].max().item()) + 1
        else:
            self.batch_size = len(feats)
            coords, feats = sparse_collate(coords, feats)
            coords = coords.to(feats.device)

        if device is not None:
            coords, feats = coords.to(device), feats.to(device)

        if SparseTensorWrapper.library == 'mink':
            self.sparse_tensor = SparseTensor(feats, coords, coordinate_manager=coordinate_manager)
        else:
            raise ValueError
        self.names = names

    def to(self, device):
        self.sparse_tensor = SparseTensor(self.feats.to(device), self.coords.to(device))
        return self

    @staticmethod
    def init_sparse_conv(config):
        '''
        run this method at first to set the library that we are
        '''
        SparseTensorWrapper.library = 'mink'
        # if config['backbone']['name'].startswith('Mink'):
        #     SparseTensorWrapper.library = 'mink'
        # else:
        #     raise NotImplementedError

    @property
    def feats(self) -> torch.tensor:
        return self.sparse_tensor.F if SparseTensorWrapper.library == 'mink' \
            else self.sparse_tensor.features

    @property
    def coords(self) -> torch.tensor:
        return self.sparse_tensor.C if SparseTensorWrapper.library == 'mink' \
            else self.sparse_tensor.indices

    def idx_at(self, batch_idx) -> torch.tensor:
        return self.coords[:, 0] == batch_idx

    def feats_at(self, batch_idx) -> torch.tensor:
        return self.feats[self.coords[:, 0] == batch_idx, :]

    def coords_at(self, batch_idx) -> torch.tensor:
        return self.coords[self.coords[:, 0] == batch_idx, 1:]

    def feats_and_coords_at(self, batch_idx) -> (torch.tensor, torch.tensor):
        valid_idxs = self.coords[:, 0] == batch_idx
        if valid_idxs.sum() == 0:
            raise RuntimeError('no valid coordinates inside the batch')
        return self.feats[valid_idxs, :], self.coords[valid_idxs, 1:]

    def set_feats(self, feats: torch.tensor):
        if SparseTensorWrapper.library == 'mink':
            self.sparse_tensor._F = feats
        else:
            raise ValueError

    @property
    def device(self):
        if SparseTensorWrapper.library == 'mink':
            return self.sparse_tensor._F.device
        else:
            raise ValueError

    @staticmethod
    def from_sparse_tensors(x: Union[SparseTensor]):
        if type(x) == SparseTensor:
            return SparseTensorWrapper(x.F, x.C, collated=True)
        else:
            raise ValueError

    @property
    def shape(self):
        return self.feats.shape

    @property
    def coordinate_manager(self):
        return self.sparse_tensor.coordinate_manager


