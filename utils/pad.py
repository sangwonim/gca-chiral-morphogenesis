import torch
import torch_scatter
from utils.sparse_tensor import SparseTensorWrapper
from typing import List


def get_shifts(padding, data_dim, pad_type='hypercubic', include_batch=False):
    """
    Arguments:
        padding: number of padding to add to shifts
        data_dim: dimension of data
    Returns
        shifts:
            Tensor of shape ((2 * padding + 1) ** data_dim) x  data_dim
            Each row of shifts represent nearby coordinates

    Ex)
        >>> get_shifts(1, 2)
        torch.tensor([
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1], [0, 0], [0, 1],
            [1, -1], [1, 0], [1, -1]
        ])
    """

    shifts = []
    if pad_type == 'hypercubic':
        for i in range(-padding, padding + 1):
            for j in range(-padding, padding + 1):
                if data_dim == 2:
                    shifts.append([i, j])
                    continue
                for k in range(-padding, padding + 1):
                    shifts.append([i, j, k])
    elif pad_type == 'hypercross':
        for x in range(padding + 1):
            for y in range(padding + 1 - x):
                if data_dim == 2:
                    shifts.append([x, y])
                    shifts.append([-x, y])
                    shifts.append([x, -y])
                    shifts.append([-x, -y])
                    continue
                for z in range(padding + 1 - x - y):
                    shifts.append([x, y, z])
                    shifts.append([-x, y, z])
                    shifts.append([x, -y, z])
                    shifts.append([x, y, -z])
                    shifts.append([-x, -y, z])
                    shifts.append([-x, y, -z])
                    shifts.append([x, -y, -z])
                    shifts.append([-x, -y, -z])
    elif pad_type == 'hybrid':
        for i in range(-padding, padding + 1):
            for j in range(-padding, padding + 1):
                for k in range(-padding, padding + 1):
                    if ((i == -padding) or (i == padding)) \
                        and ((j == -padding) or (j == padding)) \
                        and ((k == -padding) or (k == padding)):
                        continue
                    shifts.append([i, j, k])
    else:
        raise ValueError(f'padding {pad_type} not allowed')

    shifts = torch.unique(torch.Tensor(shifts), dim=0)
    if include_batch:
        # shifts = ME.utils.batched_coordinates(shifts)
        shifts = torch.cat([torch.zeros(shifts.shape[0], 1), shifts], dim=1)
    return shifts


def get_out_channels(padding, pad_type, data_dim, out_dim):
    if pad_type == 'hypercubic':
        return (2 * padding + 1) ** data_dim * out_dim
    elif pad_type == 'hypercross':
        if data_dim == 2:
            return (2 * padding ** 2 + 2 * padding + 1) * out_dim
        elif data_dim == 3:
            return ((4 * (padding ** 2) + 2 * padding + 1)
                    + (2 * padding * (padding - 1) * (2 * padding - 1)) // 3) * out_dim
    elif pad_type == 'hybrid':
        return (2 * padding + 1) ** data_dim * out_dim - (2 ** data_dim)


def unpack(x: SparseTensorWrapper, shifts, out_dim) -> SparseTensorWrapper:
    x_unpacked_coord = []
    x_unpacked_feat = []

    for batch_idx in range(x.batch_size):
        x_feat, x_coord = x.feats_and_coords_at(batch_idx)
        shifted_coord = torch.cat([x_coord + shift for shift in shifts])
        assert shifted_coord.is_cuda, 'coordinate is not on cuda before calling unique()'

        unique_coord, inverse_idxs = torch.unique(
            shifted_coord, dim=0, sorted=True, return_inverse=True
        )
        x_feat = torch.split(x_feat, split_size_or_sections=out_dim, dim=1)
        x_feat = torch.cat(x_feat)

        feat = torch_scatter.scatter(src=x_feat, index=inverse_idxs, dim=0, reduce="mean")
        x_unpacked_coord.append(unique_coord)
        x_unpacked_feat.append(feat)

    return SparseTensorWrapper(x_unpacked_feat, x_unpacked_coord)


def pad_gt_coord(y_hat_coord: torch.tensor, y_coord: torch.tensor) -> torch.tensor:
    '''
    Args:
        y_hat_coord: coordinate of N1 x dim
        y_coord: coordinate of N2 x dim
    Returns:
        x_next: torch.tensor of shape N1 with value of either 0 or 1
            the value is 1 for index i, if the y_hat_coord[i, :] is included in y_coord
            else the value is 0
    '''
    assert y_hat_coord.is_cuda, 'coordinate is not on cuda before calling unique()'
    y_unique_coord, y_count = torch.unique(
        torch.cat([y_hat_coord, y_hat_coord, y_coord.int()]),
        return_counts=True, dim=0, sorted=True
        # Why sort this? Sorting preserves order which is used for computing loss
    )
    # find indices not either in 1) x_coord or 2) x_coord and y_coord
    mask_idx = (y_count != 1)
    return y_count[mask_idx] - 2



def get_gt_values(x: SparseTensorWrapper, y: SparseTensorWrapper)\
        -> (List[torch.tensor], List[torch.tensor]):
    one_hot_gt = []
    y_pad_coords = []

    for batch_idx in range(x.batch_size):
        # This process might be inefficient, doing duplicate indexing
        x_coord = x.coords_at(batch_idx).to(x.device)
        y_coord = y.coords_at(batch_idx).to(y.device)
        # find out gt of the voxels
        assert x_coord.is_cuda, 'coordinate is not on cuda before calling unique()'
        y_unique_coord, y_count = torch.unique(
            torch.cat([x_coord, x_coord, y_coord.int()]),
            return_counts=True, dim=0, sorted=True
            # Why sort this? Sorting preserves order which is used for computing loss
        )
        # find indices not either in 1) x_coord or 2) x_coord and y_coord
        mask_idx = (y_count != 1)
        one_hot_gt.append(y_count[mask_idx] - 2)
        y_pad_coords.append(y_unique_coord[y_count == 3])
    return one_hot_gt, y_pad_coords

