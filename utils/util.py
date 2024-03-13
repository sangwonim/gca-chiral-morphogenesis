import torch
import time
import math
from MinkowskiEngine.utils import sparse_quantize


def timeit(func):
	def wrapper(*args, **kwargs):
		start_time = time.time()
		result = func(*args, **kwargs)
		time_executed = time.time() - start_time
		if hasattr(args[0], 'scalar_summaries'):  # args[0] is self
			args[0].scalar_summaries['resources/time/{}'.format(func.__name__)] += [time_executed]
		return result

	return wrapper


def downsample(x: torch.tensor, sample_num: int) -> torch.tensor:
	'''
	Args:
		x: torch tensor of N x 3
		sample_num: number of samples to downsample to
	Returns:
		downsampled output of torch tensor {sample_num} x 3
	'''
	if x.shape[0] == 0:
		return torch.zeros(1, 3)

	if x.shape[0] < sample_num:
		multiplier = (int(sample_num) // x.shape[0])
		x_multiply = torch.cat((x, ) * multiplier, dim=0)
		sample_num -= multiplier * x.shape[0]
		return torch.cat([downsample(x, sample_num), x_multiply], dim=0)
	rand_idx = torch.randperm(x.shape[0])
	keep_idx = rand_idx[:sample_num]
	return x[keep_idx, :]


def get_unique_rot_mats():
	rot_mats = []
	for i in range(4):
		i = torch.tensor((math.pi / 2) * i)
		for j in range(4):
			j = torch.tensor((math.pi / 2) * j)
			for k in range(4):
				k = torch.tensor((math.pi / 2) * k)
				rot_mat = torch.eye(3)
				rot_mat = torch.tensor([
					[1., 0., 0.],
					[0., torch.cos(i), -torch.sin(i)],
					[0., torch.sin(i), torch.cos(i)]
				]) @ rot_mat
				rot_mat = torch.tensor([
					[torch.cos(j), 0., -torch.sin(j)],
					[0., 1., 0.],
					[torch.sin(j), 0., torch.cos(j)]
				]) @ rot_mat
				rot_mat = torch.tensor([
					[torch.cos(k), -torch.sin(k), 0.],
					[torch.sin(k), torch.cos(k), 0.],
					[0., 0., 1.],
				]) @ rot_mat
				rot_mat = rot_mat.view(-1)
				rot_mat[torch.abs(rot_mat) < 0.1] = 0.
				rot_mats.append(rot_mat)
	rot_mats = torch.stack(rot_mats, dim=0)
	rot_mats = torch.unique(rot_mats, dim=0, sorted=True).double()  # 24 x 9
	return rot_mats.view(24, 3, 3).float()


def random_rotate(coord):
	rot_mat = torch.eye(3)
	for axis in range(3):
		angle = torch.randint(high=4, size=(1, 1))[0, 0].float()
		angle = (math.pi / 2) * angle
		if axis == 0:
			rot_mat = torch.tensor([
				[1., 0., 0.],
				[0., torch.cos(angle), -torch.sin(angle)],
				[0., torch.sin(angle), torch.cos(angle)]
			]) @ rot_mat
		elif axis == 1:
			rot_mat = torch.tensor([
				[torch.cos(angle), 0., -torch.sin(angle)],
				[0., 1., 0.],
				[torch.sin(angle), 0., torch.cos(angle)]
			]) @ rot_mat
		elif axis == 2:
			rot_mat = torch.tensor([
				[torch.cos(angle), -torch.sin(angle), 0.],
				[torch.sin(angle), torch.cos(angle), 0.],
				[0., 0., 1.],
			]) @ rot_mat
		else:
			raise ValueError('Argument has to be either 0 or 1 or 2')
	coord = coord.float() @ rot_mat
	return coord


def quantize_data(coord, feat, voxel_size, quantize='floor', coord_jitter=False):
	# Create SparseTensor
	coord = coord / voxel_size
	if isinstance(coord_jitter, bool):
		coord_jitter = float(coord_jitter)
	jitter = torch.floor(3 * torch.rand_like(coord) - 1)
	no_jitter_idx = torch.randperm(coord.shape[0])[int(coord_jitter * coord.shape[0]):]
	jitter[no_jitter_idx, :] = 0.
	coord = coord + jitter

	if (quantize == 'floor') or (quantize is None):
		coord = torch.floor(coord)
	elif quantize == 'round':
		coord = torch.round(coord)
	elif quantize == 'zero':
		coord[coord >= 0] = torch.floor(coord[coord >= 0])
		coord[coord < 0] = torch.ceil(coord[coord < 0])
	else:
		raise ValueError('quantization strategy {} not allowed'.format(quantize))
	coord, idxs = sparse_quantize(
		coord.cpu(), return_index=True, quantization_size=1
	)
	return coord, feat[idxs, :]
