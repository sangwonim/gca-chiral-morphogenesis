import torch
import os
import numpy as np
from abc import ABC
from collections import Iterator
from torch.utils.data import (
	Sampler,
	Dataset,
	DataLoader,
)
from torch.utils.tensorboard import SummaryWriter
from typing import List
from models.base_model import Model
from models.gca import GCA
from utils.phase import Phase
from utils.sparse_tensor import SparseTensorWrapper
from utils.util import get_unique_rot_mats, random_rotate, quantize_data
import glob
import random
import math
import point_cloud_utils as pcu



# =====================
# Base Classes and ABCs
# =====================

class RepetitiveSampler(Sampler):
	def __init__(self, data_source, iter_cnt):
		self.data_source = data_source
		self.iter_cnt = iter_cnt

	def __iter__(self):
		return iter([int(i / self.iter_cnt) for i in range(self.iter_cnt * len(self.data_source))])

	def __len__(self):
		return self.iter_cnt * len(self.data_source)


class InfiniteRandomSampler(Sampler):
	def __init__(self, dataset, max_samples=None):
		super().__init__(dataset)
		self.dataset = dataset
		self.len_data = len(self.dataset)
		self.iterator = iter(torch.randperm(self.len_data).tolist())
		self.max_samples = max_samples
		self.sample_num = 0

	def __next__(self):
		if self.max_samples is not None and \
				self.sample_num >= self.max_samples:
			raise StopIteration

		self.sample_num += 1

		try:
			idx = next(self.iterator)
		except StopIteration:
			self.iterator = iter(torch.randperm(self.len_data).tolist())
			idx = next(self.iterator)

		return idx

	def __iter__(self):
		return self

	def __len__(self):
		if self.max_samples is not None:
			return self.max_samples
		else:
			return len(self.dataset)


class DataBuffer(object):

	def __init__(self, config, dataset):
		self.config = config
		self.buffer_size = config['buffer_size']
		self.buffer = []
		self.dataset = dataset

	def push(self, x: SparseTensorWrapper, s: SparseTensorWrapper, y: SparseTensorWrapper, phases: List[Phase]):
		for batch_idx in range(len(phases)):
			s_feat, s_coord = s.feats_and_coords_at(batch_idx)
			phase = phases[batch_idx]
			if phase.finished:
				if self.config.get('auto_reload'):
					y_name, seq_key = y.names[batch_idx].split('/')
					data_seq = self.dataset.data_seqs[seq_key]
					y_idx = data_seq.index(y_name)
					# check if the y_name is the last order
					if y_idx == len(data_seq) - 1:
						last_equi_phase = self.config.get('last_equilibrium_max_phase')

						# termination conditions
						if phase.phase >= phase.max_phase:
							continue
						if last_equi_phase is None:
							continue
						if last_equi_phase < phases[batch_idx].equilibrium_phase:
							continue
						else:
							y_feat, y_coord = y.feats_and_coords_at(batch_idx)
					else:
						next_name = data_seq[y_idx + 1]
						y_coord = self.dataset.data[next_name]
						y_feat = torch.ones(y_coord.shape[0], 1)

						if self.config.get('random_rotate'):
							y_coord = random_rotate(y_coord)

						y_coord, y_feat = quantize_data(
							y_coord, y_feat,
							voxel_size=self.config['voxel_size'],
							quantize=self.config.get('quantize'),
							coord_jitter=self.config.get('coord_jitter_y')
						)

						# change phase and names
						phases[batch_idx].reset_equilibrium_mode()
						y.names[batch_idx] = next_name + '/' + seq_key
				else:
					continue
			else:
				y_feat, y_coord = y.feats_and_coords_at(batch_idx)
			x_feat, x_coord = x.feats_and_coords_at(batch_idx)
			self.buffer.append((
				x_feat.cpu(), x_coord.cpu(),
				s_feat.cpu(), s_coord.cpu(),
				y_feat.cpu(), y_coord.cpu(),
				phases[batch_idx], x.names[batch_idx]
			))
		del x, s, y

	def sample(self, batch_size):
		assert len(self.buffer) == self.buffer_size
		x_coords, x_feats = [], []
		s_coords, s_feats = [], []
		y_coords, y_feats = [], []
		phase_list = []
		names = []

		for batch_cnt in range(batch_size):
			idx = random.randint(0, len(self.buffer) - 1)
			sample = self.buffer.pop(idx)
			x_feats.append(sample[0])
			x_coords.append(sample[1])
			s_feats.append(sample[2])
			s_coords.append(sample[3])
			y_feats.append(sample[4])
			y_coords.append(sample[5])
			phase_list.append(sample[6])
			names.append(sample[7])
		return SparseTensorWrapper(x_feats, x_coords, names=names), \
			   SparseTensorWrapper(s_feats, s_coords, names=names), \
			   SparseTensorWrapper(y_feats, y_coords, names=names), phase_list

	def is_full(self):
		return len(self.buffer) == self.buffer_size


class DataScheduler(Iterator):
	def __init__(self, config, testset=False):
		self.config = config
		self.dataset = DATASET[self.config['dataset']](config, mode='train')
		self.eval_dataset = DATASET[self.config['dataset']](config, mode='val')
		self.test_dataset = None if config.get('test_root') is None \
			else DATASET[self.config['dataset']](config, mode='test')
		self.total_epoch = self.config['epoch']
		self.step_cnt = 0
		self.epoch_cnt = 0
		self._remainder = len(self.dataset)
		self.sampler = InfiniteRandomSampler(
			self.dataset,
			max_samples=int(self.config['epoch'] * len(self.dataset))
		)
		self.data_loader = iter(DataLoader(
			self.dataset,
			batch_size=self.config['batch_size'],
			num_workers=self.config['num_workers'],
			collate_fn=self.dataset.collate_fn,
			sampler=self.sampler,
			drop_last=True,
		))
		self.data_buffer = DataBuffer(config, self.dataset)
		self.single_data_loader = iter(DataLoader(
			self.dataset,
			batch_size=1,
			num_workers=self.config['num_workers'],
			collate_fn=self.dataset.collate_fn,
			sampler=self.sampler,
			drop_last=True
		))
		self._check_vis = {}

		if testset:
			# add configs for old configs not containing these
			if config.get('test_root') is None:
				config['test_root'] = './data/pcn_shapenet/test'
			config['test_dataset'] = 'incomplete_shapenet_test_all'
			self.eval_dataset = DATASET[config['test_dataset']](config)

	def __next__(self):
		if self.data_loader is None:
			raise StopIteration
		while self.data_buffer.is_full() is False:
			data = next(self.single_data_loader)
			self.data_buffer.push(
				data[0], data[0], data[1],  # x, s, y = x, x, y
				[Phase(self.config['max_phase'], self.config['equilibrium_max_phase'])]
			)
			self._remainder -= 1
			if self._remainder < 1:
				self._remainder = len(self.dataset)
				self.epoch_cnt += 1

		self.step_cnt += 1
		x, s, y, phase = self.data_buffer.sample(self.config['batch_size'])
		# Get next data
		return x, s, y, phase, self.epoch_cnt

	def __len__(self):
		return len(self.sampler)

	def check_eval_step(self, step):
		return ((step + 1) % self.config['eval_step'] == 0) \
			   or self.config['debug_eval']

	def check_test_step(self, step):
		return (step + 1) % self.config['test_step'] == 0 \
			if self.config.get('test_step') is not None else False

	def check_vis_step(self, step):
		vis = False
		vis_config = self.config['vis']
		for (k, v) in vis_config.items():
			# check if valid visualization config
			if not isinstance(v, dict):
				continue
			if ((step + 1) % v['step'] == 0) or (self.config['debug_vis']):
				self._check_vis[k] = True
				vis = True
			else:
				self._check_vis[k] = False
		return vis

	def check_summary_step(self, step):
		return (step + 1) % self.config['summary_step'] == 0

	def check_empty_cache_step(self, step):
		empty_cache_step = self.config.get('empty_cache_step')
		if empty_cache_step is None:
			empty_cache_step = 10
		return (step + 1) % empty_cache_step == 0

	def eval(self, model, writer, step):
		self.eval_dataset.eval(model, writer, step)

	def test(self, model, writer, step, save_vis=True, save_tensor=True):
		if self.test_dataset is not None:
			self.test_dataset.test(model, writer, step, save_vis, save_tensor)

	def visualize_test(self, model, writer, step):
		self.test_dataset.visualize_test(model, writer, step)

	def visualize(self, model, writer, step, skip_eval=False):

		# find options to visualize in this step
		options = []
		for (k, v) in self._check_vis.items():
			if not v:
				continue
			else:
				options.append(k)

		if isinstance(self.config['overfit_one_ex'], int) or skip_eval:
			self.dataset.visualize(model, options, writer, step)
		else:
			self.dataset.visualize(model, options, writer, step)  # train dataset
			self.eval_dataset.visualize(model, options, writer, step)  # eval dataset
		# reset _check_vis
		self._check_vis = {}


class BaseDataset(Dataset, ABC):
	name = 'base'

	def __init__(self, config, mode: str = 'train'):
		self.config = config
		self.mode = mode
		self.device = config['device']
		self.data_dim = config['data_dim']
		self.rot_mats = get_unique_rot_mats()

	'''
	Note that dataset's __getitem__() returns (x_coord, x_feat, y_coord, y_feat, name)
	But the collated batch returns type of (SparseTensorWrapper, SparseTensorWrapper)
	'''

	def __getitem__(self, idx) \
			-> (torch.tensor, torch.tensor, torch.tensor, torch.tensor, List[str]):
		# sparse tensor and tensor should have equal size
		raise NotImplemented

	def prepare_sparse_tensor(self, x_coord: torch.tensor, y_coord: torch.tensor, idx=None) \
			-> (torch.tensor,) * 4:
		'''
		Takes raw coordinates as input, and
			1. apply transform for data augmentation
			2. quantize the coordinates to integer (required for sparse tensor)
			3. add features for sparse tensor

		Args:
			x_coord: tensor of N x data_dim with unprocessed incomplete shape
			y_coord: tensor of M x data_dim with unprocessed complete shape
		Returns:
			x_coord: tensor of N x data_dim with processed incomplete shape
			x_feat: tensor of N x 1
			y_coord: tensor of N x data_dim with processed incomplete shape
			y_feat: tensor of N x 1
		'''
		x_feat = torch.ones(x_coord.shape[0], 1)
		y_feat = torch.ones(y_coord.shape[0], 1)

		# create sparse tensor
		if self.mode == 'train':
			len_x = x_coord.shape[0]
			coord = torch.cat([x_coord, y_coord], dim=0)
			x_coord, y_coord = coord[:len_x, :], coord[len_x:, :]

		coord_jitter_x = self.config.get('coord_jitter_x') if self.mode == 'train' else False

		if self.mode == 'train':
			if self.config.get('random_rotate'):
				x_coord = random_rotate(x_coord)
				y_coord = random_rotate(y_coord)
		elif idx is not None:
			rot_mat = self.rot_mats[idx]
			x_coord = x_coord @ rot_mat
			y_coord = y_coord @ rot_mat

		truncate = self.config.get('truncate_data')
		if truncate is not None:
			if truncate['max'] is not None:
				x_coord = x_coord[x_coord[:, 0] <= truncate['max'], :]
				x_coord = x_coord[x_coord[:, 1] <= truncate['max'], :]
				x_coord = x_coord[x_coord[:, 2] <= truncate['max'], :]
			if truncate['min'] is not None:
				x_coord = x_coord[x_coord[:, 0] >= truncate['min'], :]
				x_coord = x_coord[x_coord[:, 1] >= truncate['min'], :]
				x_coord = x_coord[x_coord[:, 2] >= truncate['min'], :]

		x_coord, x_feat = quantize_data(
			x_coord, x_feat,
			voxel_size=self.config['voxel_size'],
			quantize=self.config.get('quantize'),
			coord_jitter=coord_jitter_x
		)
		coord_jitter_y = self.config.get('coord_jitter_y') if self.mode == 'train' else False
		y_coord, y_feat = quantize_data(
			y_coord, y_feat,
			voxel_size=self.config['voxel_size'],
			quantize=self.config.get('quantize'),
			coord_jitter=coord_jitter_y
		)

		return x_coord, x_feat, y_coord, y_feat

	def collate_fn(self, batch) -> (SparseTensorWrapper, SparseTensorWrapper):
		'''
		Args:
			batch: List of (torch.tensor, torch.tensor, torch.tensor, torch.tensor, str)
		Returns:
			x: incomplete shape
			y: complete shape
		'''
		x_coords, x_feats, y_coords, y_feats, names = list(zip(*batch))
		x = SparseTensorWrapper(x_feats, x_coords, names=names)
		y = SparseTensorWrapper(y_feats, y_coords, names=names)
		return x, y

	def collate_fn_generative(self, batch):
		x_coords, x_feats, y_coords, y_feats, ref_set = list(zip(*batch))
		x = SparseTensorWrapper(x_feats, x_coords)
		y = SparseTensorWrapper(y_feats, y_coords)
		return x, y, ref_set

	def eval(self, model: Model, writer: SummaryWriter, step):
		training = model.training
		model.eval()
		data_loader = DataLoader(
			self,
			batch_size=self.config['eval_batch_size'],
			num_workers=self.config['num_workers'],
			collate_fn=self.collate_fn,
			drop_last=True,
		)

		print('')
		eval_losses = []
		for eval_step, data in enumerate(data_loader):
			x, y = data[0], data[1]
			eval_loss = model.evaluate(x, y, step)
			eval_losses.append(eval_loss)

			print('\r[Evaluating, Step {:7}, Loss {:5}]'.format(
				eval_step, '%.3f' % eval_loss), end=''
			)

		print('')
		model.write_dict_summaries(step)
		model.train(training)

	def test(self, model: GCA, writer: SummaryWriter, step):
		raise NotImplementedError()

	def visualize(self, model: GCA, options: List, writer: SummaryWriter, step):
		training = model.training
		model.eval()

		# fix vis_indices
		vis_indices = self.config['vis']['indices']
		if isinstance(vis_indices, int):
			# sample data points from n data points with equal interval
			n = len(self)
			vis_indices = torch.linspace(0, n - 1, vis_indices).int()
			vis_indices = torch.unique(vis_indices).tolist()

		# override to the index when in overfitting debug mode
		if isinstance(self.config['overfit_one_ex'], int):
			vis_indices = torch.tensor([self.config['overfit_one_ex']])
		for option in options:
			if option == 'imgs':
				model.visualize_imgs(self, vis_indices, step)
			elif (option == 'all_imgs') and (self.mode != 'train'):
				model.visualize_all_imgs(self, step)
			elif option == 'debug':
				model.visualize_debug(self, vis_indices, step)
			elif option == 'heatmap':
				model.visualize_heatmap(self, step)
			elif option == 'heatmap_3d':
				model.visualize_heatmap_3d(self, step)
			else:
				raise ValueError
		model.train(training)

	def visualize_test(self, model: GCA, writer: SummaryWriter, step):
		training = model.training
		model.eval()

		# fix vis_indices
		vis_indices = self.config['vis']['indices']
		if isinstance(vis_indices, int):
			# sample data points from n data points with equal interval
			n = len(self)
			vis_indices = torch.linspace(0, n - 1, vis_indices).int().tolist()

		# override to the index when in overfitting debug mode
		if isinstance(self.config['overfit_one_ex'], int):
			vis_indices = torch.tensor([self.config['overfit_one_ex']])

		model.visualize_test(self, vis_indices, step)
		model.train(training)


# =================
# Concrete Datasets
# =================
class Gold(BaseDataset):
	name = 'gold'

	def __init__(self, config, mode: str):
		BaseDataset.__init__(self, config, mode)
		self.data_root = config['root']

		# load data
		self.data = {}
		file_paths = list(sorted(glob.glob(os.path.join(self.data_root, '*.ply'))))
		for file_path in file_paths:
			data = torch.tensor(pcu.load_mesh_v(file_path)).float()
			# data = torch.tensor(np.load(file_path, allow_pickle=True)).float()
			data_name = os.path.basename(file_path).replace('.ply', '')
			self.data[data_name] = data

		# add small_sphere to data
		radius = config.get('small_sphere_radius') if config.get('small_sphere_radius') else 25
		coord1, coord2 = torch.meshgrid([
			torch.arange(-radius, radius + 1).float(),
			torch.arange(-radius, radius + 1).float()
		])
		xy_coords = torch.stack([
			coord1, coord2,
		], dim=2).view(-1, 2)  # N x 2
		in_sphere_idxs = torch.sum(xy_coords ** 2, dim=1) <= (radius ** 2)
		xy_coords = xy_coords[in_sphere_idxs, :]
		z_coords = torch.sqrt(
			radius ** 2 - torch.sum(xy_coords ** 2, dim=1)
		).view(-1)
		plus_coords = torch.stack([
			xy_coords[:, 0], z_coords, xy_coords[:, 1]
		], dim=1)
		minus_coords = torch.stack([
			xy_coords[:, 0], -z_coords, xy_coords[:, 1]
		], dim=1)
		self.data['small_sphere'] = torch.cat([plus_coords, minus_coords], dim=0)

		if config.get('auto_reload'):
			data_seqs = {
				'train-1': ['train1_1', 'train1_2', 'train1_3'],
				'train-1-wo-middle': ['train1_1', 'train1_3'],
				'RD-1': ['growth1_01', 'growth1_01'],
				'Cube-1': ['growth1_02', 'growth1_02'],
				'RC-1': ['growth1_03', 'growth1_03'],
				'CRD-1': ['growth1_04', 'growth1_04'],
				'HOH-1': ['growth1_05', 'growth1_05'],
				'H3_inter-1': ['growth1_06', 'growth1_06'],
				'H3_inter_D-1': ['growth1_07', 'growth1_07'],
				'H3_inter2-1': ['growth1_08', 'growth1_08'],
				'H3_inter2_D-1': ['growth1_09', 'growth1_09'],
				'H1_inter-1': ['growth1_10', 'growth1_10'],
				'H1_inter_D-1': ['growth1_11', 'growth1_11'],
				'H1_inter2-1': ['growth1_12', 'growth1_12'],
				'H1_inter2_D-1': ['growth1_13', 'growth1_13'],
				'H2-1': ['growth1_14', 'growth1_14'],
				'H2_D-1': ['growth1_15', 'growth1_15'],

				'train-2': ['train2_1', 'train2_2', 'train2_3'],
				'train-2-wo-middle': ['train2_1', 'train2_3'],
				'RD-2': ['growth2_01', 'growth2_01'],
				'Cube-2': ['growth2_02', 'growth2_02'],
				'RC-2': ['growth2_03', 'growth2_03'],
				'CRD-2': ['growth2_04', 'growth2_04'],
				'HOH-2': ['growth2_05', 'growth2_05'],
				'H3_inter-2': ['growth2_06', 'growth2_06'],
				'H3_inter_D-2': ['growth2_07', 'growth2_07'],
				'H3_inter2-2': ['growth2_08', 'growth2_08'],
				'H3_inter2_D-2': ['growth2_09', 'growth2_09'],
				'H1_inter-2': ['growth2_10', 'growth2_10'],
				'H1_inter_D-2': ['growth2_11', 'growth2_11'],
				'H1_inter2-2': ['growth2_12', 'growth2_12'],
				'H1_inter2_D-2': ['growth2_13', 'growth2_13'],
				'H2-2': ['growth2_14', 'growth2_14'],
				'H2_D-2': ['growth2_15', 'growth2_15'],

				'HI3': ['HI3', 'HI3'],
				'HI4': ['HI4', 'HI4'],
				'HI5': ['HI5', 'HI5'],
				'HI6': ['HI6', 'HI6'],
				'HI7': ['HI7', 'HI7'],

				'RDR2': ['RDR2', 'RDR2'],
				'CBR2': ['CBR2', 'CBR2'],

			}

			self.size_str = '-size'
			config_data_seqs = config['data_seqs'] \
				if self.mode == 'train' else config['val_data_seqs']
			trimmed_seqs = [
				seq_name.split(self.size_str)[0]
				if self.size_str in seq_name else seq_name
				for seq_name in config_data_seqs
			]
			self.data_seqs = {
				seq_name: data_seqs[seq_name]
				for seq_name in trimmed_seqs
			}

			# convert the end of validation seqs to end of train seqs
			if self.mode != 'train':
				train_last_name = data_seqs[self.config['data_seqs'][0]][-1]
				for k in self.data_seqs.keys():
					self.data_seqs[k][-1] = train_last_name
			self.seq_keys = sorted(config_data_seqs)

		# rotate if we are visualizing the heatmap
		self.is_crop = (self.seq_keys[0] in ['H3_inter_train', 'HOH2_train', 'H2']) \
		               and (len(self.seq_keys) == 1)
		self.is_crop = self.is_crop | (self.config['vis']['heatmap_3d']['step'] == 1)

	def __getitem__(self, idx):
		if self.is_crop:
			seq_key = self.seq_keys[0]
		else:
			seq_key = self.seq_keys[idx]
		size = 1.
		if self.size_str in seq_key:
			seq_keys = seq_key.split(self.size_str)
			seq_key = seq_keys[0]
			size = float(seq_keys[-1])
			data_seq = self.data_seqs[seq_key]
			seq_key += '-size={}'.format(seq_keys[-1])
		else:
			data_seq = self.data_seqs[seq_key]
		x_coord = self.data[data_seq[0]] * size
		y_coord = self.data[data_seq[1]]
		# if self.is_crop:
		# 	return (*self.prepare_sparse_tensor(x_coord, y_coord, idx), seq_key)  # need to rotate
		return (*self.prepare_sparse_tensor(x_coord, y_coord), seq_key)


	def collate_fn(self, batch) -> (SparseTensorWrapper, SparseTensorWrapper):
		'''
		Args:
			batch: List of (torch.tensor, torch.tensor, torch.tensor, torch.tensor, str)
		Returns:
			x: incomplete shape
			y: complete shape
		'''
		x_coords, x_feats, y_coords, y_feats, names = list(zip(*batch))
		x_names = [
			self.data_seqs[seq_key][0] + '/' + seq_key
			for seq_key in names
		]
		y_names = [
			self.data_seqs[seq_key][1] + '/' + seq_key
			for seq_key in names
		]
		x = SparseTensorWrapper(x_feats, x_coords, names=x_names)
		y = SparseTensorWrapper(y_feats, y_coords, names=y_names)
		return x, y

	def __len__(self):
		return len(self.seq_keys)


DATASET = {
	Gold.name: Gold,
}
