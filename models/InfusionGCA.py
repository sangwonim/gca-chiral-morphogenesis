import time
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from typing import List
from models.gca import GCA
from utils.pad import unpack
from utils.util import timeit
from utils.scheduler import InfusionScheduler
from utils.visualization import sparse_tensors2tensor_imgs, save_tensor_img, plt_to_tensor
from utils.phase import Phase
from utils.sparse_tensor import SparseTensorWrapper
from pyntcloud.io.ply import write_ply
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

import pandas as pd


def save_ply(coords, file_name):
	coords = pd.DataFrame({
		'x': coords[:, 0],
		'y': coords[:, 1],
		'z': coords[:, 2],
	})
	write_ply(filename=file_name, points=coords)


class InfusionGCA(GCA):
	name = 'infusion_gca'

	def __init__(self, config, writer: SummaryWriter):
		GCA.__init__(self, config, writer)
		self.input_format = self.config['input_format']
		self.infusion_scheduler = InfusionScheduler(config)
		self.bce_loss = torch.nn.BCEWithLogitsLoss()

	@timeit
	def forward(self, x):
		'''
		Forward pass through sparse convolution network
		and unpack the output

		input:
			x: SparseTensorWrapper of
				coordinates with shape N x 3
				features with shape N x 1
		output:
			x_hat: SparseTensorWrapper of
				coordinates with shape M x 3
				features with shape M x k (parameter outputs)
		'''
		if self.config.get('coord_feats'):
			new_feats = x.coords.float()[:, :3] * self.config['voxel_size']
			x = SparseTensorWrapper(new_feats, x.coords, collated=True)
		out_packed = self.backbone(x.sparse_tensor)
		out_packed = SparseTensorWrapper.from_sparse_tensors(out_packed)
		out_unpacked = unpack(out_packed, self.shifts, self.out_dim)
		return out_unpacked

	@timeit
	def learn(
			self, x: SparseTensorWrapper, s: SparseTensorWrapper, y: SparseTensorWrapper,
			step: float, phases: List[Phase], mode: str = 'train'
	) -> (SparseTensorWrapper, List, float):
		'''
		Args:
			x: Incomplete sparse tensor (Given)
			s: Current state of sparse tensor for the neural network to forward
			y: Complete sparse tensor for the neural network to compute loss
			step: Current training step. Required for the summary writer
			phases: Phase information for the sparse tensors
			mode: The mode of learning
				if 'train', returns tuple of
					x_next: SparseTensorWrapper, phases: List[Phase], loss: float,
				else (either 'eval' or 'eval_no_gt')...., returns tuple of
					x_next: SparseTensorWrapper, y_pad: SparseTensorWrapper, loss: float
					where x_next is the predicted next set of voxels and y_pad is subset of ground truth voxels
		'''
		# forward pass
		s, y = s.to(self.device), y.to(self.device)

		# currently use get for backward compatibilty
		s = self.append_input(x, s) if self.config.get('append_input') else s
		s_hat = self.forward(s)

		infusion_rates = self.infusion_scheduler.sample(phases)
		losses = []
		s_next_coords = []
		s_next_feats = []

		for batch_idx, (infusion_rate, phase) in enumerate(zip(infusion_rates, phases)):
			s_hat_coord = s_hat.coords_at(batch_idx)
			s_hat_feat = s_hat.feats_at(batch_idx)
			y_coord = y.coords_at(batch_idx)

			with torch.no_grad():
				dists, _, _, s_hat_y_idx = self._chamfer_dist(
					s_hat_coord.unsqueeze(0).float().to(self.device),
					y_coord.unsqueeze(0).float().to(self.device),
					return_idx=True
				)
			s_hat_y_idx = torch.unique(s_hat_y_idx.squeeze(0)).long()
			s_hat_y = torch.zeros(s_hat_coord.shape[0]).to(self.device)
			s_hat_y[s_hat_y_idx] = 1

			# infusion training
			# rel_dist = (dists.squeeze(0) - dists.min()) / (dists.max() - dists.min())
			# infusion_rate = torch.clamp(infusion_rate + self.dist_infusion_speed * rel_dist, max=1.)
			s_next_prob = infusion_rate * s_hat_y \
				+ (1. - infusion_rate) * torch.sigmoid(s_hat_feat).squeeze(1)
			s_next_feat = torch.bernoulli(s_next_prob)

			# compute loss
			losses.append(self.bce_loss(s_hat_feat.squeeze(1), s_next_feat.float()))
			s_next_coords.append(s_hat_coord[s_next_feat.bool()])
			s_next_feats.append(torch.ones(s_next_coords[batch_idx].shape[0], 1))

			# update_phases
			phases[batch_idx] = phase + 1
			if self.config['early_pop']:
				completion_rate = (dists == 0).sum().item() \
								  / float(y.coords_at(batch_idx).shape[0])
				if completion_rate >= self.config['completion_rate']:
					if not phase.equilibrium_mode:
						phase.set_complete()
						self.list_summaries['completion_phase/{}'.format(mode)] += [phase.phase]

				elif (phase.phase > self.config['max_phase']) and (mode == 'train'):
					incomplete_key = 'phase/incomplete_cnt'
					self.scalar_summaries[incomplete_key] = [self.scalar_summaries[incomplete_key][0] + 1] if \
						len(self.scalar_summaries[incomplete_key]) != 0 else [1]

		loss = torch.stack(losses).mean()
		s_next = SparseTensorWrapper(s_next_feats, s_next_coords, device='cpu')

		# write summaries
		batch_size = len(phases)
		self.scalar_summaries['loss/{}/total'.format(mode)] += [loss.item()]
		self.list_summaries['loss/{}/total_histogram'.format(mode)] += torch.stack(losses).cpu().tolist()
		self.scalar_summaries['num_points/input'] += [s.coords_at(i).shape[0] for i in range(batch_size)]
		self.scalar_summaries['num_points/output'] += [s_next_coords[i].shape[0] for i in range(batch_size)]
		self.list_summaries['scheduler/infusion_rates'] += infusion_rates

		if mode != 'train':
			return s_next, s_next, loss.item()

		# take gradient descent
		self.zero_grad()
		loss.backward()
		self.clip_grad()
		self.optimizer.step()
		self.lr_scheduler.step()

		return s_next, phases, loss.item()

	def append_input(self, x: SparseTensorWrapper, s: SparseTensorWrapper):
		x_coords = x.coords.to(self.device)
		s_coords = s.coords.to(self.device)
		new_coords = torch.unique(torch.cat([x_coords, s_coords], dim=0), dim=0)
		new_feats = torch.ones(new_coords.shape[0], 1)
		return SparseTensorWrapper(new_feats, new_coords, collated=True, device=s.device)

	def visualize_imgs(
			self, dataset, vis_indices: List, step
	):
		training = self.training
		self.eval()
		img_config = self.config['vis']['imgs']
		max_phase = self.config['max_eval_phase']
		eval_batch_size = self.config['eval_batch_size']
		mini_batches = [
			vis_indices[i: i + eval_batch_size]
			for i in range(0, len(vis_indices), self.config['eval_batch_size'])
		]

		for i, mini_batch_idxs in enumerate(mini_batches):
			batch = [dataset[i] for i in mini_batch_idxs]
			batch_size = len(batch)

			x_coords, x_feats, y_coords, y_feats, names = list(zip(*batch))
			x = SparseTensorWrapper(x_feats, x_coords)
			y = SparseTensorWrapper(y_feats, y_coords)

			input_imgs = sparse_tensors2tensor_imgs(
				x, data_dim=self.config['data_dim'],
				img_config=img_config, batch_size=batch_size,
			)  # list of tensor of C x H x W
			gt_imgs = sparse_tensors2tensor_imgs(
				y, data_dim=self.config['data_dim'],
				img_config=img_config, batch_size=batch_size,
			)  # list of tensor of C x H x W
			debug_save_dir = os.path.join(self.config['log_dir'], 'debug_imgs', 'step-{}'.format(step))
			os.makedirs(debug_save_dir, exist_ok=True)

			chamfer_dists = defaultdict(list)

			try:
				for trial in range(img_config['trials']):
					pred_imgs_list = []
					s = x
					for p in range(max_phase):
						s = self.append_input(x, s) if self.config.get('append_input') else s

						if img_config.get('save_pt'):
							for batch_idx in range(batch_size):
								torch.save(
									s.coords_at(batch_idx).cpu(),
									'{}/{}_{}_{}.pt'.format(
										debug_save_dir, names[batch_idx], trial, p
									)
								)

						if img_config.get('save_ply'):
							for batch_idx in range(batch_size):
								file_name = '{}/{}_{}_{}.ply'.format(
									debug_save_dir, names[batch_idx], trial, p
								)
								save_ply(
									s.coords_at(batch_idx).cpu().float().numpy(),
									file_name
								)

						with torch.no_grad():
							s = self.transition(s)

							for batch_idx in range(batch_size):
								s_coord = s.coords_at(batch_idx).unsqueeze(0).float()
								y_coord = y.coords_at(batch_idx).unsqueeze(0).float().to(self.device)
								with torch.no_grad():
									dist1, dist2 = self._chamfer_dist(s_coord, y_coord)
								dist = ((dist1.mean() + dist2.mean()) / (150. / self.config['voxel_size'])) / 2
								chamfer_dists['cd-{}/step={}?{}'.format(names[batch_idx], step, p)] += [dist.cpu().item()]

								if p == (max_phase - 1):
									self.writer.add_scalar(
										'final_cd/{}'.format(names[batch_idx]),
										dist.cpu().item(), step
									)

						if (p % img_config['phase_interval'] == 0) or (p == (max_phase - 1)):
							imgs = sparse_tensors2tensor_imgs(
								s, data_dim=self.config['data_dim'],
								img_config=img_config, batch_size=batch_size,
							)
							pred_imgs_list.append(imgs)
							if p == (max_phase - 1):
								for batch_idx in range(batch_size):
									save_tensor_img(
										imgs[batch_idx],
										'{}/{}_{}_{}.png'.format(
											debug_save_dir, names[batch_idx], trial, p
										)
									)

					if img_config.get('save_pt'):
						for batch_idx in range(batch_size):
							torch.save(
								s.coords_at(batch_idx).cpu(),
								'{}/{}_{}_{}.pt'.format(
									debug_save_dir, names[batch_idx], trial, max_phase
								)
							)

					if img_config.get('save_ply'):
						for batch_idx in range(batch_size):
							file_name = '{}/{}_{}_{}.ply'.format(
								debug_save_dir, names[batch_idx], trial, max_phase
							)
							save_ply(
								s.coords_at(batch_idx).cpu().float().numpy(),
								file_name
							)

					for batch_idx, pred_imgs in enumerate(zip(*pred_imgs_list)):
						num_frames = len(pred_imgs)
						pred_imgs = torch.stack(list(pred_imgs), dim=0)  # tensor of T x C x H x W
						input_img = torch.stack([input_imgs[batch_idx]] * num_frames)  # tensor of T x C x H x W
						gt_img = torch.stack([gt_imgs[batch_idx]] * num_frames)  # tensor of T x C x H x W
						vid_tensor = torch.cat(
							[input_img, pred_imgs, gt_img],
							dim=3
						).unsqueeze(0)  # tensor of 1 x T x C x H x W

						self.writer.add_video(
							'{}-img-{}/trial{}'.format(
								dataset.mode, names[batch_idx], trial
							), vid_tensor, step
						)

				for k, v in chamfer_dists.items():
					avg_dist = sum(v) / len(v)
					k = k.split('?')
					tag, phase = k[0], int(k[1])
					self.writer.add_scalar(tag, avg_dist, phase)
			except RuntimeError:
				pass
		self.train(training)

	def visualize_debug(self, dataset, vis_indices, step):
		img_config = self.config['vis']['debug']
		max_phase = self.config['max_eval_phase']
		eval_batch_size = self.config['eval_batch_size']
		mini_batches = [
			vis_indices[i: i + eval_batch_size]
			for i in range(0, len(vis_indices), self.config['eval_batch_size'])
		]

		for i, mini_batch_idxs in enumerate(mini_batches):
			batch = [dataset[i] for i in mini_batch_idxs]
			batch_size = len(batch)

			x_coords, x_feats, y_coords, y_feats, names = list(zip(*batch))
			x = SparseTensorWrapper(x_feats, x_coords).to(self.device)
			y = SparseTensorWrapper(y_feats, y_coords).to(self.device)

			input_imgs = sparse_tensors2tensor_imgs(
				x, data_dim=self.config['data_dim'],
				img_config=img_config, batch_size=batch_size,
			)  # tensor of C x H x W
			gt_imgs = sparse_tensors2tensor_imgs(
				y, data_dim=self.config['data_dim'],
				img_config=img_config, batch_size=batch_size,
			)  # tensor of C x H x W

			sequential_imgs_list = []  # appends input, pred, gt imgs

			for trial in range(img_config['trials']):
				# create place holder phases
				try:
					s = x
					phases = [Phase(0, 0)] * batch_size
					for p in range(max_phase):
						s = s.to(self.device)
						# no overflow should occur
						with torch.no_grad():
							s_next, y_pad, loss = self.learn(x, s, y, step, phases, mode='vis_debug')

						if (p % img_config['phase_interval'] == 0) or (p == (max_phase - 1)):
							s_imgs = sparse_tensors2tensor_imgs(
								s, data_dim=self.config['data_dim'],
								img_config=img_config, batch_size=batch_size,
							)
							s_next_imgs = sparse_tensors2tensor_imgs(
								s_next, data_dim=self.config['data_dim'],
								img_config=img_config, batch_size=batch_size,
							)
							y_pad_imgs = sparse_tensors2tensor_imgs(
								y_pad, data_dim=self.config['data_dim'],
								img_config=img_config, batch_size=batch_size,
							)
							batch_img_list = []
							for s_img, s_next_img, y_pad_img in zip(s_imgs, s_next_imgs, y_pad_imgs):
								batch_img_list.append(torch.cat([s_img, s_next_img, y_pad_img], dim=2))
							sequential_imgs_list.append(batch_img_list)
						s = s_next

					for batch_idx, sequential_img in enumerate(zip(*sequential_imgs_list)):
						num_frames = len(sequential_img)
						sequential_img = torch.stack(list(sequential_img), dim=0)  # tensor of T x C x H x (n * W)
						input_img = torch.stack([input_imgs[batch_idx]] * num_frames)  # tensor of T x C x H x W
						gt_img = torch.stack([gt_imgs[batch_idx]] * num_frames)  # tensor of T x C x H x W
						# visualize input_img, input_img, sequential_img, output_img(y_pad), gt_img
						vid_tensor = torch.cat(
							[input_img, sequential_img, gt_img],
							dim=3
						).unsqueeze(0)  # tensor of 1 x T x C x H x W

						self.writer.add_video(
							'debug-{}-img-{}/trial{}'.format(
								dataset.mode, names[batch_idx], trial
							), vid_tensor, step
						)
				except RuntimeError:
					pass

	def _get_heatmap_crops(self, coord, img_config):
		# get crop centers
		heatmap_len = img_config['heatmap_len']
		grid_type = img_config['grid_type']
		if grid_type == 'cube':
			grid = np.meshgrid(
				np.linspace(-1, 1, heatmap_len, endpoint=True),
				np.linspace(-1, 1, heatmap_len, endpoint=True),
				np.linspace(-1, 1, heatmap_len, endpoint=True),
			)
			grid = np.stack(grid, axis=-1).reshape(-1, 3).astype(np.float32)  # N X 3
			grid_surface = np.min(np.abs(np.abs(grid) - 1), axis=1) < 1e-4
			grid = grid[grid_surface]
		elif grid_type == 'geodesic':
			grid = np.meshgrid(
				np.linspace(0, 2 * np.pi, heatmap_len, endpoint=False),
				np.linspace(0, 2 * np.pi, heatmap_len, endpoint=False),
			)
			grid = np.stack(grid, axis=-1).reshape(-1, 2).astype(np.float32)  # N X 2
			grid = np.stack([
				np.sin(grid[:, 0]) * np.cos(grid[:, 1]),
				np.sin(grid[:, 0]) * np.sin(grid[:, 1]),
				np.cos(grid[:, 0])
			], axis=1)
		elif grid_type == 'icosahedron':
			grid = self.icosahedron2sphere(heatmap_len)[0]
		else:
			raise ValueError(f'grid {grid_type} not allowed')
		grid = torch.tensor(grid)

		# import point_cloud_utils as pcu
		# pcu.save_mesh_v('grid.ply', grid.numpy())
		# breakpoint()

		cos_sim_thresh = torch.tensor(np.cos(np.radians(img_config['crop_len'])))  # degrees
		crops = []
		for center in grid:
			# compute cosine distance btw center and coords
			cos_sim = (torch.sum(center * coord, dim=1) /
			            (torch.linalg.norm(center) * torch.linalg.norm(coord.float(), dim=1)))
			crops.append(coord[cos_sim > cos_sim_thresh])

		return crops

	def icosahedron2sphere(self, level):
		# this function use a icosahedron to sample uniformly on a sphere
		a = 2 / (1 + np.sqrt(5))
		M = np.array([
			0, a, -1, a, 1, 0, -a, 1, 0,
			0, a, 1, -a, 1, 0, a, 1, 0,
			0, a, 1, 0, -a, 1, -1, 0, a,
			0, a, 1, 1, 0, a, 0, -a, 1,
			0, a, -1, 0, -a, -1, 1, 0, -a,
			0, a, -1, -1, 0, -a, 0, -a, -1,
			0, -a, 1, a, -1, 0, -a, -1, 0,
			0, -a, -1, -a, -1, 0, a, -1, 0,
			-a, 1, 0, -1, 0, a, -1, 0, -a,
			-a, -1, 0, -1, 0, -a, -1, 0, a,
			a, 1, 0, 1, 0, -a, 1, 0, a,
			a, -1, 0, 1, 0, a, 1, 0, -a,
			0, a, 1, -1, 0, a, -a, 1, 0,
			0, a, 1, a, 1, 0, 1, 0, a,
			0, a, -1, -a, 1, 0, -1, 0, -a,
			0, a, -1, 1, 0, -a, a, 1, 0,
			0, -a, -1, -1, 0, -a, -a, -1, 0,
			0, -a, -1, a, -1, 0, 1, 0, -a,
			0, -a, 1, -a, -1, 0, -1, 0, a,
			0, -a, 1, 1, 0, a, a, -1, 0])
		coor = M.T.reshape(3, 60, order='F').T
		coor, idx = np.unique(coor, return_inverse=True, axis=0)
		tri = idx.reshape(3, 20, order='F').T
		# extrude
		coor = list(coor / np.tile(np.linalg.norm(coor, axis=1, keepdims=True), (1, 3)))
		for _ in range(level):
			triN = []
			for t in range(len(tri)):
				n = len(coor)
				coor.append((coor[tri[t, 0]] + coor[tri[t, 1]]) / 2)
				coor.append((coor[tri[t, 1]] + coor[tri[t, 2]]) / 2)
				coor.append((coor[tri[t, 2]] + coor[tri[t, 0]]) / 2)
				triN.append([n, tri[t, 0], n + 2])
				triN.append([n, tri[t, 1], n + 1])
				triN.append([n + 1, tri[t, 2], n + 2])
				triN.append([n, n + 1, n + 2])
			tri = np.array(triN)
			# uniquefy
			coor, idx = np.unique(coor, return_inverse=True, axis=0)
			tri = idx[tri]
			# extrude
			coor = list(coor / np.tile(np.sqrt(np.sum(coor * coor, 1, keepdims=True)), (1, 3)))
		return np.array(coor), np.array(tri)

	def visualize_heatmap_3d(self, dataset, step):
		training = self.training
		self.eval()
		global hist_abs_imgs
		if dataset.mode == 'train':
			return

		# assumes that eval dataset has only 1 data
		img_config = self.config['vis']['heatmap_3d']
		max_phase = self.config['max_eval_phase']

		data_loader = DataLoader(
			dataset,
			batch_size=1,
			num_workers=0,
			collate_fn=dataset.collate_fn,
			drop_last=False,
		)

		x_union = []
		# used for prev rotation
		for data in data_loader:
			x_union.append(data[0].coords_at(0))
		x_union = torch.unique(torch.cat(x_union, dim=0), dim=0)

		debug_dir = os.path.join(self.config['log_dir'], 'heatmap_figure', 'step-{}'.format(step))
		os.makedirs(debug_dir, exist_ok=True)
		union_coord = x_union.float().cpu().numpy()
		file_name = os.path.join(debug_dir, 'union-input.ply')
		save_ply(union_coord, file_name)

		heatmap_sums = [
			SparseTensorWrapper(
				[torch.zeros(x_union.shape[0], 1)], [x_union],
				device=self.device
			) for _ in range(max_phase + 1)
		]
		heatmap_cnts = [
			SparseTensorWrapper(
				[torch.zeros(x_union.shape[0], 1)], [x_union],
				device=self.device,
				coordinate_manager=heatmap_sums[p].coordinate_manager
			) for p in range(max_phase + 1)
		]

		for data in data_loader:
			x, y = data
			x_original_coord = x.coords_at(0)
			y_coord = y.coords_at(0)
			trials = img_config['trials']

			file_name = os.path.join(debug_dir, 'gt.ply')
			save_ply(y_coord, file_name)
			y = y.to(self.device)

			x_cropped_coords = self._get_heatmap_crops(x_original_coord, img_config)
			y_cropped_coords = self._get_heatmap_crops(y_coord, img_config)

			chamfer_dists = []

			for crop_idx, x_coord in tqdm(enumerate(x_cropped_coords), total=len(x_cropped_coords)):
				x_feat = torch.ones(x_coord.shape[0], 1)
				x = SparseTensorWrapper([x_feat] * trials, [x_coord] * trials, device=self.device)
				img_config_copy = deepcopy(img_config)
				img_config_copy['height'] = 200
				img_config_copy['width'] = 200
				chamfer_dists.append([])
				s = x

				input_coord = s.coords_at(0).float().cpu().numpy()
				file_name = os.path.join(debug_dir, '{}-input.ply'.format(crop_idx))
				save_ply(input_coord, file_name)

				y_cropped_coord = y_cropped_coords[crop_idx].float()
				file_name = os.path.join(debug_dir, '{}-gt.ply'.format(crop_idx))
				save_ply(y_cropped_coord.float().cpu().numpy(), file_name)
				y_cropped_coord = y_cropped_coord.to(self.device).unsqueeze(0)

				for p in range(max_phase + 1):
					chamfer_dists_single_crop = []
					for trial in range(trials):
						s_coord = s.coords_at(trial).unsqueeze(0).float()
						with torch.no_grad():
							dist1, dist2 = self._chamfer_dist(y_cropped_coord, s_coord)
							one_way_chamfer = (dist1.mean() / (150. / self.config['voxel_size'])) / 2
							chamfer_dists_single_crop.append(one_way_chamfer)

							heatmap_single = SparseTensorWrapper(
								[torch.ones(x_coord.shape[0], 1).to(self.device) * one_way_chamfer], [x_coord],
								device=self.device,
								coordinate_manager=heatmap_sums[p].coordinate_manager
							)
							heatmap_sums[p].sparse_tensor = heatmap_sums[p].sparse_tensor + heatmap_single.sparse_tensor

							heatmap_cnt_single = SparseTensorWrapper(
								[torch.ones(x_coord.shape[0], 1).to(self.device)], [x_coord],
								device=self.device,
								coordinate_manager=heatmap_sums[p].coordinate_manager
							)
							heatmap_cnts[p].sparse_tensor = heatmap_cnts[p].sparse_tensor + heatmap_cnt_single.sparse_tensor
					chamfer_dists[crop_idx].append(torch.tensor(chamfer_dists_single_crop).mean().cpu().item())

					if p >= max_phase:
						break
					with torch.no_grad():
						s = self.transition(s)
				torch.cuda.empty_cache()

				output_coord = s.coords_at(0).float().cpu().numpy()
				file_name = os.path.join(debug_dir, '{}-output.ply'.format(crop_idx))
				save_ply(output_coord, file_name)

				heatmap_sums = [
					SparseTensorWrapper(
						[heatmap_sums[p].feats_at(0)], [heatmap_sums[p].coords_at(0)],
						device=self.device
					) for p in range(max_phase + 1)
				]
				heatmap_cnts = [
					SparseTensorWrapper(
						[heatmap_cnts[p].feats_at(0)], [heatmap_cnts[p].coords_at(0)],
						device=self.device,
						coordinate_manager=heatmap_sums[p].coordinate_manager
					) for p in range(max_phase + 1)
				]

		for p in range(max_phase + 1):
			save_dir = os.path.join(
				self.config['log_dir'], 'step-' + str(step),
				'3d_heatmap'
			)
			save_path = os.path.join(
				save_dir,
				f'crop_len={img_config["crop_len"]}-heatmap_len={img_config["heatmap_len"]}-phase={p}.txt'
			)
			os.makedirs(save_dir, exist_ok=True)
			heatmap_sums[p].sparse_tensor = heatmap_sums[p].sparse_tensor / heatmap_cnts[p].sparse_tensor

			save_arr = torch.cat([
				heatmap_sums[p].coords_at(0),
				heatmap_sums[p].feats_at(0)
			], dim=1).cpu().numpy()
			np.savetxt(save_path, save_arr)

		self.train(training)


	def evaluate(self, x: SparseTensorWrapper, y: SparseTensorWrapper, step) -> float:
		batch_size = x.batch_size
		max_eval_phase = self.config['max_eval_phase']

		modes = ['eval', 'eval_no_gt']
		losses = []
		for mode in modes:
			s = x
			phases = [
				Phase(max_eval_phase, self.config['equilibrium_max_phase'])
				for _ in range(batch_size)
			]
			for p in range(max_eval_phase):
				try:
					with torch.no_grad():
						s_next, y_pad, loss = self.learn(x, s, y, step, phases, mode=mode)
				except RuntimeError:
					overflow_key = 'overflow/eval'
					self.scalar_summaries[overflow_key] = [self.scalar_summaries[overflow_key][0] + 1] if \
						len(self.scalar_summaries[overflow_key]) != 0 else [1]
					continue
				s = y_pad if mode == 'eval' else s_next
				losses.append(loss)
		return sum(losses) / float(len(losses))

	@timeit
	def test(
			self, x: SparseTensorWrapper, y: SparseTensorWrapper,
			step, save_vis=True, save_tensor=True
	) -> (List[float], List[float]):
		batch_size = x.batch_size
		test_trials = self.config['test_trials']
		max_test_phase = self.config['max_test_phase']
		voxel_size = self.config['voxel_size']
		testset = torch.load(
			os.path.join(
				self.config['test_root'],
				self.config['obj_class'], self.config['sampled_testset']
			)
		)  # torch tensor of N x 2048 x 3
		# debug_save_dir = os.path.join(self.config['log_dir'], 'debug_imgs')
		# os.makedirs(debug_save_dir, exist_ok=True)

		pred_coords = []  # list of coords
		preds = []  # list of sparse tensor wrapper
		for trial in range(test_trials):
			s = x
			for phase in range(max_test_phase):
				s = self.append_input(x, s) if self.config.get('append_input') else s
				with torch.no_grad():
					s = self.transition(s)
					# torch.save(s.coords_at(0), '{}/test_trial{}_{}.pt'.format(debug_save_dir, trial, phase))
			pred_coords.append([s.coords_at(batch_idx) for batch_idx in range(batch_size)])
			preds.append(s.to('cpu'))

		tmds = []
		mean_aligned_tmds = []
		uhds = []
		img_config = self.config['test_vis']
		# reshape s from test_trials x batch_size to batch_size x test_trials
		pred_coords = list(zip(*pred_coords))

		for batch_idx in range(batch_size):
			pred_coords_down = [
				downsample(c, self.config['test_pred_downsample']).unsqueeze(0)
				for c in pred_coords[batch_idx]
			]

			# compute alot of mmds
			# tensor of shape {test_trials} x {test_pred_downsample} x 3
			pred_coords_down = torch.cat(pred_coords_down, dim=0).float().to(self.device) * voxel_size
			testset = testset.to(self.device)
			self.mmd_calculator.add_generated_set(pred_coords_down, testset)

			mean_aligned_pred = pred_coords_down - pred_coords_down.mean(dim=1).unsqueeze(dim=1)
			self.mean_aligned_mmd_calculator.add_generated_set(mean_aligned_pred, testset)

			min_aligned_pred = pred_coords_down - pred_coords_down.min(dim=1).values.unsqueeze(dim=1)
			min_aligned_testset = testset - testset.min(dim=1).values.unsqueeze(dim=1)
			self.min_aligned_mmd_calculator.add_generated_set(min_aligned_pred, min_aligned_testset)

			# tmd
			tmds.append(mutual_difference(pred_coords_down))
			mean_aligned_tmds.append(mutual_difference(mean_aligned_pred))

			# uhd
			x_coord_down = downsample(x.coords_at(batch_idx), self.config['test_input_downsample'])
			x_coord_down = x_coord_down.float().to(self.device) * voxel_size
			uhds.append(unidirected_hausdorff_distance(x_coord_down, pred_coords_down))

		save_dir = os.path.join(
			self.config['log_dir'], img_config['save_dir'],
			'step-' + str(step)
		)
		os.makedirs(save_dir, exist_ok=True)

		if save_vis:
			# visualize and save x_input, y, predictions
			input_imgs = sparse_tensors2tensor_imgs(
				x, data_dim=self.config['data_dim'],
				img_config=img_config, batch_size=batch_size,
			)  # list of tensor of C x H x W
			gt_imgs = sparse_tensors2tensor_imgs(
				y, data_dim=self.config['data_dim'],
				img_config=img_config, batch_size=batch_size,
			)  # list of tensor of C x H x W
			pred_imgs = [
				sparse_tensors2tensor_imgs(
					preds[trial], data_dim=self.config['data_dim'],
					img_config=img_config, batch_size=batch_size,
				) for trial in range(test_trials)
			]  # list of length {test_trials} containing list of {batch_size} containing tensor of C x H x W

			for batch_idx, single_batch_pred_imgs in enumerate(zip(*pred_imgs)):
				imgs = [input_imgs[batch_idx], gt_imgs[batch_idx], *single_batch_pred_imgs]
				img = torch.cat(imgs, dim=1)  # tensor of C x (N * H) x W
				save_tensor_img(
					img, os.path.join(
						save_dir, 'img_' + x.names[batch_idx] + '.png'
					)
				)

		if save_tensor:
			# save the output tensors
			for batch_idx in range(batch_size):
				for trial in range(test_trials):
					torch.save(
						pred_coords[batch_idx][trial],
						os.path.join(save_dir, 'points_{}_trial{}.pt'.format(x.names[batch_idx], trial))
					)


		self.scalar_summaries['metrics/tmd'] += tmds
		self.list_summaries['metrics/tmd_histogram'] += tmds
		self.scalar_summaries['metrics/uhd'] += uhds
		self.list_summaries['metrics/uhd_histogram'] += uhds
		self.scalar_summaries['metrics/mean_aligned_tmd'] += mean_aligned_tmds
		self.list_summaries['metrics/mean_aligned_tmd_histogram'] += mean_aligned_tmds
		return tmds, uhds

	def generative_test(self, x: SparseTensorWrapper, y: SparseTensorWrapper, reference_set, step):

		batch_size = x.batch_size
		max_test_phase = self.config['max_test_phase']
		voxel_size = self.config['voxel_size']

		generated_coord_set = []  # list of coords
		generated_set = []  # list of spare tensor wrapper (for visualize)
		print('start testing')
		# generate |reference_set| shapes
		for trials in range(reference_set.shape[0]):
			try:
				s = x
				for phase in range(max_test_phase):
					with torch.no_grad():
						s = self.transition(s)
				generated_coord_set.append([s.coords_at(batch_idx) for batch_idx in range(batch_size)])
				generated_set.append(s.to('cpu'))
			except RuntimeError:
				breakpoint()
		print("complete generating {} shapes".format(reference_set.shape[0]))

		img_config = self.config['test_vis']
		generated_coord_set = list(zip(*generated_coord_set))

		for batch_idx in range(batch_size):
			generated_coord_set_down = [
				downsample(c, self.config['test_pred_downsample']).unsqueeze(0) for c in generated_coord_set[batch_idx]
			]
			generated_coord_set_down = torch.cat(generated_coord_set_down, dim=0).float().to(self.device) * voxel_size
			ref_set = reference_set.to(self.device)
			mean_aligned_genset = generated_coord_set_down - generated_coord_set_down.mean(dim=1).unsqueeze(dim=1)
			mean_aligned_refset = (reference_set - reference_set.mean(dim=1).unsqueeze(dim=1)).to(self.device)
			min_aligned_genset = generated_coord_set_down - generated_coord_set_down.min(dim=1).values.unsqueeze(dim=1)
			min_aligned_refset = (reference_set - reference_set.min(dim=1).values.unsqueeze(dim=1)).to(self.device)

			g, r, cd_mat = cd_matrix(generated_coord_set_down, ref_set)
			g, r, cd_mat_mean = cd_matrix(mean_aligned_genset, ref_set)
			g, r, cd_mat_ref_mean = cd_matrix(mean_aligned_genset, mean_aligned_refset)
			g, r, cd_mat_min = cd_matrix(min_aligned_genset, min_aligned_refset)
			# emd_mat = emd_matrix(generated_coord_set_down, reference_set)

			jsd = jensen_shannon_divergence(generated_coord_set_down, reference_set, 28)
			cov = coverage(g, r, cd_mat, cd_mat)
			mmd = minimum_matching_distance(g, r, cd_mat, cd_mat)
			nna = nearest_neighbor_accuracy(g, r, cd_mat, cd_mat)
			metrics = [jsd, cov, mmd, nna]

			jsd_mean = jensen_shannon_divergence(mean_aligned_genset, reference_set, 28)
			cov_mean = coverage(g, r, cd_mat_mean, cd_mat_mean)
			mmd_mean = minimum_matching_distance(g, r, cd_mat_mean, cd_mat_mean)
			nna_mean = nearest_neighbor_accuracy(g, r, cd_mat_mean, cd_mat_mean)
			metrics_mean = [jsd_mean, cov_mean, mmd_mean, nna_mean]

			jsd_ref_mean = jensen_shannon_divergence(mean_aligned_genset, mean_aligned_refset, 28)
			cov_ref_mean = coverage(g, r, cd_mat_ref_mean, cd_mat_ref_mean)
			mmd_ref_mean = minimum_matching_distance(g, r, cd_mat_ref_mean, cd_mat_ref_mean)
			nna_ref_mean = nearest_neighbor_accuracy(g, r, cd_mat_ref_mean, cd_mat_ref_mean)
			metrics_ref_mean = [jsd_ref_mean, cov_ref_mean, mmd_ref_mean, nna_ref_mean]
			print(metrics_ref_mean)
			jsd_min = jensen_shannon_divergence(min_aligned_genset, min_aligned_refset, 28)
			cov_min = coverage(g, r, cd_mat_min, cd_mat_min)
			mmd_min = minimum_matching_distance(g, r, cd_mat_min, cd_mat_min)
			nna_min = nearest_neighbor_accuracy(g, r, cd_mat_min, cd_mat_min)
			metrics_min = [jsd_min, cov_min, mmd_min, nna_min]

		pred_imgs = [
			sparse_tensors2tensor_imgs(
				generated_set[trial], data_dim=self.config['data_dim'],
				img_config=img_config, batch_size=batch_size,
			) for trial in range(reference_set.shape[0])
		]
		'''
		pred_imgs_mean = [
			sparse_tensors2tensor_imgs(
			)
		]
		'''

		save_dir = os.path.join(
			self.config['log_dir'], img_config['save_dir'],
			'step-' + str(step)
		)

		os.makedirs(save_dir, exist_ok=True)

		for batch_idx, single_batch_pred_imgs in enumerate(zip(*pred_imgs)):
			imgs = [*single_batch_pred_imgs]
			data_len = len(imgs)
			img_res = torch.tensor([])
			for i in range(10000):
				if data_len > 10:
					tmp_imgs = imgs[i * 10: (i + 1) * 10]
					if i == 0:
						img = torch.cat(tmp_imgs, dim=1)
					else:
						img = torch.cat((img, torch.cat(tmp_imgs, dim=1)), dim=2)
				else:
					tmp_imgs_res = imgs[i * 10:]
					img_res = torch.cat(tmp_imgs_res, dim=1)
				data_len -= 10
				if data_len < 0:
					break
			save_tensor_img(
				img, os.path.join(
					save_dir, 'generated_set.png'
				)
			)
			if img_res.shape[0] != 0:
				save_tensor_img(
					img_res, os.path.join(
						save_dir, 'generated_set_res.png'
					)
				)
		for batch_idx in range(batch_size):
			for trial in range(reference_set.shape[0]):
				torch.save(
					generated_coord_set[batch_idx][trial],
					os.path.join(save_dir, 'trial_{}.pt'.format(trial))
				)
		return metrics, metrics_mean, metrics_ref_mean, metrics_min
