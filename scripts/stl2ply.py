from argparse import ArgumentParser
from glob import glob
from utils.util import get_unique_rot_mats
import os
import trimesh
import numpy as np
import tqdm
import typer
import point_cloud_utils as pcu

app = typer.Typer()

def get_unique_points(pts, voxel_size):
	# perform unique operation
	pts = np.around(pts / voxel_size).astype(np.int32)
	bbox_min, bbox_max = pts.min(axis=0), pts.max(axis=0)
	bbox_width = bbox_max - bbox_min + 1
	grid = np.zeros(bbox_width, dtype=bool)
	pts = pts - bbox_min
	grid[tuple(pts.T)] = True
	pts = np.stack(np.where(grid), axis=1)
	pts = voxel_size * (pts + bbox_min)
	return pts


def sample_pts(mesh_path, voxel_size, num_pts, rotate=False):
	mesh = trimesh.load_mesh(mesh_path)
	pts = trimesh.sample.sample_surface(mesh, num_pts)[0]
	pts = get_unique_points(pts, voxel_size)

	if rotate:
		# we rotate points to make the point cloud (rotationally) symmetrical
		rot_mats = get_unique_rot_mats().numpy()
		pts_rots = []
		for rot_mat in rot_mats:
			pts_rot = pts @ rot_mat
			pts_rots.append(np.round(pts_rot / voxel_size).astype(np.int32))
		pts_rots = np.unique(np.concatenate(pts_rots), axis=0)
		pts = voxel_size * pts_rots

	return pts

@app.command()
def main(
		stl_root,
		out_root,
		num_pts: int = 100000000,
		voxel_size: float = 3,
):
	os.makedirs(out_root, exist_ok=True)
	stl_paths = list(sorted(glob(os.path.join(stl_root, '*.stl'))))
	exclude_rot_names = [
		'RDR2',
		'CBR2',
	]  # do not rotate

	for file_path in tqdm.tqdm(stl_paths):
		rotate = True
		for exclude_rot_name in exclude_rot_names:
			if exclude_rot_name in file_path:
				rotate = False
				break
		pts = sample_pts(file_path, voxel_size, num_pts, rotate=rotate)
		out_path = os.path.join(out_root, os.path.basename(file_path)[:-4] + '.ply')
		pcu.save_mesh_v(out_path, pts)

if __name__ == '__main__':
	app()
