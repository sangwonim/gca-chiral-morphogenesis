from mpl_toolkits.mplot3d import Axes3D
from torchvision.transforms import ToTensor, Resize, ToPILImage
from typing import List
from utils.sparse_tensor import SparseTensorWrapper

import io
import torch
import os
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt


def save_tensor_img(img, path):
    img = ToPILImage()(img)
    img.save(path)


def plt_to_tensor(plt, h, w, clear=True):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    if clear:
        plt.clf()
    buf.seek(0)
    img = PIL.Image.open(buf)
    img = Resize((h, w))(img)
    return ToTensor()(img)


def vis_3d_coords(x: SparseTensorWrapper, num_views, max_sample, h, w, alpha, axis_ranges, batch_size, losses=None) \
        -> (List[torch.Tensor], List):
    '''
    Args
        x:
            SparseTensor of 3 dim
        max_sample:
            maximum number of samples
        num_view:
            number of views to visualize in plot
        axis_ranges:
            list of [x_lim, y_lim, z_lim]
            used to make the scale of axis consistent
    :return:
        imgs: list of img (in tensor)
        new_axis_range: [x_lim, y_lim, z_lim]
            x_lim: range of img's x axes (tuple of min max)
            y_lim: range of img's y axes (tuple of min max)
            z_lim: range of img's z axes (tuple of min max)
    '''
    imgs = []

    for batch_idx in range(batch_size):
        # Create Image Figure
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


        # Set axis range for plotting
        if axis_ranges is not None:
            ax.set_xlim(axis_ranges[0])
            ax.set_ylim(axis_ranges[1])
            ax.set_zlim(axis_ranges[2])

            try:
                coord = x.coords_at(batch_idx).cpu()
                if max_sample > 0:
                    sample = np.random.RandomState(0)\
                        .permutation(coord.shape[0])[:max_sample]
                    coord = coord[sample]
            except RuntimeError:
                coord = torch.zeros(1, 3)

        ax.scatter(
            xs=coord[:, 0], ys=coord[:, 2], zs=coord[:, 1],
            alpha=alpha, marker='o', linewidths=0
        )
        if losses != None:
            ax.set_title("Loss: {0:.3f}".format(losses[batch_idx]), fontsize=24)

        single_img = []
        init_angle = -30
        for angle in range(init_angle, init_angle + 360, int(360 / num_views)):
            ax.view_init(elev=None, azim=angle)
            single_img.append(plt_to_tensor(plt, h, w, clear=False))
        imgs.append(torch.cat(single_img, dim=2))
        plt.close('all')

    return imgs


def vis_2d_coords(x: SparseTensorWrapper, h, w, batch_size=None) -> List[torch.Tensor]:
    '''
    Input:
        x: SparseTensorWrapper of 2 dim
        h: height of output tensor
        w: width of output tensor
    Output:
        list of tensors (imgs) of h x w
    '''
    assert x.coords.shape[1] - 1 == 2
    if not batch_size:
        batch_size = x.coords[:, 0].max() + 1
    imgs = []

    for batch_idx in range(batch_size):
        single_idx = (x.coords[:, 0] == batch_idx)
        img = torch.zeros([h, w])

        if len(single_idx) != 0:
            coord = x.coords[single_idx, 1:]
            # remove mask if exists
            feat = x.feats[single_idx, 0] \
                if len(x.feats.shape) == 2 else x.feats[single_idx]
            for (i, c) in enumerate(coord):
                if (c[0] >= h) or (c[0] < 0):
                    continue
                if (c[1] >= w) or (c[1] < 0):
                    continue
                img[c[0], c[1]] = feat[i]

        imgs.append(img.unsqueeze(dim=0))
    return imgs

def vis_3d_tensor(x, num_views, max_sample, h, w, alpha, axis_ranges, losses=None) -> (List[torch.Tensor], List):
    # x: B x N x 3 tensor(?)
    imgs = []
    batch_size = x.shape[0]
    for batch_idx in range(batch_size):
        # Create Image Figure
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set axis range for plotting
        if axis_ranges is not None:
            ax.set_xlim(axis_ranges[0])
            ax.set_ylim(axis_ranges[1])
            ax.set_zlim(axis_ranges[2])

            coord = x[batch_idx]

        ax.scatter(
            xs=coord[:, 0], ys=coord[:, 2], zs=coord[:, 1],
            alpha=alpha, marker='o', linewidths=0
        )
        if losses != None:
            ax.set_title("Loss: {0:.3f}".format(losses[batch_idx]), fontsize=24)

        single_img = []
        init_angle = -30
        for angle in range(init_angle, init_angle + 360, int(360 / num_views)):
            ax.view_init(elev=None, azim=angle)
            single_img.append(plt_to_tensor(plt, h, w, clear=False))
        imgs.append(torch.cat(single_img, dim=2))
        plt.close('all')

    return imgs


def sparse_tensors2tensor_imgs(
        x: SparseTensorWrapper, data_dim,
        img_config, batch_size,
        losses=None
) -> List[torch.Tensor]:
    if data_dim == 2:
        tensor_imgs = vis_2d_coords(x, img_config['height'], img_config['width'])
    elif data_dim == 3:
        tensor_imgs = vis_3d_coords(
            x, num_views=img_config['num_views'], max_sample=img_config['max_sample'],
            h=img_config['height'], w=img_config['width'], alpha=img_config['alpha'],
            axis_ranges=img_config['axis_ranges'], batch_size=batch_size,
            losses=losses
        )
    else:
        raise NotImplementedError

    return tensor_imgs


def save_visualized_tensor(intermediate_coords, tensor_path, train_str, mini_batch_indices, phase):
    for vis_cnt, coords in enumerate(intermediate_coords):
        for i in coords[:, 0].unique():
            torch.save(
                coords[coords[:, 0] == i, 1:],
                os.path.join(
                    tensor_path, '{}-img-{}-phase-{}-{}.pt'
                    .format(
                        train_str, str(mini_batch_indices[i]).zfill(4),
                        str(phase).zfill(2), vis_cnt
                    )
                )
            )

