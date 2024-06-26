#!/usr/bin/env python3
from argparse import ArgumentParser
import os
import yaml
import resource
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from data import DataScheduler
from models import MODEL
from train import train_model
from test import test_model
from vis import vis_model

# Increase maximum number of open files from 1024 to 4096
# as suggested in https://github.com/pytorch/pytorch/issues/973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

parser = ArgumentParser()
parser.add_argument(
    '--config', '-c',
    default='configs/material.yaml'
)
parser.add_argument(
    '--log-dir', '-l',
    default='./log/' + datetime.now().strftime('%m-%d-%H:%M:%S')
)
parser.add_argument('--resume-ckpt')
parser.add_argument('--override', default='')
parser.add_argument('--evaluate', '-e', default=False, action='store_true')
parser.add_argument('--test-vis', default=False, action='store_true')
parser.add_argument('--test', default=None, type=int)  # number of testing
parser.add_argument('--vis', default=False, action='store_true')

def main():
    args = parser.parse_args()

    # Load config
    config_path = args.config
    if args.resume_ckpt:
        base_dir = os.path.dirname(os.path.dirname(args.resume_ckpt))
        config_path = os.path.join(base_dir, 'config.yaml')
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    # Override options
    for option in args.override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                raise ValueError('{} is not defined in config file. '
                                 'Failed to override.'.format(address))
            here = here[key]
        if keys[-1] not in here:
            raise ValueError('{} is not defined in config file. '
                             'Failed to override.'.format(address))
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)

    # Set log directory
    config['log_dir'] = args.log_dir
    if not args.resume_ckpt and os.path.exists(args.log_dir):
        print('WARNING: %s already exists' % args.log_dir)
        input('Press enter to continue')
        print()

    if args.resume_ckpt and not args.log_dir:
        config['log_dir'] = os.path.dirname(
            os.path.dirname(args.resume_ckpt)
        )

    # Save config
    os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
    if not args.resume_ckpt:
        config_save_path = os.path.join(config['log_dir'], 'config.yaml')
        yaml.dump(config, open(config_save_path, 'w'))
        print('Config saved to {}'.format(config['log_dir']))

    # set random seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # Build components
    writer = SummaryWriter(config['log_dir'])
    model = MODEL[config['model']](config, writer)

    model.to(config['device'])
    if args.resume_ckpt:
        # load trained model
        checkpoint = torch.load(args.resume_ckpt)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('lr_scheduler_state_dict') is not None:
            model.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        resume_step = checkpoint['step'] + 1
    else:
        resume_step = 0

    data_scheduler = DataScheduler(config, args.test_vis)
    if args.resume_ckpt:
        data_scheduler.epoch_cnt = checkpoint['epoch']

    if args.test is not None:
        test_model(config, model, data_scheduler, writer, resume_step, args.test)
    elif args.vis:
        vis_model(config, model, data_scheduler, writer, resume_step)
    else:
        train_model(config, model, data_scheduler, writer, resume_step)


if __name__ == '__main__':
    main()
