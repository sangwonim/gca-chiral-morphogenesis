import os
from torch.utils.tensorboard import SummaryWriter
from models.gca import GCA
from data import DataScheduler


def vis_model(
        config, model: GCA,
        scheduler: DataScheduler,
        writer: SummaryWriter,
        resume_step: int = 0
):
    saved_model_path = os.path.join(config['log_dir'], 'ckpts')
    os.makedirs(saved_model_path, exist_ok=True)

    # visualize
    if scheduler.check_vis_step(resume_step):
        print("\nVisualizing...")
        scheduler.dataset.config['vis']['indices'] = 100000
        scheduler.visualize(model, writer, resume_step, skip_eval=False)
