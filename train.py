import os
import torch
from torch.utils.tensorboard import SummaryWriter
from models.gca import GCA
from data import DataScheduler


def train_model(
        config, model: GCA,
        scheduler: DataScheduler,
        writer: SummaryWriter,
        resume_step: int = 0
):
    saved_model_path = os.path.join(config['log_dir'], 'ckpts')
    os.makedirs(saved_model_path, exist_ok=True)

    skip_batch = 0

    for step, (x, s, y, phases, epoch) in enumerate(scheduler, resume_step):  # step starts from load_step
        # save checkpoint
        if ((step + 1) % config['ckpt_step'] == 0) or ((step + 1) % config['summary_step'] == 0):
            check_point_prefix = 'ckpt-step-'
            torch.save({
                'step': step,
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'lr_scheduler_state_dict': model.lr_scheduler.state_dict()
            },
                os.path.join(
                    saved_model_path,
                    check_point_prefix + str(step + 1)
                )
            )

            if (step + 1 - config['summary_step']) % config['ckpt_step'] != 0:
                # remove latest checkpoint
                latest_ckpt_path = os.path.join(
                    saved_model_path,
                    check_point_prefix + str(step + 1 - config['summary_step'])
                )
                if os.path.exists(latest_ckpt_path):
                    os.remove(latest_ckpt_path)

        # evaluate
        if scheduler.check_eval_step(step):
            scheduler.eval(model, writer, step)

        # test
        if scheduler.check_test_step(step):
            scheduler.test(model, writer, step)

        # visualize
        if scheduler.check_vis_step(step):
            print("\nVisualizing...")
            scheduler.visualize(model, writer, step)
            writer.add_scalar('skip_batch', skip_batch, step)

        # write summary
        if scheduler.check_summary_step(step):
            model.write_summary(step)

        if scheduler.check_empty_cache_step(step):
            torch.cuda.empty_cache()

        # since number of points vary in the dataset,
        # we skip if gpu overflow occurs
        if config['skip_gpu_overflow']:
            try:
                s_next, phases, train_loss = model.learn(x, s, y, step, phases)
            except RuntimeError:
                skip_batch += 1
                continue
        else:
            s_next, phases, train_loss = model.learn(x, s, y, step, phases)
        scheduler.data_buffer.push(x, s_next, y, phases)

        # model learns
        print('\r[Epoch {:4}, Step {:7}, Overflow: {:7}, Loss {:5}]'.format(
            epoch, step + 1, skip_batch, '%.3f' % train_loss), end=''
        )

        if config.get('last_step') is not None:
            if (step + 1) == config['last_step']:
                return

