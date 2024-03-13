from torch.utils.tensorboard import SummaryWriter
from models.gca import GCA
from data import DataScheduler


def test_model(
        config, model: GCA,
        scheduler: DataScheduler,
        writer: SummaryWriter,
        step, num_test
):
    for test_cnt in range(num_test):
        print('test step {}'.format(test_cnt))
        scheduler.test(
            model, writer, step + test_cnt,
            save_vis=False, save_tensor=False
        )
