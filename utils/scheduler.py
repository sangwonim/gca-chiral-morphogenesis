import torch.tensor
from utils.phase import Phase
from typing import List


class InfusionScheduler:
    def __init__(self, config):
        self.scheduler_config = config.get('infusion_scheduler')
        if self.scheduler_config is None:
            # for backward compatibility
            self.scheduler = lambda step: config['infusion_rate']
        else:
            speed = float(self.scheduler_config['speed'])
            initial_rate = float(self.scheduler_config['initial_rate'])
            if self.scheduler_config.get('type') == 'linear':
                self.scheduler = lambda phase: min(initial_rate + speed * phase, 1.)
            else:
                raise ValueError(
                    'scheduling type {} does not exist'.format(
                        self.scheduler_config.get('type')
                    )
                )

    def sample(self, phases: List[Phase]) -> List[float]:
        return [self.scheduler(p.phase) for p in phases]

    def current_infusion_rate(self, step):
        return self.scheduler(step)

