import math
import torch

class Dynamics(torch.nn.Sequential):
    num_dims = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        pass


class DubinsCarDynamics(Dynamics):
    def __init__(self,
                 velocity: float,
                 control: float,
                 disc_step: float = 0.1,
                 **kwargs):

        self.num_dims = 3
        self.velocity = velocity
        self.control = control
        self.disc_step = disc_step

        super(DubinsCarDynamics, self).__init__()


    def __call__(self, state: torch.Tensor):
        next_state_1 = state[:, 0] + self.disc_step * self.velocity * torch.sin(state[:, 2])
        next_state_2 = state[:, 1] + self.disc_step * self.velocity * torch.cos(state[:, 2])
        next_state_3 = (state[:, 2] + self.disc_step * self.control) % (2 * math.pi)

        return torch.stack((next_state_1, next_state_2, next_state_3), dim=1)