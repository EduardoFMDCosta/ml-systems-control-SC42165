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
                 controller: int,
                 velocity: float,
                 disc_step: float,
                 **kwargs):

        self.num_dims = 3
        self.controller = controller
        self.velocity = velocity
        self.disc_step = disc_step

        super(DubinsCarDynamics, self).__init__()


    def __call__(self, state: torch.Tensor):
        next_state_1 = state[:, 0] + self.disc_step * self.velocity * torch.sin(state[:, 2])
        next_state_2 = state[:, 1] + self.disc_step * self.velocity * torch.cos(state[:, 2])

        if self.controller == 0:
            next_state_3 = (state[:, 2] + self.disc_step * 1.0) % (2 * math.pi)

        elif self.controller == 1:

            next_state_3 = state[:, 2]

            mask_first_quad = (state[:, 0] >= 0) & (state[:, 1] >= 0)
            mask_second_quad = (state[:, 0] >= 0) & (state[:, 1] < 0)
            mask_third_quad = (state[:, 0] < 0) & (state[:, 1] < 0)
            mask_fourth_quad = (state[:, 0] < 0) & (state[:, 1] >= 0)

            next_state_3[mask_first_quad] = math.pi / 4 + math.pi
            next_state_3[mask_second_quad] = math.pi / 4 + 3 * math.pi / 2
            next_state_3[mask_third_quad] = math.pi / 4
            next_state_3[mask_fourth_quad] = math.pi / 4 + math.pi / 2

        return torch.stack((next_state_1, next_state_2, next_state_3), dim=1)

class PendulumDynamics(Dynamics):
    def __init__(self,
                 gravity: float,
                 length: float,
                 mass: float,
                 friction: float,
                 disc_step: float,
                 **kwargs):

        self.num_dims = 2
        self.gravity = gravity
        self.length = length
        self.mass = mass
        self.friction = friction
        self.disc_step = disc_step

        super(PendulumDynamics, self).__init__()

    def __call__(self, state: torch.Tensor):

        next_state_1 = state[:, 0] + self.disc_step * state[:, 1]
        next_state_2 = state[:, 1] + self.disc_step * (-self.gravity / self.length * torch.sin(state[:, 0]) - self.friction / (self.mass * self.length ** 2) * state[:, 1])

        return torch.stack((next_state_1, next_state_2), dim=1)

def get_dynamics(dynamics_type: str, **kwargs):
    if dynamics_type == 'DubinsCar':
        return DubinsCarDynamics(**kwargs)
    elif dynamics_type == 'Pendulum':
        return PendulumDynamics(**kwargs)
    else:
        raise ValueError(f"Unknown dynamics: {dynamics_type}")