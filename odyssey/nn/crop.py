import torch
import torch.nn as nn
import torch.nn.functional as F

class Crop2D(nn.Module):
    def __init__(self, height, width, height_target, width_target):
        super(Crop2D, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target

        width_grid = self._step_to_range(2 / (self.width - 1), self.width_target)
        width_grid = width_grid[None, :].expand(self.height_target, -1)
        self.register_buffer("width_grid", width_grid)

        height_grid = self._step_to_range(2 / (self.height - 1), height_target)
        height_grid = height_grid[:, None].expand(-1, self.width_target)
        self.register_buffer("height_grid", height_grid)

    def _step_to_range(self, step, num_steps):
        return torch.tensor([step * (i - num_steps // 2) for i in range(num_steps)])

    def forward(self, inputs, x, y):
        """Calculates centered crop around given x,y coordinates.

        Args:
           inputs [B x H x W] or [B x C x H x W]
           coordinates [B x 2] x,y coordinates

        Returns:
           [B x C x H' x W'] inputs cropped and centered around x,y coordinates.
        """

        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(1).float()

        assert inputs.shape[2] == self.height, "expected %d but found %d" % (
            self.height,
            inputs.shape[2],
        )
        assert inputs.shape[3] == self.width, "expected %d but found %d" % (
            self.width,
            inputs.shape[3],
        )

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )
        crop = torch.round(F.grid_sample(inputs, grid, align_corners=True)).squeeze(1)
        return crop