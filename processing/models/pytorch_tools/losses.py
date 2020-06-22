import torch.nn as nn
import torch


class cMSE(nn.Module):
    def __init__(self, c):
        super(cMSE, self).__init__()

        self.c = c

    def forward(self, x, y):
        mse = torch.mean((x[:, 1] - y[:, 1]) ** 2)
        dx = x[:, 1] - x[:, 0]
        dy = y[:, 1] - y[:, 0]
        dmse = torch.mean(((dx - dy)) ** 2)

        return mse + self.c * dmse, mse, dmse


class gcMSE(nn.Module):
    def __init__(self, mean_y, std_y, freq, pega_coeff, rega_coeff, c):
        super(gcMSE, self).__init__()
        self.pega_grid = PEGA_Grid()
        self.rega_grid = REGA_Grid()
        self.freq = freq
        self.c = c
        self.mean_y = torch.Tensor(mean_y).cuda()
        self.std_y = torch.Tensor(std_y).cuda()
        self.pega_coeff = torch.Tensor(list(pega_coeff.values())).cuda()
        self.rega_coeff = torch.Tensor(list(rega_coeff.values())).cuda()

    def _scale(self, x):
        return x * self.std_y + self.mean_y

    def _rate_of_change(self, x):
        return (x[:, 1] - x[:, 0]) / self.freq

    def forward(self, x, y):
        x_scaled, y_scaled = self._scale(x), self._scale(y)
        dx_scaled, dy_scaled = self._rate_of_change(x_scaled), self._rate_of_change(y_scaled)

        pega = self.pega_grid(x_scaled[:, 1], y_scaled[:, 1], dy_scaled)
        rega = self.rega_grid(dx_scaled, dy_scaled)
        pega_weights = torch.sum(pega.transpose(1, 0).float() * self.pega_coeff, axis=1)
        rega_weights = torch.sum(rega.transpose(1, 0).float() * self.rega_coeff, axis=1)

        dx, dy = x[:, 1] - x[:, 0], y[:, 1] - y[:, 0]
        mse = torch.mean(pega_weights * (x[:, 1] - y[:, 1]) ** 2)
        dmse = torch.mean(rega_weights * ((dx - dy)) ** 2)

        pega_a_plus_b = torch.sum(pega[:2])
        rega_a_plus_b = torch.sum(rega[:2])

        return mse + self.c * dmse, mse, dmse, pega_a_plus_b, rega_a_plus_b


class PEGA_Grid(nn.Module):
    def _compute_mod(self, dy):
        mod1_ind = ((dy > -2) & (dy <= -1)) | ((dy < 2) & (dy >= 1))
        mod2_ind = ((dy <= -2)) | ((dy >= 2))

        mod = torch.zeros_like(dy)
        mod1 = torch.where(mod1_ind, torch.ones_like(dy) * 10, torch.zeros_like(dy))
        mod2 = torch.where(mod2_ind, torch.ones_like(dy) * 20, torch.zeros_like(dy))
        mod = mod + mod1 + mod2

        return mod

    def A_region(self, x, y, mod):
        return (((x <= 70 + mod) & (y <= 70)) | ((x <= y * 6 / 5 + mod) & (x >= y * 4 / 5 - mod)))

    def B_region(self, x, y, mod):
        return (~self.A_region(x, y, mod)) & (~self.uC_region(x, y, mod)) & (~self.lC_region(x, y, mod)) & \
        (~self.uD_region(x, y, mod)) & (~self.lD_region(x, y, mod)) & (~self.uE_region(x, y, mod)) & \
        (~self.lE_region(x, y, mod))

    def uC_region(self, x, y, mod):
        return ((y > 70) & (x > y * 22 / 17 + (180 - 70 * 22 / 17) + mod))

    def lC_region(self, x, y, mod):
        return ((y <= 180) & (x < y * 7 / 5 - 182 - mod))

    def uD_region(self, x, y, mod):
        return ((x > 70 + mod) & (x > y * 6 / 5 + mod) & (y <= 70) & (x <= 180 + mod))

    def lD_region(self, x, y, mod):
        return ((y > 240) & (x < 180 - mod) & (x >= 70 - mod))

    def uE_region(self, x, y, mod):
        return ((x > 180 + mod) & (y <= 70))

    def lE_region(self, x, y, mod):
        return ((y > 180) & (x < 70 - mod))

    def forward(self, x, y, dy):
        mod = self._compute_mod(dy)
        return torch.stack([self.A_region(x, y, mod), self.B_region(x, y, mod),
                            self.uC_region(x, y, mod), self.lC_region(x, y, mod), self.uD_region(x, y, mod),
                            self.lD_region(x, y, mod), self.uE_region(x, y, mod), self.lE_region(x, y, mod)])

class REGA_Grid(nn.Module):
    def A_region(self, dx, dy):
        return ((dx >= dy - 1) & (dx <= dy + 1)) | ((dx <= dy / 2) & (dx >= dy * 2)) | ((dx <= dy * 2) & (dx >= dy / 2))

    def B_region(self, dx, dy):
        return (~self.A_region(dx,dy)) & (((dx <= -1) & (dy <= -1)) | ((dx <= dy + 2) & (dx >= dy - 2)) | ((dx >= 1) & (dy >= 1)))

    def uC_region(self, dx, dy):
        return ((dy < 1) & (dy >= -1) & (dx > dy + 2))

    def lC_region(self, dx, dy):
        return ((dy <= 1) & (dy > -1) & (dx < dy - 2))

    def uD_region(self, dx, dy):
        return ((dx <= 1) & (dx > -1) & (dx > dy + 2))

    def lD_region(self, dx, dy):
        return ((dx < 1) & (dx >= -1) & (dx < dy - 2))

    def uE_region(self, dx, dy):
        return ((dx > 1) & (dy < -1))

    def lE_region(self, dx, dy):
        return ((dx < -1) & (dy > 1))

    def forward(self, dx, dy):
        return torch.stack(
            [self.A_region(dx, dy), self.B_region(dx, dy), self.uC_region(dx, dy), self.lC_region(dx, dy),
             self.uD_region(dx, dy), self.lD_region(dx, dy), self.uE_region(dx, dy), self.lE_region(dx, dy)])


