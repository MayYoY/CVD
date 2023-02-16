import math
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class SNR_loss(nn.Module):
    def __init__(self, clip_length=300, delta=3, loss_type=1, use_wave=False, device="cpu"):
        super(SNR_loss, self).__init__()

        self.clip_length = clip_length
        self.time_length = 300
        self.delta = delta
        self.delta_distribution = [0.4, 0.25, 0.05]
        self.low_bound = 40
        self.high_bound = 150

        self.bpm_range = torch.arange(self.low_bound, self.high_bound, dtype=torch.float).to(device)
        self.bpm_range = self.bpm_range / 60.0

        self.pi = 3.14159265
        two_pi_n = Variable(2 * self.pi * torch.arange(0, self.time_length, dtype=torch.float))
        hanning = Variable(torch.from_numpy(np.hanning(self.time_length)).type(torch.FloatTensor),
                           requires_grad=True).view(1, -1)  # 1 x T

        self.two_pi_n = two_pi_n.to(device)
        self.hanning = hanning.to(device)

        self.cross_entropy = nn.CrossEntropyLoss()
        self.nll = nn.NLLLoss()
        self.l1 = nn.L1Loss()

        self.loss_type = loss_type
        self.eps = 0.0001

        self.lambda_l1 = 0.1
        self.use_wave = use_wave

    def forward(self, wave, gt, fps, pred=None, flag=None):  # all variable operation
        """
        DFT: F(**k**) = \sum_{n = 0}^{N - 1} f(n) * \exp{-j2 \pi n **k** / N}
        :param wave: predict ecg  B x T
        :param gt: beat num
        :param fps:
        :param pred:
        :param flag:
        :return:
        """

        if flag is not None:
            idx = flag.eq(1)
            wave = wave[idx, :]
            gt = gt[idx, :]
            fps = fps[idx, :]
            pred = pred[idx, :]

            if gt.shape[0] == 0:
                loss = 0.0
                return loss, 0

        # 由心拍次数转换为心跳
        hr = torch.mul(gt, fps)
        hr = hr * 60 / self.clip_length
        # truncate
        hr[hr.ge(self.high_bound)] = self.high_bound - 1
        hr[hr.le(self.low_bound)] = self.low_bound

        if pred is not None:
            pred = torch.mul(pred, fps)
            pred = pred * 60 / self.clip_length

        batch_size = wave.shape[0]

        f_t = self.bpm_range / fps  # dft 中的 k = N x fk / fs, N 与 -j2 \pi n **k** / N 抵消
        preds = wave * self.hanning  # B x T

        preds = preds.view(batch_size, 1, -1)  # B x 1 x T
        f_t = f_t.view(batch_size, -1, 1)  # B x range x 1

        tmp = self.two_pi_n.repeat(batch_size, 1)
        tmp = tmp.view(batch_size, 1, -1)  # B x 1 x T

        # B x range
        complex_absolute = torch.sum(preds * torch.sin(f_t * tmp), dim=-1) ** 2 \
                           + torch.sum(preds * torch.cos(f_t * tmp), dim=-1) ** 2

        target = hr - self.low_bound  # 移动区间 [40, 150] -> [0, 110]
        target = target.type(torch.long).view(-1)

        whole_max_val, whole_max_idx = complex_absolute.max(1)
        whole_max_idx = whole_max_idx + self.low_bound  # 预测心率

        if self.loss_type == 1:
            loss = self.cross_entropy(complex_absolute, target)

        elif self.loss_type == 7:
            loss = self.cross_entropy(complex_absolute, target)

            norm_t = (torch.ones(batch_size).to(wave.device) / torch.sum(complex_absolute, dim=1))
            norm_t = norm_t.view(-1, 1)
            complex_absolute = complex_absolute * norm_t

            idx_l = target - self.delta
            idx_l[idx_l.le(0)] = 0
            idx_r = target + self.delta
            idx_r[idx_r.ge(self.high_bound - self.low_bound - 1)] = self.high_bound - self.low_bound - 1

            loss_snr = 0.0
            for i in range(0, batch_size):
                loss_snr = loss_snr + 1 - torch.sum(complex_absolute[i, idx_l[i]:idx_r[i]])

            loss_snr = loss_snr / batch_size

            loss = loss + loss_snr

        return loss, whole_max_idx
