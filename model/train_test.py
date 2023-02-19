import torch
import torch.nn as nn
from torch.utils import data
import os
import random
import numpy as np
from tqdm.auto import tqdm
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from dataset import utils, vipl_hr, Pixelmap
from evaluate import metric, postprocess
from configs import running
from . import model_disentangle
from loss_function.loss_cross import Cross_loss
from loss_function.loss_r import Neg_Pearson
from loss_function.loss_SNR import SNR_loss


# TODO: adjust weight
lambda_hr = 1

# lambda_img = 0.0000025  # 50 -> 1
lambda_img = 1  # 50 -> 1
# lambda_img = 50  # 50 -> 1

lambda_low_rank = 10
lambda_ecg = 0.02  # 2
# lambda_ecg = 2  # 2
lambda_snr = 1

# lambda_cross_fhr = 0.000005  # 10 -> 0.2
# lambda_cross_fn = 0.000005  # 10 -> 0.2
lambda_cross_fhr = 0.2  # 10 -> 0.2
lambda_cross_fn = 0.2  # 10 -> 0.2
# lambda_cross_fhr = 10  # 10 -> 0.2
# lambda_cross_fn = 10  # 10 -> 0.2

lambda_cross_hr = 1


def train_test(path, train_config, test_config, mode="Train", model_path=""):
    """
    :param path: for saving models
    :param train_config:
    :param test_config:
    :param mode:
    :param model_path:
    :return:
    """
    train_set = vipl_hr.VIPL_HR(config=train_config)
    test_set = vipl_hr.VIPL_HR(config=test_config)
    train_iter = data.DataLoader(train_set, batch_size=train_config.batch_size,
                                 shuffle=True, num_workers=8)
    test_iter = data.DataLoader(test_set, batch_size=test_config.batch_size,
                                shuffle=False, num_workers=8)
    # init and train
    net = model_disentangle.HR_disentangle_cross()
    if mode == "Train":
        net = net.to(train_config.device)
        optimizer = torch.optim.Adam([{'params': net.parameters(), 'lr': 0.0005}])
        print("Training...")
        train(net, optimizer, train_iter, train_config, test_iter, test_config)
        os.makedirs(path, exist_ok=True)
        # TODO: 修改名字
        torch.save(net.state_dict(), path + os.sep + f"cvd_aug_celoss_fold3.pt")
    else:
        assert model_path, "Pretrained model is required!"
        net.load_state_dict(torch.load(model_path))

    # test
    net = net.to(test_config.device)
    print(f"Evaluating...")
    # MAE, RMSE, MAPE, R
    temp = test(net, test_iter, test_config)
    print(f"Std: {temp[0]: .3f}\n"
          f"MAE: {temp[1]: .3f}\n"
          f"RMSE: {temp[2]: .3f}\n"
          f"R: {temp[3]: .3f}")


def train(net: nn.Module, optimizer: torch.optim.Optimizer,
          train_iter: data.DataLoader, train_config: running.TrainConfig,
          test_iter: data.DataLoader, test_config: running.TestConfig):
    net = net.to(train_config.device)
    net.train()

    lossfunc_HR = nn.L1Loss()
    lossfunc_img = nn.L1Loss()
    lossfunc_cross = Cross_loss(lambda_cross_fhr=lambda_cross_fhr, lambda_cross_fn=lambda_cross_fn,
                                lambda_cross_hr=lambda_cross_hr)
    lossfunc_ecg = Neg_Pearson(downsample_mode=0)
    # TODO: use SNR loss (7) or not (1)
    lossfunc_SNR = SNR_loss(clip_length=300, loss_type=1, device=train_config.device)
    # TODO: check scheduler
    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.5)

    train_loss = metric.Accumulate(1)  # for print
    progress_bar = tqdm(range(train_config.num_epochs))

    for epoch in range(train_config.num_epochs):
        train_loss.reset()
        for data, bpm, fps, bvp, gt, _ in train_iter:
            data = Variable(data)
            bvp = Variable(bvp)
            # bpm = Variable(bpm.view(-1, 1))  # raw lables
            fps = Variable(fps.view(-1, 1))
            data = data.to(train_config.device)
            # bpm = bpm.to(train_config.device)
            fps = fps.to(train_config.device)
            bvp = bvp.to(train_config.device)
            gt = gt.to(train_config.device).reshape(-1, 1)

            """feat_hr, feat_n, output, img_out, feat_hrf1, feat_nf1, hrf1, \
                idx1, feat_hrf2, feat_nf2, hrf2, idx2, ecg, ecg1, ecg2 = net(data)"""
            feat_hr, feat_n, output, img_out, feat_hrf1, feat_nf1, hrf1, \
                idx1, feat_hrf2, feat_nf2, hrf2, idx2, ecg = net(data)

            # calculate loss
            loss_hr = lossfunc_HR(output, gt) * lambda_hr  # HR 预测损失
            loss_img = lossfunc_img(data, img_out) * lambda_img  # MSTMap 重建损失
            loss_ecg = lossfunc_ecg(ecg, bvp) * lambda_ecg  # ECG 相关损失
            loss_SNR, tmp = lossfunc_SNR(ecg, gt, fps, pred=output, flag=None) * lambda_snr  # 信噪比
            loss = loss_hr + loss_ecg + loss_img + loss_SNR
            loss_cross, loss_hr1, loss_hr2, loss_fhr1, loss_fhr2, loss_fn1, loss_fn2, \
                loss_hr_dis1, loss_hr_dis2 = lossfunc_cross(feat_hr, feat_n, output,
                                                            feat_hrf1, feat_nf1,
                                                            hrf1, idx1,
                                                            feat_hrf2, feat_nf2,
                                                            hrf2, idx2, gt)
            loss = loss + loss_cross

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # for print
            train_loss.update(val=[loss.data], n=1)
        progress_bar.update(1)
        scheduler.step()
        print(f"****************************************************\n"
              f"Epoch{epoch + 1}:\n"
              f"Train loss: {train_loss.acc[0] / train_loss.cnt[0]: .3f}\n"
              f"****************************************************")


def test(net: nn.Module, test_iter: data.DataLoader,
         test_config: running.TestConfig) -> list:
    net = net.to(test_config.device)
    net.eval()
    predictions = dict()
    labels = dict()
    progress_bar = tqdm(range(len(test_iter)))
    for data, bpm, fps, bvp, gt, subjects in test_iter:
        data = data.to(test_config.device)
        # bpm = bpm.to(test_config.device)
        fps = fps.to(test_config.device)
        # bvp = bvp.to(test_config.device)
        gt = gt.to(test_config.device)

        """feat_hr, feat_n, output, img_out, feat_hrf1, feat_nf1, hrf1, idx1, \
            feat_hrf2, feat_nf2, hrf2, idx2, ecg, ecg1, ecg2 = net(data)"""
        feat_hr, feat_n, output, img_out, feat_hrf1, feat_nf1, hrf1, \
                idx1, feat_hrf2, feat_nf2, hrf2, idx2, ecg = net(data)

        for i in range(len(data)):
            file_name = subjects[i]
            if file_name not in predictions.keys():
                predictions[file_name] = []
                labels[file_name] = []
            # beat num to heart rate !!!
            # bpm = gt_temp*clip_length/fps/60;
            predictions[file_name].append((output[i] / 300 * fps[i] * 60).detach().cpu().numpy())
            labels[file_name].append((gt[i] / 300 * fps[i] * 60).detach().cpu().numpy())
        progress_bar.update(1)
    pred_phys = []
    label_phys = []
    # 合并同一视频的预测 hr
    for file_name in predictions.keys():
        # average hr
        pred_temp = np.asarray(predictions[file_name]).mean()
        label_temp = np.asarray(labels[file_name]).mean()

        pred_phys.append(pred_temp)
        label_phys.append(label_temp)
    pred_phys = np.asarray(pred_phys)
    label_phys = np.asarray(label_phys)

    temp = metric.cal_metric(pred_phys, label_phys)
    print(temp)
    return temp


def fixSeed(seed: int):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi gpu
    # torch.backends.cudnn.deterministic = True  # 会大大降低速度
    torch.backends.cudnn.benchmark = True  # False会确定性地选择算法，会降低性能
    torch.backends.cudnn.enabled = True  # 增加运行效率，默认就是True
    torch.manual_seed(seed)
