# -*- coding:utf-8 -*-
import torch.nn.functional as F
import torch


def calculate_one_layer(this, stored, alpha=0.01):
    return (this[0] - stored[0]).abs().mean() + (this[1] - stored[1]).abs().mean() * alpha


def layer_1_loss(model, pretrained_params, bn_f, alpha=0.01):
    loss = calculate_one_layer([bn_f[0].features.mean(dim=(0, -2, -1)), bn_f[0].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.1.running_var']], alpha=alpha)
    return loss


def layer_2_loss(model, pretrained_params, bn_f, alpha=0.01):
    loss = calculate_one_layer([bn_f[1].features.mean(dim=(0, -2, -1)), bn_f[1].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.4.0.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.4.0.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[2].features.mean(dim=(0, -2, -1)), bn_f[2].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.4.0.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['rn.4.0.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[3].features.mean(dim=(0, -2, -1)), bn_f[3].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.4.1.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.4.1.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[4].features.mean(dim=(0, -2, -1)), bn_f[4].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.4.1.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['rn.4.1.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[5].features.mean(dim=(0, -2, -1)), bn_f[5].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.4.2.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.4.2.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[6].features.mean(dim=(0, -2, -1)), bn_f[6].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.4.2.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['rn.4.2.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[7].features.mean(dim=(0, -2, -1)), bn_f[7].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.5.0.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.5.0.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[8].features.mean(dim=(0, -2, -1)), bn_f[8].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.5.0.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['rn.5.0.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[9].features.mean(dim=(0, -2, -1)), bn_f[9].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.5.0.downsample.1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.5.0.downsample.1.running_var']], alpha=alpha)
    return loss


def layer_3_loss(model, pretrained_params, bn_f, alpha=0.01):
    loss = calculate_one_layer([bn_f[10].features.mean(dim=(0, -2, -1)), bn_f[10].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.5.1.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.5.1.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[11].features.mean(dim=(0, -2, -1)), bn_f[11].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.5.1.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['rn.5.1.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[12].features.mean(dim=(0, -2, -1)), bn_f[12].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.5.2.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.5.2.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[13].features.mean(dim=(0, -2, -1)), bn_f[13].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.5.2.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['rn.5.2.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[14].features.mean(dim=(0, -2, -1)), bn_f[14].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.5.3.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.5.3.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[15].features.mean(dim=(0, -2, -1)), bn_f[15].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.5.3.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['rn.5.3.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[16].features.mean(dim=(0, -2, -1)), bn_f[16].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.6.0.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.6.0.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[17].features.mean(dim=(0, -2, -1)), bn_f[17].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.6.0.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['rn.6.0.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[18].features.mean(dim=(0, -2, -1)), bn_f[18].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.6.0.downsample.1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.6.0.downsample.1.running_var']], alpha=alpha)
    return loss


def layer_4_loss(model, pretrained_params, bn_f, alpha=0.01):
    loss = calculate_one_layer([bn_f[19].features.mean(dim=(0, -2, -1)), bn_f[19].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.6.1.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.6.1.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[20].features.mean(dim=(0, -2, -1)), bn_f[20].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.6.1.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['rn.6.1.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[21].features.mean(dim=(0, -2, -1)), bn_f[21].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.6.2.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.6.2.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[22].features.mean(dim=(0, -2, -1)), bn_f[22].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.6.2.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['rn.6.2.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[23].features.mean(dim=(0, -2, -1)), bn_f[23].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.6.3.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.6.3.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[24].features.mean(dim=(0, -2, -1)), bn_f[24].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.6.3.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['rn.6.3.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[25].features.mean(dim=(0, -2, -1)), bn_f[25].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.6.4.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.6.4.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[26].features.mean(dim=(0, -2, -1)), bn_f[26].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.6.4.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['rn.6.4.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[27].features.mean(dim=(0, -2, -1)), bn_f[27].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.6.5.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.6.5.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[28].features.mean(dim=(0, -2, -1)), bn_f[28].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.6.5.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['rn.6.5.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[29].features.mean(dim=(0, -2, -1)), bn_f[29].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.7.0.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.7.0.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[30].features.mean(dim=(0, -2, -1)), bn_f[30].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.7.0.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['rn.7.0.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[31].features.mean(dim=(0, -2, -1)), bn_f[31].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.7.0.downsample.1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.7.0.downsample.1.running_var']], alpha=alpha)
    return loss


def layer_5_loss(model, pretrained_params, bn_f, alpha=0.01):
    loss = calculate_one_layer([bn_f[32].features.mean(dim=(0, -2, -1)), bn_f[32].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.7.1.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.7.1.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[33].features.mean(dim=(0, -2, -1)), bn_f[33].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.7.1.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['rn.7.1.bn2.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[34].features.mean(dim=(0, -2, -1)), bn_f[34].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.7.2.bn1.running_mean'],
                                      pretrained_params['model_state_dict']['rn.7.2.bn1.running_var']], alpha=alpha) + \
                 calculate_one_layer([bn_f[35].features.mean(dim=(0, -2, -1)), bn_f[35].features.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['rn.7.2.bn2.running_mean'],
                                      pretrained_params['model_state_dict']['rn.7.2.bn2.running_var']], alpha=alpha)
    return loss


def layer_6_loss(model, pretrained_params, bn_f, alpha=0.01):
    f = F.relu(torch.cat([bn_f[36].features, bn_f[37].features], dim=1))
    loss = calculate_one_layer([f.mean(dim=(0, -2, -1)), f.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['up1.bn.running_mean'],
                                      pretrained_params['model_state_dict']['up1.bn.running_var']], alpha=alpha)
    return loss


def layer_7_loss(model, pretrained_params, bn_f, alpha=0.01):
    f = F.relu(torch.cat([bn_f[38].features, bn_f[39].features], dim=1))
    loss = calculate_one_layer([f.mean(dim=(0, -2, -1)), f.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['up2.bn.running_mean'],
                                      pretrained_params['model_state_dict']['up2.bn.running_var']], alpha=alpha)
    return loss


def layer_8_loss(model, pretrained_params, bn_f, alpha=0.01):
    f = F.relu(torch.cat([bn_f[40].features, bn_f[41].features], dim=1))
    loss = calculate_one_layer([f.mean(dim=(0, -2, -1)), f.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['up3.bn.running_mean'],
                                      pretrained_params['model_state_dict']['up3.bn.running_var']], alpha=alpha)
    return loss


def layer_9_loss(model, pretrained_params, bn_f, alpha=0.01):
    f = F.relu(torch.cat([bn_f[42].features, bn_f[43].features], dim=1))
    loss = calculate_one_layer([f.mean(dim=(0, -2, -1)), f.var(dim=(0, -2, -1))],
                                     [pretrained_params['model_state_dict']['up4.bn.running_mean'],
                                      pretrained_params['model_state_dict']['up4.bn.running_var']], alpha=alpha)
    return loss


def bn_loss(model, pretrained_params, bn_f, alpha=0.01, i=5):
    loss_list = [
        layer_1_loss(model, pretrained_params, bn_f, alpha=alpha),
        layer_2_loss(model, pretrained_params, bn_f, alpha=alpha),
        layer_3_loss(model, pretrained_params, bn_f, alpha=alpha),
        layer_4_loss(model, pretrained_params, bn_f, alpha=alpha),
        layer_5_loss(model, pretrained_params, bn_f, alpha=alpha),
        layer_6_loss(model, pretrained_params, bn_f, alpha=alpha),
        layer_7_loss(model, pretrained_params, bn_f, alpha=alpha),
        layer_8_loss(model, pretrained_params, bn_f, alpha=alpha),
        layer_9_loss(model, pretrained_params, bn_f, alpha=alpha)
    ]

    total_loss = 0
    for n in range(i):
        total_loss += loss_list[n]

    return total_loss
