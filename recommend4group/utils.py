"""
    Some handy functions for pytroch model training ...
"""
import torch
import pandas as pd
import numpy as np


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):

    pretrained_dict = torch.load(model_dir, map_location=lambda storage, loc: storage.cuda(device=device_id))
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), 'CUDA is not available'
        torch.cuda.set_device(device_id)


def use_optimizer(network, params):
    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam([{"params": [y[1] for y in network.named_parameters() if ("attention" not in y[0]) and (y[1].requires_grad)]},
                                      {"params": [y[1] for y in network.named_parameters() if ("attention" in y[0]) and (y[1].requires_grad)],
                                       "lr": params['adam_lr']*params['lr_ratio']}], lr=params['adam_lr'], weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, network.parameters()),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer

def load_friends(path):
    user_friends = {}
    cnt = 0

    df = pd.read_csv(path, header=0, dtype={0: np.int})
    for i, row in df.iterrows():
        friends = [int(i) for i in row[1].split(",")]
        user_friends[row[0]] = friends
        cnt = max(cnt, max(friends))

    return user_friends, cnt+1
