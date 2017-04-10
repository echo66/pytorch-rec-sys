import torch

def compute_prediction(UFeats, IFeats):
    pred = torch.mm(UFeats, IFeats.t())
    return pred

def compute_mse(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return mse

def compute_se(pred, target):
    se = torch.sum((pred - target) ** 2)
    return se

def compute_mabse(pred, target):
    mabse = torch.mean(torch.abs(pred - target))
    return mabse

def compute_abse(pred, target):
    abse = torch.sum(torch.abs(pred - target))
    return abse

def compute_regl2(tensor):
    regl2 = torch.sum(tensor * tensor)
    return regl2

def compute_mean_regl2(tensor):
    mean_regl2 = torch.mean(tensor * tensor)
    return mean_regl2