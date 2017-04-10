import numpy as np
import torch

def create_user_features(numberOfLatentFeatures, numberOfUsers):
    userFeatsWeights = np.random.uniform(low=0.0, high=1.0, size=(numberOfUsers,numberOfLatentFeatures))
    userFeatsWeights = torch.autograd.Variable(torch.Tensor(userFeatsWeights), requires_grad=True)
    
    userBiasTerm = np.random.uniform(low=0.0, high=1.0, size=(numberOfUsers,1))
    userBiasTerm = torch.autograd.Variable(torch.Tensor(userBiasTerm), requires_grad=True)
    
    #userFeatsWeights = torch.cat((userFeatsWeights,userBiasTerm), 1)
    
    return (userFeatsWeights, userBiasTerm)
    
def create_item_features(numberOfLatentFeatures, numberOfItems):
    itemFeatsWeights = np.random.uniform(low=0.0, high=1.0, size=(numberOfItems,numberOfLatentFeatures))
    itemFeatsWeights = torch.autograd.Variable(torch.Tensor(itemFeatsWeights), requires_grad=True)
    
    ones = torch.autograd.Variable(torch.ones(numberOfItems,1), requires_grad=False)
    
    #itemFeatsWeights = torch.cat((itemFeatsWeights,ones), 1)
    
    return itemFeatsWeights

# For the moment, I need this function because PyTorch sparse tensors features are not very good.
def create_sub_matrix(mtx, indexes, dim=0):
    l = []
    
    for idx in indexes:
        l.append(mtx[idx:(idx+1)])
        
    return torch.cat(l, dim)