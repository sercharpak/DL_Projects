from torch import Tensor
import matplotlib.pyplot as plt
import math

def convert_to_one_hot_labels(input, target):
    """ THIS FUNCTION WAS TAKEN FROM THE DLC_PRACTICAL_PROLOGUE"""
    tmp = input.new(target.size(0), target.max() + 1).fill_(-1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp

def generate_disk_dataset(n_points,one_hot_labels=True):
    input = Tensor(2,n_points).uniform_(0,1)
    center = Tensor([1/2,1/2]).view(-1,1)
    label = ((input-center).norm(p=2,dim=0) < 1/math.sqrt(2*math.pi)).long()
    if(one_hot_labels):
        label = convert_to_one_hot_labels(input,label).t()
    return input, label

def plot_with_labels(input,labels,ax):
    colors = []
    for b in labels:
        if b==1:
            colors.append('r')
        else:
            colors.append('b')
    ax.scatter(input[0,:],input[1,:],color = colors)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
