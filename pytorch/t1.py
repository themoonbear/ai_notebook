#!/usr/bin/env python
# coding=utf-8
import torch
import numpy as np

np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()

print(
    '\n numpy array:', np_data,
    '\n torch tensor:', torch_data,
    '\ntensor to array:', tensor2array,
)

data = [-1,-2,1,2]
tensor = torch.FloatTensor(data)
print(
    '\nabs',
    '\nnumpy:', np.abs(data),
    '\ntorch:', torch.abs(tensor)
)

print(
    '\nsin',
    '\nnumpy:', np.sin(data),
    '\ntorch:', torch.sin(tensor)
)

data = [[1,2],[3,4]]
tensor = torch.FloatTensor(data)
print(
    '\nmatrix multiplication (matmul)',
    '\nnumpy: ', np.matmul(data, data),
    '\ntorch: ', torch.mm(tensor, tensor)
)

data = np.array(data)
print(
    '\nmatrix multiplication (dot)',
    '\nnumpy: ', data.dot(data),
    # '\ntorch: ', tensor.dot(tensor)
)
