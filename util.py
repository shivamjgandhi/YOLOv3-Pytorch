from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np 
import cv2

"""

predict_transform takes in 5 params
- prediction (out output)
- inp_dim is the input dimensions
- anchors
- num_classes
- optional CUDA flag

"""

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
	"""

	This function takes a detection feature map and turns it into a 
	2-D tensor, where each row of the tensor corresponds to attributes
	of a bounding box

	"""

	batch_size = prediction.size(0)
	stride = inp_dim // prediction.size(2)
	grid_size = inp_dim // stride
	bbox_attrs = 5 + num_classes
	num_anchors = len(anchors)

	print(prediction.size())
	prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
	prediction = prediction.transpose(1,2).contiguous()
	prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

	anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

	# Sigmoid the center_X, center_Y, and object confidence
	prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
	prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
	prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

	# Add the center offsets
	grid = np.arange(grid_size)
	a,b = np.meshgrid(grid, grid)

	x_offset = torch.FloatTensor(a).view(-1,1)
	y_offset = torch.FloatTensor(b).view(-1,1)

	if CUDA:
		x_offset = x_offset.cuda()
		y_offset = y_offset.cuda()

	x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1,2).unsqueeze(0)

	prediction[:,:,:2] += x_y_offset

	# Apply the anchors to the dimensions of the bounding box

	# log space transform height and the width
	anchors = torch.FloatTensor(anchors)

	if CUDA:
		anchors = anchors.cuda()

	anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
	prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

	# Apply sigmoid activation to the class scores
	prediction[:,:,5:(5+num_classes)] = torch.sigmoid((prediction[:,:,5:(5+num_classes)]))

	# Last, want to resize the detections map to size of input image
	prediction[:,:,:4] *= stride

	return prediction
