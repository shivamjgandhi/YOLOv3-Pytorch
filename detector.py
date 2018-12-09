from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np 
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet

# Since detector.py is the file that we'll execute to run our
# detector, we need a command line parser

def arg_parse():
	"""
	Parse arguments tot he detect module

	"""

	parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

	parser.add_argument("--images", dest = 'images', help=
		"Image / Directory containing images to perform detection upon",
		default = "imgs", type = str)
	parser.add_argument("--det", dest = 'det', help = 
		"Image / Directory to store detections to", 
		default = "det", type = str)
	parser.add_argument("--bs", dest = "bs", help = "Batch size", default = 1)
	parser.add_argument("--confidence", dest = "confidence", help = "Object confidence to filter predictions", default = 0.5)
	parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
	parser.add_argument("--cfg", dest = "cfgfile", help = "Config file",
		default = "cfg/yolov3.cfg", type = str)
	parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile",
		default = "C:/Users/Sjgandhi1998/Software/Data/Yolov3", type = str)
	parser.add_argument("--reso", dest = 'reso', help = 
		"Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
		default = "416", type = str)

	return parser.parse_args()

args = arg_parse()
images = args.images
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thresh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

# These are for the files we're going to be working on
num_classes = 80
classes = load_classes("data/coco.names")

# Initialize the network and load weights

# Set up the neural network
print("Loading network...")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

mode.net_info["height"] = args.reso 
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32

# If there's a GPU available put the model on GPU
if CUDA:
	model.cuda()

# Set the model in evaluation mode
model.eval()

# Read the images from the disk, or the images from a dir.

read_dir = time.time()
# Detection phase
try:
	imlist = [osp.join(osp.realpath(','), images, img) for img in os.listdir(images)]
except NotADirectoryError:
	imlist = []
	imlist.appent(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
	print("No file or directory with the name {}".format(images))
	exit()

# If the directory to save the detections, defined by the det flag, doesn't exist, create it
if not os.path.exists(args.det):
	os.makedirs(args.det)

# Use OpenCV to load the images
load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]

# PyTorch variables for images
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

# List containing dimensions of original images
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

if CUDA:
	im_dim_list = im_dim_list.cuda()

# Create the batches
leftover = 0
if (len(im_dim_list) % batch_size):
	leftover = 1

if batch_size != 1:
	num_batches = len(imlist) // batch_size + leftover
	im_batches = [torch.cat((im_batches[i*batch_size: min((i+1)*batch_size,
		len(im_batches))])) for i in range(num_batches)]

	