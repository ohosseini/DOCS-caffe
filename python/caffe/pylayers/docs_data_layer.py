import caffe
import numpy as np
import cv2
from random import shuffle
import scipy.io as sio
import matplotlib.pylab as plt

class DOCSDataLayer(caffe.Layer):

	def setup(self, bottom, top):
		
		layer_params = eval(self.param_str)

		input_size = layer_params['input_size']

		rgb_means = layer_params['rgb_means']
		rgb_means = np.asarray([float(x) for x in rgb_means.split()], np.float32)
		layer_params['bgr_means'] = rgb_means[::-1] # Get the input from prototxt as RGB but flip it to BGR because cv2 reads in BGR

		# Create data loader object.
		self.data_loader = DataLoader(layer_params)

		# Reshape only once since we know all the shapes before hand.
		top[0].reshape(1, 3, input_size, input_size) # RGB Image A input

		# Reshape only once since we know all the shapes before hand.
		top[1].reshape(1, 3, input_size, input_size) # RGB Image B input

		# Reshape the gt A output.
		top[2].reshape(1, 1, input_size, input_size)

		# Reshape the gt B output.
		top[3].reshape(1, 1, input_size, input_size)

	def reshape(self, bottom, top):
		# No reshape because we already did it in setup (pre-defined).
		pass


	def forward(self, bottom, top):
		image_a, image_b, gt_a, gt_b = self.data_loader.load_next_image()
		top[0].data[...] = image_a
		top[1].data[...] = image_b
		top[2].data[...] = gt_a
		top[3].data[...] = gt_b


	def backward(self, top, propagate_down, bottom):
		# No Backpropagation
		pass


class DataLoader(object):

	def __init__(self, layer_params):
		self.img_list_file = layer_params['image_list_file']
		self.input_size = layer_params['input_size']
		self.data_folder = layer_params['data_path']
		self.bgr_means = np.array(layer_params['bgr_means'], dtype=np.float32)

		# Get file with list of images.
		list_file = self.data_folder + self.img_list_file

		# Read all the file names.
		with open(list_file) as f:
			self.file_list = f.read().splitlines()         

		shuffle(self.file_list) # Shuffle the list

		self.cur = 0  # current image

	"""
	Load a pair of images and their co-segmentation GTs.
	The images and their GTs are concatenated (side-by-side) and in this function we split them into two. 
	We assume that the size is fixed in the dataset (width: input_size*2 , height: input_size).
	"""
	def load_next_image(self):

		if self.cur == len(self.file_list):
			self.cur = 0
			shuffle(self.file_list)

		# Load the pair
		im = cv2.imread(self.data_folder + 'images/' + self.file_list[self.cur] + '.jpg') #NOTE: the extension
		im = im.astype(np.float32)
		im -= self.bgr_means

		# split the image into two images
		imgs = []
		for i in range(2):
			imgs.append(im[:,i*self.input_size:(i+1)*self.input_size].transpose((2,0,1)))

		# Get the GT Object Coordinates data
		gt = cv2.imread(self.data_folder + 'labels/' + self.file_list[self.cur] + '.png') #NOTE: the extension
		gt[gt==255]=0
		gt[gt>0]=1

		# split the gt image into two images
		gts = []
		for i in range(2):
			gts.append(gt[:,i*self.input_size:(i+1)*self.input_size,0].transpose((0,1)))

		self.cur += 1
		return imgs[0], imgs[1], gts[0], gts[1]

