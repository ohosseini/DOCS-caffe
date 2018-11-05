import sys
import os
import numpy as np
import argparse

def parse_args():
	parser = argparse.ArgumentParser(description='Deep Object Co-Segmentation (DOCS) - Train script')
	parser.add_argument('gpu', metavar='GPU', type=int, help='gpu-id')
	parser.add_argument('--steps', metavar='STEPS', type=int, default=100000, help='number of solver steps.')
	parser.add_argument('--weights', metavar='WEIGHTS', help='weights for initializing the network.')
	return parser.parse_args()

def main():
	args = parse_args()

	sys.path.insert(0,os.path.abspath('../../python/caffe/'))
	sys.path.insert(0,os.path.abspath('../../python/'))
	import caffe

	caffe.set_mode_gpu()
	caffe.set_device(args.gpu)

	solver = caffe.AdamSolver('solver.prototxt')

	if args.weights is not None:
		solver.net.copy_from(args.weights)

	solver.step(args.steps)

if __name__ == '__main__':
	main()
