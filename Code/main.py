import numpy as np
import time
import argparse
from dataio import *
from model import *
from train import *
from inf import *

parser = argparse.ArgumentParser(description='PyTorch Implementation of SSR-TVD')
parser.add_argument('--lr_D', type=float, default=4e-4, metavar='LR',
                    help='learning rate of D')
parser.add_argument('--lr_G', type=float, default=1e-4, metavar='LR',
                    help='learning rate of G')
parser.add_argument('--lambda_adv', type=float, default=1e-3, metavar='W',
                    help='weight of adversarial loss')
parser.add_argument('--lambda_mse', type=float, default=1, metavar='W',
                    help='weight of content loss')
parser.add_argument('--lambda_percep', type=float, default=5e-2, metavar='W',
                    help='weight of perceptual loss')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training')
parser.add_argument('--dataset', type=str, default = 'Vortex',
                    help='dataset')
parser.add_argument('--mode', type=str, default='train' ,
                    help='training or inference')
parser.add_argument('--epochs', type=int, default=400, metavar='N',
                    help='number of epochs to train (default: 500)')
parser.add_argument('--croptimes', type=int, default=4, metavar='N',
                    help='number of crop times per data')
parser.add_argument('--scale', type=int, default=4, metavar='N',
                    help='spatial upscaling factor')
parser.add_argument('--crop', type=str, default='yes', metavar='N',
                    help='whether to cop data')
parser.add_argument('--f', type=float, default=0.4, metavar='N',
                    help='ratio of training samples')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 30, 'pin_memory': True} if args.cuda else {}


def main(args):
	if args.mode == 'train':
		DataSet = ScalarDataSet(args)
		DataSet.ReadData()

		Model = SSR()
		Ds = spatial_D()
		Dt = temporal_D()

		if args.cuda:
			Model.cuda()
			Dt.cuda()
			Ds.cuda()
		Model.apply(weights_init_kaiming)
		Ds.apply(weights_init_kaiming)
		Dt.apply(weights_init_kaiming)

		trainGAN(Model,Dt,Ds,args,DataSet)

	elif args.mode == 'inf':
		DataSet = ScalarDataSet(args)
		inf(args,DataSet)

if __name__== "__main__":
    main(args)
