from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from torch.optim import lr_scheduler
import torch.nn.parallel
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn as nn
import torch
import torch.autograd as autograd
from torch.nn import init
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init
import time

def BuildResidualBlock(channels,dropout,kernel,depth,bias):
	layers = []
	for i in range(int(depth)):
		layers += [spectral_norm(nn.Conv3d(channels,channels,kernel_size=kernel,padding=kernel//2,bias=bias),eps=1e-4),
		           nn.ReLU(True)]
		if dropout:
			layers += [nn.Dropout(0.5)]
	layers += [spectral_norm(nn.Conv3d(channels,channels,kernel_size=kernel,padding=kernel//2,bias=bias),eps=1e-4),
		       ]
	return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
	def __init__(self,channels,dropout,kernel,depth,bias):
		super(ResidualBlock,self).__init__()
		self.block = BuildResidualBlock(channels,dropout,kernel,depth,bias)

	def forward(self,x):
		out = x+self.block(x)
		return out


def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find("Conv")!=-1:
		init.kaiming_uniform_(m.weight.data)
	elif classname.find("Linear")!=-1:
		init.kaiming_uniform_(m.weight.data)
	elif classname.find("BatchNorm")!=-1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)

def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find("Conv")!=-1:
		init.normal_(m.weight.data, 1.0, 0.02)
	elif classname.find("Linear")!=-1:
		init.normal_(m.weight.data, 1.0, 0.02)
	elif classname.find("BatchNorm")!=-1:
		init.normal_(m.weight.data, 1.0, 0.02)
		init.constant_(m.bias.data, 0.0)


class Block(nn.Module):
	def __init__(self,inchannels,outchannels,dropout,kernel,bias,depth,mode):
		super(Block,self).__init__()
		layers = []
		for i in range(int(depth)):
			layers += [
			           spectral_norm(nn.Conv3d(inchannels,inchannels,kernel_size=kernel,padding=kernel//2,bias=bias),eps=1e-4),
			           nn.InstanceNorm3d(inchannels),
			           nn.LeakyReLU(0.2,inplace=True)]
			if dropout:
				layers += [nn.Dropout(0.5)]
		self.model = nn.Sequential(*layers)

		if mode == 'down':
			self.conv1 = spectral_norm(nn.Conv3d(inchannels,outchannels,4,stride=2,padding=1,bias=bias),eps=1e-4)
			self.conv2 = spectral_norm(nn.Conv3d(inchannels,outchannels,4,stride=2,padding=1,bias=bias),eps=1e-4)
		elif mode == 'up':
			self.conv1 = spectral_norm(nn.ConvTranspose3d(inchannels,outchannels,4,stride=2,padding=1,bias=bias),eps=1e-4)
			self.conv2 = spectral_norm(nn.ConvTranspose3d(inchannels,outchannels,4,stride=2,padding=1,bias=bias),eps=1e-4)
		elif mode == 'same':
			self.conv1 = spectral_norm(nn.Conv3d(inchannels,outchannels,kernel_size=3,stride=1,padding=1,bias=bias),eps=1e-4)

			self.conv2 = spectral_norm(nn.Conv3d(inchannels,outchannels,kernel_size=3,stride=1,padding=1,bias=bias),eps=1e-4)

	def forward(self,x):
		y = self.model(x)
		y = self.conv1(y)
		x = self.conv2(x)
		return y+x


class SSR(nn.Module):
	def __init__(self):
		super(SSR,self).__init__()
		self.b1 = Block(inchannels=1,outchannels=16,dropout=False,kernel=3,bias=False,depth=3,mode='same')
		self.b2 = Block(inchannels=16,outchannels=64,dropout=False,kernel=3,bias=False,depth=3,mode='same')
		self.b3 = Block(inchannels=80,outchannels=128,dropout=False,kernel=3,bias=False,depth=3,mode='same')
		self.b4 = Block(inchannels=128,outchannels=128,dropout=False,kernel=3,bias=False,depth=3,mode='same')
		self.b5 = Block(inchannels=256,outchannels=64,dropout=False,kernel=3,bias=False,depth=3,mode='same')
		self.b6 = Block(inchannels=32,outchannels=8,dropout=False,kernel=3,bias=False,depth=3,mode='same')
		self.b7 = Block(inchannels=40,outchannels=1,dropout=False,kernel=3,bias=False,depth=3,mode='same')
		self.deconv1 = VoxelShuffle(128,128,2)
		self.deconv2 = VoxelShuffle(64,32,2)

	def forward(self,x):
		x1 = F.relu(self.b1(x))
		x2 = F.relu(self.b2(x1))
		x3 = F.relu(self.b3(torch.cat((x1,x2),dim=1)))
		x4 = F.relu(self.deconv1(x3))
		x5 = F.relu(self.b4(x4))
		x6 = F.relu(self.b5(torch.cat((x4,x5),dim=1)))
		x7 = F.relu(self.deconv2(x6))
		x8 = F.relu(self.b6(x7))
		x9 = torch.tanh(self.b7(torch.cat((x7,x8),dim=1)))
		return x9

class spatial_D(nn.Module):

	def __init__(self):
		super(spatial_D,self).__init__()
		self.conv1 = spectral_norm(nn.Conv3d(1,64,4,stride=2,padding=1),eps=1e-4)
		self.leaky = nn.LeakyReLU(0.2,inplace=True)
		self.conv2 = spectral_norm(nn.Conv3d(64,128,4,stride=2,padding=1),eps=1e-4)
		self.conv3 = spectral_norm(nn.Conv3d(128,256,4,stride=2,padding=1),eps=1e-4)
		self.conv4 = spectral_norm(nn.Conv3d(256,1,4,stride=2,padding=1),eps=1e-4)


	def forward(self,x):
		result = []
		x = self.leaky(self.conv1(x))
		result.append(x)
		x = self.leaky(self.conv2(x))
		result.append(x)
		x = self.leaky(self.conv3(x))
		result.append(x)
		x = self.leaky(self.conv4(x))
		result.append(x)
		x = F.avg_pool3d(x,x.size()[2:]).view(-1)
		result.append(x)
		return result


class temporal_D(nn.Module):

	def __init__(self):
		super(temporal_D,self).__init__()
		self.conv1 = spectral_norm(nn.Conv3d(3,64,4,stride=2,padding=1),eps=1e-4)
		self.leaky = nn.LeakyReLU(0.2,inplace=True)
		self.conv2 = spectral_norm(nn.Conv3d(64,128,4,stride=2,padding=1),eps=1e-4)
		self.conv3 = spectral_norm(nn.Conv3d(128,256,4,stride=2,padding=1),eps=1e-4)
		self.conv4 = spectral_norm(nn.Conv3d(256,1,4,stride=2,padding=1),eps=1e-4)


	def forward(self,x):
		result = []
		x = self.leaky(self.conv1(x))
		result.append(x)
		x = self.leaky(self.conv2(x))
		result.append(x)
		x = self.leaky(self.conv3(x))
		result.append(x)
		x = self.leaky(self.conv4(x))
		result.append(x)
		x = F.avg_pool3d(x,x.size()[2:]).view(-1)
		result.append(x)
		return result


