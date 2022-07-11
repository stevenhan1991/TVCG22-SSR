import numpy as np
import torch
import skimage 
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage import data,img_as_float
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os

class ScalarDataSet():
	def __init__(self,args):
		self.dataset = args.dataset
		self.scale = args.scale
		self.f = args.f
		self.crop = args.crop
		self.croptimes = args.croptimes
		if self.dataset == 'HR':
			self.dim = [480,720,120]
			self.total_samples = 100
			self.cropsize = [30,40,15]
			self.data_path = '../Data/jet_hr_'                
		elif self.dataset == 'H':
			self.dim = [600,248,248]
			self.total_samples = 100
			self.cropsize = [32,32,32]
			self.data_path = '../Data/GT-'
		elif self.dataset == 'Vortex':
			self.dim = [128,128,128]
			self.total_samples = 90
			self.cropsize = [16,16,16]
			self.data_path = '../Data/vorts'


		self.samples = int(self.total_samples*self.f)

	def ReadData(self):
		self.high = []
		self.low = []
		high = np.zeros((1,self.dim[0],self.dim[1],self.dim[2]))
		low = np.zeros((1,self.dim[0]//self.scale,self.dim[1]//self.scale,self.dim[2]//self.scale))
		for i in range(1,self.samples+1):
			print(i)
			d = np.fromfile(self.data_path+'{:04d}'.format(i)+'.dat',dtype='<f')
			d = 2*(d-np.min(d))/(np.max(d)-np.min(d))-1
			d = d.reshape(self.dim[2],self.dim[1],self.dim[0]).transpose()
			high[0] = d
			self.high.append(high)
			d = resize(d,(self.dim[0]//self.scale,self.dim[1]//self.scale,self.dim[2]//self.scale),order=3)
			d = 2*(d-np.min(d))/(np.max(d)-np.min(d))-1
			low[0] = d
			self.low.append(low)

		self.high = np.asarray(self.high)
		self.low = np.asarray(self.low)

	def GetData(self):
		if self.crop == 'yes':
			low = []
			high = []
			for k in range(0,self.samples-2):
				n = 0
				while n < self.croptimes:
					if self.dim[0]//self.scale == self.cropsize[0]:
						x = 0
					else:
						x = np.random.randint(0,self.dim[0]//self.scale-self.cropsize[0])

					if self.dim[1]//self.scale == self.cropsize[1]:
						y = 0
					else:
						y = np.random.randint(0,self.dim[1]//self.scale-self.cropsize[1])

					if self.dim[1]//self.scale == self.cropsize[1]:
						z = 0
					else:
						z = np.random.randint(0,self.dim[2]//self.scale-self.cropsize[2])

					c0 = self.low[k][:,x:x+self.cropsize[0],y:y+self.cropsize[1],z:z+self.cropsize[2]]
					c1 = self.low[k+1][:,x:x+self.cropsize[0],y:y+self.cropsize[1],z:z+self.cropsize[2]]
					c2 = self.low[k+2][:,x:x+self.cropsize[0],y:y+self.cropsize[1],z:z+self.cropsize[2]]
					low.append(np.concatenate((c0,c1,c2),axis=0))

					c0 = self.high[k][:,self.scale*x:self.scale*(x+self.cropsize[0]),self.scale*y:self.scale*(y+self.cropsize[1]),self.scale*z:self.scale*(z+self.cropsize[2])]
					c1 = self.high[k+1][:,self.scale*x:self.scale*(x+self.cropsize[0]),self.scale*y:self.scale*(y+self.cropsize[1]),self.scale*z:self.scale*(z+self.cropsize[2])]
					c2 = self.high[k+2][:,self.scale*x:self.scale*(x+self.cropsize[0]),self.scale*y:self.scale*(y+self.cropsize[1]),self.scale*z:self.scale*(z+self.cropsize[2])]
					high.append(np.concatenate((c0,c1,c2),axis=0))

					n+= 1
			low = np.asarray(low)
			high = np.asarray(high)
			low = torch.FloatTensor(low)
			high = torch.FloatTensor(high)
		else:
			low = []
			high = []
			for k in range(0,self.samples-2):
				low.append(np.concatenate((self.low[k],self.low[k+1],self.low[k+2]),axis=0))
				high.append(np.concatenate((self.high[k],self.high[k+1],self.high[k+2]),axis=0))
			low = np.asarray(low)
			high = np.asarray(high)
			low = torch.FloatTensor(low)
			high = torch.FloatTensor(high)
		dataset = torch.utils.data.TensorDataset(low,high)
		train_loader = DataLoader(dataset=dataset,batch_size=1, shuffle=True)
		return train_loader