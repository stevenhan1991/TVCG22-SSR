import numpy as np
import torch
import time
from skimage.transform import resize
from model import *


def inf(args,dataset):
	model =  SSR()
	model.load_state_dict(torch.load('../Exp/'+args.dataset+'-'+str(args.epochs)+'.pth'))
	if args.cuda:
		model.cuda()
	x = time.time()
	for i in range(0,dataset.total_samples):
		print(i+1)
		low = np.zeros((1,1,dataset.dim[0]//args.scale,dataset.dim[1]//args.scale,dataset.dim[2]//args.scale))
		d = np.fromfile(dataset.data_path+'{:04d}'.format(i+1)+'.dat',dtype='<f')
		d = 2*(d-np.min(d))/(np.max(d)-np.min(d))-1
		d = d.reshape(dataset.dim[2],dataset.dim[1],dataset.dim[0]).transpose()
		gt = np.copy(d)
		d = resize(d,(dataset.dim[0]//args.scale,dataset.dim[1]//args.scale,dataset.dim[2]//args.scale),order=3)
		d = 2*(d-np.min(d))/(np.max(d)-np.min(d))-1
		low[0][0] = d
		low = torch.FloatTensor(low)
		if args.cuda:
			low = low.cuda()

		if args.dataset == 'Vortex':
			with torch.no_grad():
				high = model(low)
				high = high.cpu().detach().numpy()
		elif args.dataset == 'HR':
			high = concatsubvolume(model,low,[60,60,30],args)
		elif args.dataset == 'H':
			high = concatsubvolume(model,low,[30,62,62],args)
		high = high.flatten('F')
		high = np.asarray(high,dtype='<f')
		high.tofile('../Result/'+args.dataset+'/'+'SSR-'+'{:04d}'.format(i+1)+'.dat',format='<f')
	y = time.time()
	print((y-x)/dataset.total_samples)

def concatsubvolume(G,data,win_size,args):
	x,y,z = data.size()[2],data.size()[3],data.size()[4]
	w = np.zeros((args.scale*win_size[0],args.scale*win_size[1],args.scale*win_size[2]))
	for i in range(args.scale*win_size[0]):
		for j in range(args.scale*win_size[1]):
			for k in range(args.scale*win_size[2]):
				dx = min(i,args.scale*win_size[0]-1-i)
				dy = min(j,args.scale*win_size[1]-1-j)
				dz = min(k,args.scale*win_size[2]-1-k)
				d = min(min(dx,dy),dz)+1
				w[i,j,k] = d
	w = w/np.max(w)
	avI = np.zeros((args.scale*x,args.scale*y,args.scale*z))
	pmap= np.zeros((args.scale*x,args.scale*y,args.scale*z))
	avk = 4
	for i in range((avk*x-win_size[0])//win_size[0]+1):
		for j in range((avk*y-win_size[1])//win_size[1]+1):
			for k in range((avk*z-win_size[2])//win_size[2]+1):
				si = (i*win_size[0]//avk)
				ei = si+win_size[0]
				sj = (j*win_size[1]//avk)
				ej = sj+win_size[1]
				sk = (k*win_size[2]//avk)
				ek = sk+win_size[2]
				if ei>x:
					ei= x
					si=ei-win_size[0]
				if ej>y:
					ej = y
					sj = ej-win_size[1]
				if ek>z:
					ek = z
					sk = ek-win_size[2]
				d = data[:,:,si:ei,sj:ej,sk:ek]
				with torch.no_grad():
					result = G(d)
				k = np.multiply(result[0][0].cpu().detach().numpy(),w)
				avI[args.scale*si:args.scale*ei,args.scale*sj:args.scale*ej,args.scale*sk:args.scale*ek] += w
				pmap[args.scale*si:args.scale*ei,args.scale*sj:args.scale*ej,args.scale*sk:args.scale*ek] += k
	high = np.divide(pmap,avI)
	return high