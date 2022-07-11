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


def trainGAN(G,Dt,Ds,args,dataset):
	loss = open('../Exp/'+'loss-'+args.dataset+'.txt','w')
	device = torch.device("cuda:0" if args.cuda else "cpu")
	optimizer_G = optim.Adam(G.parameters(), lr=args.lr_G,betas=(0.9,0.999))
	optimizer_t_D = optim.Adam(Dt.parameters(), lr=args.lr_D,betas=(0.5,0.999))
	optimizer_s_D = optim.Adam(Ds.parameters(), lr=args.lr_D,betas=(0.5,0.999))
	fake = 0
	real = 1
	t = 0
	criterion = nn.MSELoss()
	critic = 1

	for itera in range(1,args.epochs+1):
		loss_mse = 0
		loss_Dt = 0
		loss_Ds = 0
		
		print("========================")
		print(itera)
		train_loader = dataset.GetData()
		x = time.time()
		for batch_idx,(low,high) in enumerate(train_loader):

			if args.cuda:
				low = low.cuda()
				high = high.cuda()

			for p in G.parameters():
				p.requires_grad = False
			print(high.size())
			################################
			# Update Spatial D network
			################################
			for j in range(1,critic+1):
				optimizer_s_D.zero_grad()
				# train with real
				
				output_real = Ds(high.permute(1,0,2,3,4))
				label_real = torch.ones(output_real[-1].size()).cuda()
				real_loss = criterion(output_real[-1],label_real)

				# train with fake
				fake_data = G(low.permute(1,0,2,3,4))
				label_fake = torch.zeros(output_real[-1].size()).cuda()
				
				output_fake = Ds(fake_data)
				fake_loss = criterion(output_fake[-1],label_fake)
				
				loss_gan = 0.5*(real_loss+fake_loss)
				
				loss_gan.backward()
				loss_Ds += loss_gan.mean().item()
				optimizer_s_D.step()

			#################################
			# Update Temporal D network
			#################################

			for j in range(1,critic+1):
				optimizer_t_D.zero_grad()
				# train with real
				
				output_real = Dt(high)
				label_real = torch.ones(output_real[-1].size()).cuda()
				real_loss = criterion(output_real[-1],label_real)

				# train with fake
				fake_data = G(low.permute(1,0,2,3,4))

				label_fake = torch.zeros(output_real[-1].size()).cuda()
				
				output_fake = Dt(fake_data.permute(1,0,2,3,4))
				fake_loss = criterion(output_fake[-1],label_fake)
				
				loss_gan = 0.5*(real_loss+fake_loss)
				
				loss_gan.backward()
				loss_Dt += loss_gan.mean().item()
				optimizer_t_D.step()
			
			#################################
			# Update G network
			#################################
			for p in G.parameters():
				p.requires_grad = True
			for p in Ds.parameters():
				p.requires_grad = False
			for p in Dt.parameters():
				p.requires_grad = False

			optimizer_G.zero_grad()
			
			fake_data = G(low.permute(1,0,2,3,4))
			fake_data_t = G(low.permute(1,0,2,3,4)).permute(1,0,2,3,4)
			output_t = Dt(fake_data_t)
			output_real_t = Dt(high)

			output_s = Ds(fake_data)
			output_real_s = Ds(high.permute(1,0,2,3,4))

			label_real_t = torch.ones(output_real_t[-1].size()).cuda()
			label_real_s = torch.ones(output_real_s[-1].size()).cuda()

			# adversarial loss
			L_adv = 0.5*(criterion(output_t[-1],label_real_t)+criterion(output_s[-1],label_real_s))

			# perceptual loss
			L_p = criterion(output_s[0],output_real_s[0])+criterion(output_s[1],output_real_s[1])+criterion(output_s[2],output_real_s[2])+criterion(output_s[3],output_real_s[3])

			# content loss
			L_c = criterion(fake_data,high.permute(1,0,2,3,4))

			# total loss
			error = args.lambda_adv*(L_adv)+args.lambda_mse*L_c+args.lambda_percep*L_p

			error.backward()
			loss_mse += error.item()
			optimizer_G.step()

			for p in Ds.parameters():
				p.requires_grad = True
			for p in Dt.parameters():
				p.requires_grad = True

		loss.write("Epochs "+str(itera)+": loss = "+str(loss_mse))
		loss.write('\n')

		y = time.time()

		t += y-x
		if itera%10==0 or itera==1:
			torch.save(G.state_dict(),'../Exp/'+args.dataset+'-'+str(itera)+'.pth')



