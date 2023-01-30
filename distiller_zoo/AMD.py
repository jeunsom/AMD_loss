from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class AngularPenaltyLoss(nn.Module):

	def __init__(self, out_features=1, loss_type='sphereface', eps=1e-7, s=None, m=None):

	#'''
	#Angular Penalty Softmax Loss
	#Three 'loss_types' available: ['sphereface']
	#These losses are described in the following papers: 
	#SphereFace: https://arxiv.org/abs/1704.08063
	#'''
		super(AngularPenaltyLoss, self).__init__()

		loss_type = loss_type.lower()
		assert loss_type in  ['sphereface']

		m_param = 1.35 #margin_param
		if loss_type == 'sphereface':
			self.s = 64.0 if not s else s
			self.m = m_param if not m else m
		self.loss_type = loss_type

		self.eps = eps


	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), 2)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am


	def forward(self, g_s, g_t):
		return [self.amdloss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]


	def amdloss(self, x_s, x_t,labels=1): 
		wfs = self.attention_map(x_s) 
		wft = self.attention_map(x_t) 
		wfs = wfs.view(wfs.size(0),-1) 
		wft = wft.view(wft.size(0),-1)  

		if self.loss_type == 'sphereface':
			numerator_s = self.s * torch.cos(self.m * torch.acos(torch.clamp(wfs, -1.+self.eps, 1-self.eps)))
			numerator_t = self.s * torch.cos(self.m * torch.acos(torch.clamp(wft, -1.+self.eps, 1-self.eps)))


		wfs_1 =  (1-wfs).float() 
		excl = wfs_1
		numerator_s1 = torch.exp(numerator_s) 
		temp1s = torch.exp(self.s * excl)

		denominator1 = numerator_s1 + temp1s

		L1 = numerator_s - torch.log(denominator1)

		wft_1 =  (1-wft).float()  
		excl2 = wft_1

		numerator_t1 = torch.exp(numerator_t) 
		temp2t = torch.exp(self.s * excl2) 


		denominator2 = numerator_t1 + temp2t

		L2 = numerator_t - torch.log(denominator2)


		norm_L1 = F.normalize(L1, p=2, dim=1)
		norm_L2 = F.normalize(L2, p=2, dim=1)

		norm_L1s = F.normalize(torch.log(numerator_s1), p=2, dim=1) 
		norm_L2t = F.normalize(torch.log(numerator_t1), p=2, dim=1) 

		norm_L1s2 = F.normalize(torch.log(temp1s), p=2, dim=1) 
		norm_L2t2 = F.normalize(torch.log(temp2t), p=2, dim=1) 

		loss = F.mse_loss(norm_L1, norm_L2) + F.mse_loss(norm_L1s, norm_L2t) + F.mse_loss(norm_L1s2, norm_L2t2)
		loss = loss / 3.0

		return loss




class AngularPenaltyLoss4b(nn.Module):

	def __init__(self, out_features=1, loss_type='sphereface', eps=1e-7, s=None, m=None):

	#'''
	#Angular Penalty Softmax Loss
	#Three 'loss_types' available: ['sphereface']
	#These losses are described in the following papers: 
	#SphereFace: https://arxiv.org/abs/1704.08063
	#'''
		super(AngularPenaltyLoss4b, self).__init__()

		loss_type = loss_type.lower()
		assert loss_type in  ['sphereface']

		m_param = 1.35 #margin_param
		if loss_type == 'sphereface':
			self.s = 64.0 if not s else s
			self.m = m_param if not m else m
		self.loss_type = loss_type

		self.eps = eps


	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), 2)		
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am


	def forward(self, g_s, g_t):
		return [self.amdloss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]


	def amdloss(self, x_s, x_t,labels=1): 

		wfss = self.attention_map(x_s)
		wftt = self.attention_map(x_t)

		h_s = wfss.shape[2] / 2
		w_s = wfss.shape[3] / 2

		h_s1 = wfss.shape[2] / 4
		w_s1 = wfss.shape[3] / 4


		wfs = wfss[:,:, :int(h_s), :int(w_s)]
		wfs1 = wfss[:,:, :int(h_s), int(w_s):]
		wfs2 = wfss[:,:, int(h_s):, :int(w_s)]
		wfs3 = wfss[:,:, int(h_s):, int(w_s):]
		wfs4 = wfss

		wft = wftt[:,:, :int(h_s), :int(w_s)]
		wft1 = wftt[:,:, :int(h_s), int(w_s):]
		wft2 = wftt[:,:, int(h_s):, :int(w_s)]
		wft3 = wftt[:,:, int(h_s):, int(w_s):]
		wft4 = wftt


		wfs = wfs.reshape((wfs.shape[0],-1))
		wft = wft.reshape((wfs.shape[0],-1))
		wfs1 = wfs1.reshape((wfs.shape[0],-1))
		wft1 = wft1.reshape((wfs.shape[0],-1))
		wfs2 = wfs2.reshape((wfs.shape[0],-1))
		wft2 = wft2.reshape((wfs.shape[0],-1))
		wfs3 = wfs3.reshape((wfs.shape[0],-1))
		wft3 = wft3.reshape((wfs.shape[0],-1))
		wfs4 = wfs4.reshape((wfs.shape[0],-1)) 
		wft4 = wft4.reshape((wfs.shape[0],-1))

		if self.loss_type == 'sphereface':
			numerator_s = self.s * torch.cos(self.m * torch.acos(torch.clamp(wfs, -1.+self.eps, 1-self.eps)))
			numerator_t = self.s * torch.cos(self.m * torch.acos(torch.clamp(wft, -1.+self.eps, 1-self.eps)))

			numerator_s1 = self.s * torch.cos(self.m * torch.acos(torch.clamp(wfs1, -1.+self.eps, 1-self.eps)))
			numerator_t1 = self.s * torch.cos(self.m * torch.acos(torch.clamp(wft1, -1.+self.eps, 1-self.eps)))

			numerator_s2 = self.s * torch.cos(self.m * torch.acos(torch.clamp(wfs2, -1.+self.eps, 1-self.eps)))
			numerator_t2 = self.s * torch.cos(self.m * torch.acos(torch.clamp(wft2, -1.+self.eps, 1-self.eps)))

			numerator_s3 = self.s * torch.cos(self.m * torch.acos(torch.clamp(wfs3, -1.+self.eps, 1-self.eps)))
			numerator_t3 = self.s * torch.cos(self.m * torch.acos(torch.clamp(wft3, -1.+self.eps, 1-self.eps)))

			numerator_s4 = self.s * torch.cos(self.m * torch.acos(torch.clamp(wfs4, -1.+self.eps, 1-self.eps)))
			numerator_t4 = self.s * torch.cos(self.m * torch.acos(torch.clamp(wft4, -1.+self.eps, 1-self.eps)))

		wfs_1 =  (1-wfs).float()

		numerator_sa1 = torch.exp(numerator_s) 
		temp1s1 = torch.exp(self.s * excl) 
		denominator1 = numerator_sa1 + temp1s1


		L1a = numerator_s - torch.log(denominator1)

		wft_1 =  (1-wft).float() 
		excl2 = wft_1 

		numerator_ta1 = torch.exp(numerator_t) 
		temp2t1 = torch.exp(self.s * excl2)
		denominator2 = numerator_ta1 + temp2t1


		L2a = numerator_t - torch.log(denominator2)

		#########################
		wfs_1 =  (1-wfs1).float()
		excl = wfs_1 

		numerator_sa2 = torch.exp(numerator_s1) 
		temp1s2 = torch.exp(self.s * excl) 
		denominator1 = numerator_sa2 + temp1s2


		L1b = numerator_s1 - torch.log(denominator1)

		wft_1 =  (1-wft1).float() 
		excl2 = wft_1 

		numerator_ta2 = torch.exp(numerator_t1) 
		temp2t2 = torch.exp(self.s * excl2)
		denominator2 = numerator_ta2 + temp2t2


		L2b = numerator_t1 - torch.log(denominator2)

		#########################
		wfs_1 =  (1-wfs2).float() 
		excl = wfs_1 

		numerator_sa3 = torch.exp(numerator_s2) 
		temp1s3 = torch.exp(self.s * excl) 
		denominator1 = numerator_sa3 + temp1s3

		L1c = numerator_s2 - torch.log(denominator1)

		wft_1 =  (1-wft2).float() 
		excl2 = wft_1 

		numerator_ta3 = torch.exp(numerator_t2) 
		temp2t3 = torch.exp(self.s * excl2)
		denominator2 = numerator_ta3 + temp2t3



		L2c = numerator_t2 - torch.log(denominator2)

		#########################
		wfs_1 =  (1-wfs3).float()
		excl = wfs_1 

		numerator_sa4 = torch.exp(numerator_s3) 
		temp1s4 = torch.exp(self.s * excl) 
		denominator1 = numerator_sa4 + temp1s4


		L1d = numerator_s3 - torch.log(denominator1)

		wft_1 =  (1-wft3).float()

		excl2 = wft_1
		numerator_ta4 = torch.exp(numerator_t3) 
		temp2t4 = torch.exp(self.s * excl2)
		denominator2 = numerator_ta4 + temp2t4


		L2d = numerator_t3 - torch.log(denominator2)

		#########################
		wfs_1 =  (1-wfs4).float()
		numerator_sa5 = torch.exp(numerator_s4) 
		temp1s5 = torch.exp(self.s * excl) 
		denominator1 = numerator_sa5 + temp1s5

		L1e = numerator_s4 - torch.log(denominator1)

		wft_1 =  (1-wft4).float()
		excl2 = wft_1

		numerator_ta5 = torch.exp(numerator_t4) 
		temp2t5 = torch.exp(self.s * excl2)
		denominator2 = numerator_ta5 + temp2t5


		L2e = numerator_t4 - torch.log(denominator2)


		#############

		L1s = F.normalize(L1a, p=2, dim=1)
		L2t = F.normalize(L2a, p=2, dim=1)

		L1s1 = F.normalize(L1b, p=2, dim=1)
		L2t1 = F.normalize(L2b, p=2, dim=1)

		L1s2 = F.normalize(L1c, p=2, dim=1)
		L2t2 = F.normalize(L2c, p=2, dim=1)

		L1s3 = F.normalize(L1d, p=2, dim=1)
		L2t3 = F.normalize(L2d, p=2, dim=1)

		L1sa = F.normalize(torch.log(numerator_sa1), p=2, dim=1)
		L2ta = F.normalize(torch.log(numerator_ta1), p=2, dim=1)

		L1s1a = F.normalize(torch.log(numerator_sa2), p=2, dim=1)
		L2t1a = F.normalize(torch.log(numerator_ta2), p=2, dim=1)

		L1s2a = F.normalize(torch.log(numerator_sa3), p=2, dim=1)
		L2t2a = F.normalize(torch.log(numerator_ta3), p=2, dim=1)

		L1s3a = F.normalize(torch.log(numerator_sa4), p=2, dim=1)
		L2t3a = F.normalize(torch.log(numerator_ta4), p=2, dim=1)

		L1sb = F.normalize(torch.log(temp1s1), p=2, dim=1)
		L2tb = F.normalize(torch.log(temp2t1), p=2, dim=1)

		L1s1b = F.normalize(torch.log(temp1s2), p=2, dim=1)
		L2t1b = F.normalize(torch.log(temp2t2), p=2, dim=1)

		L1s2b = F.normalize(torch.log(temp1s3), p=2, dim=1)
		L2t2b = F.normalize(torch.log(temp2t3), p=2, dim=1)

		L1s3b = F.normalize(torch.log(temp1s4), p=2, dim=1)
		L2t3b = F.normalize(torch.log(temp2t4), p=2, dim=1)



		norm_L1e = F.normalize(L1e, p=2, dim=1)
		norm_L2e = F.normalize(L2e, p=2, dim=1)

		norm_L1e1 = F.normalize(torch.log(numerator_sa5), p=2, dim=1)
		norm_L2e1 = F.normalize(torch.log(numerator_ta5), p=2, dim=1)

		norm_L1e2 = F.normalize(torch.log(temp1s5), p=2, dim=1)
		norm_L2e2 = F.normalize(torch.log(temp2t5), p=2, dim=1)


		loss1 = (F.mse_loss(L1s, L2t) + F.mse_loss(L1s1, L2t1) + F.mse_loss(L1s2, L2t2) + F.mse_loss(L1s3, L2t3)) / 4.0
		loss1a = (F.mse_loss(L1sa, L2ta) + F.mse_loss(L1s1a, L2t1a) + F.mse_loss(L1s2a, L2t2a) + F.mse_loss(L1s3a, L2t3a)) / 4.0
		loss1b = (F.mse_loss(L1sb, L2tb) + F.mse_loss(L1s1b, L2t1b) + F.mse_loss(L1s2b, L2t2b) + F.mse_loss(L1s3b, L2t3b)) / 4.0

		loss2 = F.mse_loss(norm_L1e, norm_L2e)  +  F.mse_loss(norm_L1e1, norm_L2e1) +  F.mse_loss(norm_L1e2, norm_L2e2)

		loss = (loss1 + loss1a + loss1b) / 3.0 * 0.2 + loss2 / 3.0 *0.8


		return loss



