
import numpy as np
import torch
from torch import nn

class AE(nn.Module):
	
	def __init__(self, input_shape=100, L1=10, L2=5, n_hidden=3, 
				name='', weight_path=''):
		super(AE, self).__init__()
		self.L1 = L1
		self.L2 = L2

		self.encoder = nn.Sequential(
			nn.Linear(input_shape, self.L1),
			nn.ReLU(True),
			nn.Linear(self.L1, self.L2),
			nn.ReLU(True), 
			nn.Linear(self.L2, n_hidden)
			)

		self.decoder = nn.Sequential(
			nn.Linear(n_hidden, self.L2),
			nn.ReLU(True),
			nn.Linear(self.L2, self.L1),
			nn.ReLU(True), 
			nn.Linear(self.L1, input_shape), 
			nn.Tanh()
			)

	def forward(self, x, encoding=False):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		if encoding: # Return hidden later activations
			return encoded
		return decoded

def get_hidden_activations(model, n_hidden=3, n_items=12, device='cuda'):
    ''' Assumes one hot'''
    test_dat = np.eye(n_items)
    #outarr = np.zeros((n_items, n_items))
    hiddenarr = np.zeros((n_items, n_hidden))
    for i in range(n_items):
        x = torch.Tensor(test_dat[i,:]).to(device)
        #out = model.forward(x, encoding=False).detach()
        hidden = model.forward(x, encoding=True).detach()
        hiddenarr[i,:] = hidden.cpu()
    return hiddenarr