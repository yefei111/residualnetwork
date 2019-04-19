import torch
from torch import nn
import torch.autograd as autograd
class RPBLOCK(nn.Module):
	def __init__(self, pool_winsize, medium_channels, shortcut = None):
		super(RPBLOCK, self).__init__()
		self.basic = nn.Sequential(
			nn.Conv2d(1, medium_channels, 3, 1, 1, bias = False),
			nn.BatchNorm2d(medium_channels),
			nn.Conv2d(medium_channels, 1, 3, 1, 1, bias=False),
			nn.BatchNorm2d(1),
			)
		self.shortcut = shortcut
		self.maxpooling = nn.MaxPool2d((pool_winsize, 1), stride=(2, 1))

	def forward(self, x):
		out = self.basic(x)
		residual = x if self.shortcut is None else self.shortcut(x)
		out +=residual
		out = self.maxpooling(out)
		return out

class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.convpre = nn.Conv2d(1, 1, (50, 1), stride=(2, 1))
		self.layermain = nn.Sequential(
			RPBLOCK(10,4), 
			RPBLOCK(12,8),
			RPBLOCK(14,16),
			RPBLOCK(16,32),
		) # 7,1,18,6
		self.classfier = nn.Sequential(
			nn.Linear(18*6, 50),
			nn.Linear(50, 25),
			nn.Linear(25, 1),
			nn.Sigmoid()
			)
		
	def forward(self, x):
		out = self.convpre(x)
		out = self.layermain(out)
		out = out.view(-1, 18*6)
		return self.classfier(out)


DIM=6
input = autograd.Variable(torch.rand(7, 1, 1000, DIM))
disnetwork = Discriminator()
out = disnetwork(input)
print(out.size())





