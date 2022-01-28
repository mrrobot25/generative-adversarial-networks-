import torch
import torch.nn as nn

class UBlock(nn.Module):
	def __init__(self, in_channels, out_channels, isdown=True, act='relu', dropout=False):
		super(UBlock,self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_channels, out_channels,4,2,1,
				 bias=False, padding_mode='reflect')
			if isdown	
			else nn.ConvTranspose2d(in_channels, out_channels,
				4,2,1, bias=False),
				
			nn.BatchNorm2d(out_channels),
			nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2)
					)
		self.use_dropout = dropout
		self.dropout = nn.Dropout(0.5)

	def forward(self,x):
		x = self.conv(x)
		return self.dropout(x) if self.use_dropout else x
			


class Generator(nn.Module):
	def __init__(self, in_channels=3, features=64):
		super(Generator, self).__init__()
		
		self.first_down = nn.Sequential(
				nn.Conv2d(in_channels, features, 4,2,1, 
							padding_mode='reflect'),
				nn.LeakyReLU(0.2)
			) 

		self.down1 = UBlock(features, features*2, isdown=True, act='leaky',
							dropout=False)
		self.down2 = UBlock(features*2, features*4, isdown=True, act='leaky',
							dropout=False)
		self.down3 = UBlock(features*4, features*8, isdown=True, act='leaky',
							dropout=False)
		self.down4 = UBlock(features*8, features*8, isdown=True, act='leaky',
							dropout=False)
		self.down5 = UBlock(features*8, features*8, isdown=True, act='leaky',
							dropout=False)
		self.down6 = UBlock(features*8, features*8, isdown=True, act='leaky',
							dropout=False)
		self.bottom = nn.Sequential(
				nn.Conv2d(features*8, features*8, 4, 2, 1,
						padding_mode='reflect'),
				nn.ReLU()					
				)

		self.up1 = UBlock(features*8, features*8, isdown=False, act='relu',
							dropout=True)
		self.up2 = UBlock(features*8*2, features*8, isdown=False, act='relu',
							dropout=True)
		self.up3 = UBlock(features*8*2, features*8, isdown=False, act='relu',
							dropout=True)
		self.up4 = UBlock(features*8*2, features*8, isdown=False, act='relu',
							dropout=False)
		self.up5 = UBlock(features*8*2, features*4, isdown=False, act='relu',
							dropout=False)
		self.up6 = UBlock(features*4*2, features*2, isdown=False, act='relu',
							dropout=False)
		self.up7 = UBlock(features*2*2, features, isdown=False, act='relu',
							dropout=False)
		self.last_up = nn.Sequential(
				nn.ConvTranspose2d(features*2, in_channels, 4,2,1),
				nn.Tanh()					
					)

	def forward(self, x):

		dn1 = self.first_down(x)
		dn2 = self.down1(dn1)
		dn3 = self.down2(dn2)
		dn4 = self.down3(dn3)
		dn5 = self.down4(dn4)
		dn6 = self.down5(dn5)
		dn7 = self.down6(dn6)

		bottom = self.bottom(dn7)
		up1 = self.up1(bottom)
		
		up2 = self.up2(torch.cat([up1,dn7],1))
		up3 = self.up3(torch.cat([up2,dn6],1))
		up4 = self.up4(torch.cat([up3,dn5],1))
		up5 = self.up5(torch.cat([up4,dn4],1))
		up6 = self.up6(torch.cat([up5,dn3],1))
		up7 = self.up7(torch.cat([up6,dn2],1))

		last_up = self.last_up(torch.cat([up7,dn1],1))
		return last_up

def test():
	
	x = torch.randn((1,3,256,256))
	model = Generator(in_channels=3, features=64)
	results = model(x)
	print(results.shape)

		

test()
		















				
					
