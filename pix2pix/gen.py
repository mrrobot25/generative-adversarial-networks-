import torch
import torch.nn as nn

class UBlock(nn.Module):
	def __init__(self, in_channels, out_channels,act='relu', isdown=True, dropout=False)
		super(UBlock,self).__init__()
		self.conv = nn.Sequential(
				if isdown:
					nn.Conv2d(in_channels, out_channels,4,2,1,
						bias=False, padding_mode='reflect')
				else:
					nn.ConvTranspose2d(in_channels, out_channels,
						4,2,1, bias=False)
				
				nn.BatchNorm2d(out_channels)
				nn.ReLU() if act == 'relu' else nn.LeakyReLU(0.2)	
					)
		self.use_dropout = dropout
		self.dropout = nn.Dropout(0.5)

	def forward(self,x):
		self.conv(x)
		return self.dropout(x) if self.use_dropout else x
			


	
class Generator(nn.Module):
	def __init__(self, in_chaennels=3, features=64)
		super(Generator, self).__init__()
		
		self.first_down = nn.Conv2d(in_channels, features, 4,2,1, 
							padding_mode='reflect')

		self.down1 = UBlock(features, features*2, down=True, act='leaky',
							dropout=False)
		self.down2 = UBlock(features*2, features*4, down=True, act='leaky',
							dropout=False)
		self.down3 = UBlock(features*4, features*8, down=True, act='leaky',
							dropout=False)
		self.down4 = UBlock(features*8, features*8, down=True, act='leaky',
							dropout=False)
		self.down5 = UBlock(features*8, features*8, down=True, act='leaky',
							dropout=False)
		self.down6 = UBlock(features*8, features*8, down=True, act='leaky',
							dropout=False)
		self.bottom = nn.Sequential(
				nn.Conv2d(features*8, features*8, 4, 2, 1,
						padding_mode='reflect'),
				nn.ReLU()					
				)

		self.up1 = UBlock(features*8, features*8, down=False, act='relu',
							dropout=True)
		self.up2 = UBlock(features*8*2, features*8, down=False, act='relu',
							dropout=True)
		self.up3 = UBlock(features*8*2, features*8, down=False, act='relu',
							dropout=True)
		self.up4 = UBlock(features*8*2, features*8, down=False, act='relu',
							dropout=False)
		self.up5 = UBlock(features*8*2, features*4, down=False, act='relu',
							dropout=False)
		self.up6 = UBlock(features*4*2, features*2, down=False, act='relu',
							dropout=False)
		self.up7 = UBlock(features*2*2, features, down=False, act='relu',
							dropout=False)
		self.last_up = nn.Sequential(
				nn.ConvTranspose(features*2, in_chaennels, 4,2,1)
				nn.Tanh()					
					)

	def forward(self, x):
		

		















				
					
