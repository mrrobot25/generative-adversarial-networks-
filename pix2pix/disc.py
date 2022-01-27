import torch 
import torch.nn as nn

class CNN_block(nn.Module):
	def __init__(self,in_channels, out_channels, stride = 2):
		super(CNN_block,self).__init__()
		self.conv = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, 4,
					stride, bias=False, padding_mode='reflect'),
		nn.BatchNorm2d(out_channels),
		nn.LeakyReLU(0.2)
				)

	def forward(self,x):
		return self.conv(x)



class Discriminator(nn.Module):
	def __init__(self, in_channels=3, out_channels=[64, 128, 256, 512]):
		super(Discriminator,self).__init__()
		
		self.first = nn.Sequential(
			nn.Conv2d(in_channels*2, out_channels[0], kernel_size=4,stride=2, 
								padding=1, padding_mode='reflect'),
			nn.LeakyReLU(0.2))

		layers = []
		in_channels = out_channels[0]
		for channel in out_channels[1:]:
			layers.append(
					CNN_block(in_channels, channel,
				 		stride=1 if channel == out_channels[-1] else 2)					
					)
			
			in_channels = channel
		layers.append(
				nn.Conv2d(in_channels, 1, kernel_size=4,stride=1, padding=1,
					padding_mode='reflect'))

		self.model = nn.Sequential(*layers)
		
	def forward(self,x,y):
		x = torch.cat([x,y], dim=1)
		print('before first',x.shape)
		x = self.first(x)
		print('after first',x.shape)
		return self.model(x)

def test():
	x = torch.randn((1,3,256,256))
	y = torch.randn((1,3,256,256))

	model = Discriminator()	
	result = model(x,y)
	print(result.shape)

test()	
