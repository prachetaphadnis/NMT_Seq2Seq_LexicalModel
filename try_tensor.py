import torch

if __name__ == '__main__':
	x = torch.rand(3,5, 3)
	print(x)
	print(x[0:3:2])