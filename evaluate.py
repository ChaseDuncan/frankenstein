import torch
from model import vaereg

from torch.utils.data import DataLoader
from datasets.data_loader import BraTSDataset


device = torch.device('cuda')
model = vaereg.UNet()
checkpoint = torch.load('checkpoints/vaereg-fulltrain/vaereg-fulltrain', map_location='cuda:0')
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model = model.to(device)

brats_data = BraTSDataset('/data/cddunca2/brats2018/validation/', dims=[128, 128, 128])
dataloader = DataLoader(brats_data, batch_size=1, num_workers=0)

