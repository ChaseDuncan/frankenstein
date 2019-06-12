import torch
import torch.optim as optim

from losses.cross_entropy import CrossEntropyLoss
from model.btseg import BraTSSegmentation
from datasets.test_data_loader import BraTSDataset

data_dir = "../../../data/miccai-brats-2018-data-training/"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

testloader = BraTSDataset(data_dir)
model = BraTSSegmentation(input_channels=1) 
loss = CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
model.cuda()
model.train()

for i, data in enumerate(testloader):
    print("Training epoch: " + i)
    optimizer.zero_grad()
    src, target = data
    src.to(device)
    target.to(device)
    output = model(src)
    cur_loss = loss(src, target)
    cur_loss.backward()
    optimizer.step()
    if i > 10:
        break

print("Training complete. Saving model.")
torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 'checkpoints/overfit_model.pt')

