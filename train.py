import torch
import torch.optim as optim

from losses.dice import DiceLoss
from model.btseg import BraTSSegmentation
from datasets.test_data_loader import BraTSDataset

data_dir = "/home/chase/data/miccai-brats-2018-data-training/"
#model_name = 'overfit.pt'
model_name = 'overfit_t1_t1ce.pt'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

testloader = BraTSDataset(data_dir)
model = BraTSSegmentation(input_channels=2) 

#checkpoint = torch.load('checkpoints/zero.pt')
#model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
loss = DiceLoss()
#optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
model.train()
debug = False
best_loss = 1.0

for i, data in enumerate(testloader):
    print("Training epoch: " + str(i))
    optimizer.zero_grad()
    src, target = data
    src = src.to(device, dtype=torch.float)
    target = target.to(device, dtype=torch.float)
    output = model(src)
    cur_loss = loss(output, target)
    print("cur_loss: {}".format(cur_loss))
    if cur_loss < best_loss:
        torch.save({'epoch': i, 'loss': cur_loss, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, 'checkpoints/'+ 'best_' +model_name)
                
        best_loss = cur_loss

    if i % 10 == 0 and not debug:
        print("Saving model after iteration {}".format(i))
        torch.save({'epoch': i, 'loss': cur_loss, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, 'checkpoints/'+ model_name)
    
    cur_loss.backward()
    optimizer.step()

    if cur_loss < 0.10:
        break
    if cur_loss >= 1.0:
        print("Ending training, loss is maximum.")


print("Training complete. Saving model.")
if not debug and cur_loss < 1.0: 
    torch.save({'epoch': i, 'loss': cur_loss, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, 'checkpoints/'+model_name)

