from Common.Model.LeNet import LeNet
import torch
PATH = './Model/LetNet'
#model = LeNet()
#torch.save(model, PATH) 
model=torch.load(PATH)
print(model)