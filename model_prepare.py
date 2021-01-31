from Common.Model.LeNet import LeNet
from Common.Utils.data_loader import load_data_mnist
import torch
import torchvision
#PATH = './Model/LetNet'
#model = LeNet()
#torch.save(model, PATH) 
#model=torch.load(PATH)
#print(model)
import pdb
import numpy as np 

def load_server_data_mnist(root='./Data/MNIST'):
    """Download the fashion mnist dataset and then load into memory."""
    transforms = torchvision.transforms
    mnist_train = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                        ]))
    server_data = []
    label = np.ones(10) * 10
    for i in range(len(mnist_train)):
        if label[mnist_train[i][1]] > 0:
            server_data.append(mnist_train[i])
            label[mnist_train[i][1]] -= 1
        if np.all(label == 0):
            break
    pdb.set_trace()
  
    
    #for i in range(args.num_workers):
    #torch.save(distributed_mnist_train[i], root+'/'+'mnist_train_'+str(i)+'_.pt')
    #train_iter = torch.utils.data.DataLoader(distributed_mnist_train[0], batch_size=batch_size, shuffle=True, num_workers=0)
    #test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #return train_iter, test_iter
if __name__ == "__main__":
    load_server_data_mnist()
