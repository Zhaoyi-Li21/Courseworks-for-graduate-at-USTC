from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

parser = argparse.ArgumentParser(description='')
parser.add_argument('-lr',default='1e-3',type=float, help='learning rate of neural network')
parser.add_argument('-actfunc',default='relu',type=str, help = 'activation fuction type')
parser.add_argument('-depth',default='2',type=int, help='depth of the hidden layers of neural network')
parser.add_argument('-width',default='20', type=int, help="num of hidden unit in each hidden layer.")
args = parser.parse_args()

data_size = 5000
x=np.linspace(0,4*np.pi,data_size)
print(x[1])
y=np.sin(x)+np.exp(-x)

X=np.expand_dims(x,axis=1)
Y=y.reshape(data_size,-1)

dataset=TensorDataset(torch.tensor(X,dtype=torch.float).cuda(),torch.tensor(Y,dtype=torch.float).cuda())
train_ratio = 0.7
dev_ratio = 0.15
test_ratio = 0.15
train_size = int(train_ratio*data_size)
dev_size = int(dev_ratio*data_size)
test_size = data_size - train_size - dev_size
length_split = [train_size, dev_size, test_size]
train_set, dev_set, test_set = torch.utils.data.dataset.random_split(dataset, length_split)
print(train_set[1])
print(len(train_set))
dataloader=DataLoader(train_set,batch_size=64,shuffle=True)

def trans_dataset2tensor(dataset):
    '''
    input: torch.dataset; output: corresponding tensor
    '''
    if len(dataset) == 0:
        raise Exception("Blank dataset!!!")
    for i in range(len(dataset)):
        x, y = dataset[i]
        if i == 0:
            dest_x = x
            dest_y = y
        else:
            dest_x = torch.cat((dest_x,x),0)
            dest_y = torch.cat((dest_y,y),0)
    # dest_x.shape = [len], dest_y.shape = [len]
    dest_x = dest_x.unsqueeze(1)
    dest_y = dest_y.unsqueeze(1)
    return dest_x, dest_y
x_val, y_val = trans_dataset2tensor(dev_set)
x_test, y_test = trans_dataset2tensor(test_set)
        
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.inp_layer = nn.Linear(1, args.width)
        self.hiddens = nn.ModuleList()
        '''
        to make parameter in the net.parameters()
        '''
        for i in range(args.depth-1):
            self.hiddens.append(nn.Linear(args.width, args.width).cuda())
        self.out_layer = nn.Linear(args.width, 1)
        if args.actfunc == 'relu':
            self.activate = F.relu
        elif args.actfunc == 'tanh':
            self.activate = torch.tanh
        elif args.actfunc == 'sigmoid':
            self.activate = torch.sigmoid
        else:
            raise Exception("ERROR: parameter `actfunc` is invalid!")

    def forward(self, inp):
        
        x = self.activate(self.inp_layer(inp))
        for hid_layer in self.hiddens:
            x = self.activate(hid_layer(x))
        x = self.out_layer(x)
        return x



net=Net().cuda()
print(net)

optim=torch.optim.Adam(net.parameters(), lr=args.lr)
Loss=nn.MSELoss()

epoch_num = 500
train_losses = list()
val_losses = list()
steps = list()

for epoch in range(epoch_num):
    loss=None
    loss_epoch = 0
    cnt=0
    for batch_x,batch_y in dataloader:
        net.train()
        #print('flag_')
        y_predict=net(batch_x)
        loss=Loss(y_predict,batch_y)
        #print('flag___')
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_epoch += loss.item()
        cnt += 1
    loss_epoch /= cnt
   
    if (epoch+1)%10==0:
        print("step: {0} , loss: {1}".format(epoch+1,loss.item()))
        steps.append(epoch+1)
        train_losses.append(loss_epoch)

        net.eval()
        with torch.no_grad():
            y_val_pred = net(x_val)
            val_loss = Loss(y_val, 
                            y_val_pred)
            print(val_loss.item())
            val_losses.append(val_loss.item())



figname = 'loss_'+str(args.lr)+'_'+str(args.actfunc)+'_'+str(args.depth)+'_'+str(args.width)+'.png'
plt.xlabel('epochs')
plt.ylabel('loss')
plt.plot(steps, train_losses, marker="o", label="train", color='red')
plt.plot(steps, val_losses, marker="*", label="validation", color='blue')
plt.grid()
plt.legend()
plt.title('fit loss of training set and validation set')
plt.savefig('results/'+figname)

'''
test phase
'''
y_test_pred=net(x_test)
test_loss = Loss(y_test,y_test_pred).item()
print('lr:',args.lr)
print('activation function type:', args.actfunc)
print('depth:',args.depth)
print('width:',args.width)
print('test_loss:',test_loss)

fw = open("results/collect.txt","a+")

fw.write('lr:'+str(args.lr)+' activation function type:'+args.actfunc)
fw.write('\n')
fw.write('depth:'+str(args.depth)+' width:'+str(args.width))
fw.write('\n')
fw.write('test_loss:'+str(test_loss))
fw.write('\n'+'--------------'+'\n')



