 # Pytorch 1.10
# Python 3.7
# Tesla V100 32GB
import time
import numpy as np
import torch
import torch.nn as nn
import math
import csv
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

# Set the seed for reproducibility
torch.manual_seed(1234)

# Determine the device to be used for computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the filepath to save the trained model
filepath_to_save_mode = 'case1b + hard constraints + rcnn + amp.pt'

# Generate input data points for the PDE
t = np.linspace(0, 100, 101)
x = np.linspace(0, 500, 51)
# y = np.linspace(0, 500, 51) for 2D
ms_t, ms_x = np.meshgrid(t, x)
# ms_t, ms_x, ms_y = np.meshgrid(t, x，y)
t_pde = np.ravel(ms_t).reshape(-1,1)
x_pde = np.ravel(ms_x).reshape(-1,1)
# y_pde = np.ravel(ms_y).reshape(-1,1)

# Convert the input data points to PyTorch tensors and move them to the device
pt_t_collocation = Variable(torch.from_numpy(t_pde).float(), requires_grad=True).to(device)
pt_x_collocation = Variable(torch.from_numpy(x_pde).float(), requires_grad=True).to(device)
# pt_y_collocation = Variable(torch.from_numpy(y_pde).float(), requires_grad=True).to(device)
# Define the activation function for the hidden layers
act = nn.Tanh()

# Define the neural network model using the RC architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer1 = nn.Linear(2, 64)
        # self.hidden_layer1 = nn.Linear(3, 64)
        self.hidden_layer2 = nn.Linear(64, 64)
        self.hidden_layer3 = nn.Linear(64, 64)
        self.hidden_layer4 = nn.Linear(64, 64)
        self.hidden_layer5 = nn.Linear(64, 64)
        self.hidden_layer6 = nn.Linear(64, 64)
        self.hidden_layer7 = nn.Linear(64, 64)
        self.hidden_layer1X = nn.Linear(2, 64)
         # self.hidden_layer1X = nn.Linear(3, 64)
        self.hidden_layer2X = nn.Linear(64, 64)
        self.hidden_layer3X = nn.Linear(64, 64)
        self.hidden_layer4X = nn.Linear(64, 64)
        self.hidden_layer5X = nn.Linear(64, 64)
        self.hidden_layer6X = nn.Linear(64, 64)
        self.hidden_layer7X = nn.Linear(64, 64)
        self.output_layerX = nn.Linear(64, 1)
    
    # Define a method for enforcing the hard constraints of the PDE
    def Hard_Constraints(self, t, x, nn_outputs):
        #Hard_Constraints(self, t, x,y, nn_outputs):
        output_hard = 20-2*x/500 + x*(x-500)*t * nn_outputs/(100*250*250)
        return output_hard
    
    # Define the forward pass of the model / RCNN
    def forward(self, t, x):
        X = 2*x / 500-1
        T = 2*t / 100-1
        inputs = torch.cat([T, X], dim=1)
        layer1_out = act(self.hidden_layer1(inputs))
        layer2_out = act(self.hidden_layer2(layer1_out)+self.hidden_layer1(inputs))
        layer3_out = act(self.hidden_layer3(layer2_out)+self.hidden_layer1(inputs))
        layer4_out = act(self.hidden_layer4(layer3_out)+self.hidden_layer1(inputs))
        layer5_out = act(self.hidden_layer5(layer4_out)+self.hidden_layer1(inputs))
        layer6_out = act(self.hidden_layer6(layer5_out)+self.hidden_layer1(inputs))
        layer7_out = act(self.hidden_layer7(layer6_out)+self.hidden_layer1(inputs))
        layer1_outX = act(self.hidden_layer1X(inputs))
        layer2_outX = act(self.hidden_layer2X(layer1_outX+layer1_out))
        layer3_outX = act(self.hidden_layer3X(layer2_outX+layer2_out ))
        layer4_outX = act(self.hidden_layer4X(layer3_outX +layer3_out))
        layer5_outX = act(self.hidden_layer5X(layer4_outX+layer4_out ))
        layer6_outX = act(self.hidden_layer6X(layer5_outX +layer5_out))
        layer7_outX = act(self.hidden_layer7X(layer6_outX +layer6_out))
        nn_output = self.output_layerX(layer7_outX+layer7_out)
        output = self.Hard_Constraints(t, x,nn_output)
        return output
    
    # Define the loss function for the model
    def loss_pde(net,t, x):
        h =net(t,x)
        h_t = torch.autograd.grad(h.sum(), t, create_graph=True)[0]
        h_x = torch.autograd.grad(h.sum(), x, create_graph=True)[0]
        m = h * h_x
        m_x = torch.autograd.grad(m.sum(), x, create_graph=True)[0]
        K = 30
        miu = 0.2
        w = 0.001*torch.exp(0.05*t)
        #0.1*torch.exp(-0.05*t) for Case 1a
        f = lossa(K * m_x + w - miu * h_t, 0 * h)
        return f

# Initialize the model and move it to the device
net = Net().to(device)

# Define the optimizer and the GradScaler for mixed precision training
optimizer = torch.optim.AdamW(net.parameters())
scaler = GradScaler()

# Define the loss function and the list to store the loss values
lossa = torch.nn.MSELoss()
losses = []  

# Train the model for 10,000 iterations
for i in range(10000):
    with autocast():
        optimizer.zero_grad()
        if i % 100 == 0:
            torch.save(net.state_dict(), filepath_to_save_mode)
        loss1 = net.loss_pde(pt_t_collocation, pt_x_collocation)
        loss = loss1
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    losses.append(loss.item())    
    if i % 100 == 0:
        print(f"Iteration: {i}, Loss: {loss:.12f}")
        print(f"GPU Memory Usage: {torch.cuda.max_memory_allocated() / 1024 ** 3:.8f} GB")

# Save the trained model and the loss values to files
torch.save(net.state_dict(), filepath_to_save_mode)
with open('loss.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Iteration', 'Loss'])
    for i, loss_value in enumerate(losses):
        writer.writerow([i, loss_value])

# Print the number of parameters in the model
params = torch.nn.utils.parameters_to_vector(net.parameters()).numel()
print(f"number of parameters:{params}") 

# Plot
h0 = net(pt_t_collocation,pt_x_collocation)
u1=np.asarray(h0.data)
u1 = u1.reshape(ms_t.shape)
fig = plt.figure(figsize=(20,5))
plt.subplot(1, 2, 1)
plt.imshow(u1, cmap='jet', interpolation='bilinear', aspect=0.2, extent=[0,100,500,0])
plt.colorbar()
plt.xlabel('t/d')
plt.ylabel('x/m')
plt.savefig('map.png', dpi=600, bbox_inches='tight') 
plt.show()