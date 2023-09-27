import torch
from torch.utils.data import DataLoader
from device import DEVICE
from transform import validation_transform
from dataset import CatsDogsDataSet, TEST_SET_FOLDER, DataSetMode
import json
import csv
from network import Network, Down
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import math

def get_linear_layer(net):
    children = net.seq.children()
    for child in children:
        if type(child) is torch.nn.Linear:
            return child
        
    return None

def get_conv_layer(net, depth=1):
    children = net.seq.children()
    for child in children:
        if type(child) is Down:
            if depth == 1:
                for block in child.seq.children():
                    if type(block) is torch.nn.Conv2d:#.activation.ReLU:
                        return block
                #return child.seq.children().__next__().__next__().__next__().__next__()
            else:
                depth -= 1
    
    return None

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

IDX = 0
def display_activations(name, axs):
    act = activation[name][IDX].cpu()
    C = act.shape[0]
    W = act.shape[1]
    H = act.shape[2]
    act = act.reshape(C, 1, W, H)
    grid = make_grid(act, math.ceil(math.sqrt(C)), padding=4).permute(1,2,0)
    mn = torch.min(grid)
    mx = torch.max(grid)
    grid = torch.clip(grid, 0, 1)
    #grid = (grid - mn) / (mx - mn)

    axs.imshow(grid[:,:,0], cmap="cividis")

def display_linear(name, axs):
    act = activation[name][IDX].cpu()
    act = act.reshape(32,16)
    act = torch.clip(act, 0, 1)

    axs.imshow(act, cmap="cividis")


if __name__ == "__main__":
    dataset = CatsDogsDataSet(
        TEST_SET_FOLDER, max_samples_per_class=None, mode=DataSetMode.Test, transform=validation_transform) 
    
    dataloader = DataLoader(dataset, batch_size=200, shuffle=False)
    
    net = Network().to(DEVICE)
    net.eval()
    chkpt = torch.load("model.pt")
    net.load_state_dict(chkpt["net_state_dict"])

    predictions = torch.zeros(len(dataset)).to(DEVICE)
  
    get_conv_layer(net, 1).register_forward_hook(get_activation("conv1"))
    get_conv_layer(net, 2).register_forward_hook(get_activation("conv2"))
    get_conv_layer(net, 3).register_forward_hook(get_activation("conv3"))
    get_conv_layer(net, 4).register_forward_hook(get_activation("conv4"))
    get_linear_layer(net).register_forward_hook(get_activation("linear"))

    

    with torch.no_grad():
        batch, labels = dataloader.__iter__().__next__()
        x = net(batch)

        while True:
          fig, axs = plt.subplots(2,3)
          img = batch[IDX].cpu()
          img = img.permute(1,2,0)

          
          # plt.show()

          
          axs[0,0].imshow(img)
          display_activations("conv1", axs[0,1])
          display_activations("conv2", axs[0,2])
          display_activations("conv3", axs[1,0])
          display_activations("conv4", axs[1,1])
          display_linear("linear", axs[1,2])
          
          plt.show()
          IDX = IDX + 1

         
        
    

        