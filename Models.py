from torch import nn
import torch
import torch.nn.functional as F

class LeNet(nn.Module):

  def __init__(self):
    super(LeNet, self).__init__() 
    self.conv1 = nn.Conv2d(1,6,5) # kernel 5 and sride 1 => 28X28
    self.pool1 = nn.MaxPool2d(2,2) # kernel 2 and sride 2 => 24X24
    self.conv2 = nn.Conv2d(6,16,5) # kernel 5 and sride 1 => 12X12
    self.pool2 = nn.MaxPool2d(2,2) # kernel 2 and sride 2 => 8X8 
    self.fc1 = nn.Linear(16*4*4, 120) # 4x4 feature map
    self.fc2 = nn.Linear(120,84) # 120 input and 84 output
    self.fc3 = nn.Linear(84,10) # 84 input and 10 output


  def forward(self,x):
    x = self.pool1(F.relu(self.conv1(x))) 
    x = self.pool2(F.relu(self.conv2(x))) 
    x = torch.flatten(x,1) 
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x)) 
    logits = self.fc3(x) 
    return logits