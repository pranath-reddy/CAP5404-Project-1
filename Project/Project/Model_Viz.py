'''
CAP5404 Deep Learning for Computer Graphics
Project Part-1
Author: Pranath Reddy Kumbam
UFID: 8512-0977

- Model Visualization for prototyping CNN model
'''

# import libraries
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.autograd import Variable
from torchviz import make_dot

# define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.classifier = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.ReLU(),
            nn.Conv2d(6, 12, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(72, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

# push to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)

# Print the model architecture and export graph visualization
print(model)
summary(model, (1,6,7))
x = torch.zeros(1, 1, 6, 7, dtype=torch.float, requires_grad=False)
out = model(x)
make_dot(out).render("CNN_Graph", format="png")
make_dot(out, show_attrs='True', show_saved='True').render("CNN_Graph_Full", format="png")
