import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from PIL import Image
from torchvision import transforms

dev = torch.device("mps")

def load_data(start=0, end=2001):
    files = os.listdir("circle_images")
    files = [f for f in files if f.endswith(".png")]
    files.sort()


    to_tensor = transforms.ToTensor()
    images = []
    for f in files[start:end]:
        img = Image.open("circle_images/" + f)
        #convert to black and white
        img = img.convert("L")
        img = to_tensor(img)
        images.append(img)
    images = torch.stack(images)
    return images.reshape(images.shape[0], -1).to(dev)



class Encoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Encoder, self).__init__()
        step_size = (input_size - output_size)//2
        self.l1 = nn.Linear(input_size, input_size)
        self.l2 = nn.Linear(input_size, input_size - step_size)
        self.l3 = nn.Linear(input_size - step_size, output_size)
        self.l4 = nn.Linear(output_size, output_size)

    def forward(self, xb):
        print(xb.shape)
        xb = F.relu(self.l1(xb))
        print(xb.shape)
        xb = F.relu(self.l2(xb))
        print(xb.shape)
        xb = F.relu(self.l3(xb))
        print(xb.shape)
        xb = F.relu(self.l4(xb))
        print(xb.shape)
        return xb



images = load_data(0,10)
encoder = Encoder(images.shape[-1],2)
encoder.to(dev)
encoder(images)
