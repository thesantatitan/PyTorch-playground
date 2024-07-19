import torch
from torch._prims_common import reduction_dims
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import os
from PIL import Image
from torchvision import transforms
import plotext as plt


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
        xb = F.relu(self.l1(xb))
        xb = F.relu(self.l2(xb))
        xb = F.relu(self.l3(xb))
        xb = F.relu(self.l4(xb))
        return xb

class Decoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decoder, self).__init__()
        step_size = (output_size - input_size)//2
        self.l1 = nn.Linear(input_size, input_size)
        self.l2 = nn.Linear(input_size, input_size + step_size)
        self.l3 = nn.Linear(input_size + step_size, output_size)
        self.l4 = nn.Linear(output_size, output_size)

    def forward(self, xb):
        xb = F.relu(self.l1(xb))
        xb = F.relu(self.l2(xb))
        xb = F.relu(self.l3(xb))
        xb = F.relu(self.l4(xb))
        return xb


class AutoEncoder(nn.Module):
    def __init__(self, feature_size, latent_size):
        super().__init__()
        self.enc = Encoder(feature_size, latent_size)
        self.dec = Decoder(latent_size, feature_size)

    def forward(self, x):
        return self.dec(self.enc(x))


loss_fn = F.l1_loss

epochs = 1
batch_size = 50
tmp_img = load_data(0,1)
model = AutoEncoder(tmp_img.shape[-1], 2).to(dev)
opt = optim.Adam(model.parameters())
loss_vals = []


for epoch in range(epochs):
    for i in range(2000//batch_size):
        images = load_data(i*batch_size, (i+1)*batch_size)
        images.to(dev)
        pred = model(images)
        loss = loss_fn(pred,images,reduction='sum')
        print("Batch Loss",i,loss)
        loss_vals.append(loss.item())
        loss.backward()
        opt.step()
        opt.zero_grad()

plt.scatter(loss_vals)
plt.show()

