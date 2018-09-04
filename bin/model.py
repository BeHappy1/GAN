import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
img_shape = (1, 28, 28)
sample_interval = 400
class G_mlp(nn.Module):
    def __init__(self,img_shape,in_plane=100):
        super(G_mlp, self).__init__()
        self._in_plane = in_plane
        self.img_shape = img_shape
        def block(in_plane,out_plane):
            layers = []
            layers.append(nn.Linear(in_plane,out_plane))
            layers.append(nn.BatchNorm2d(out_plane,0.8))
            layers.append(nn.LeakyReLU(0.2,inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(self._in_plane,128),
            *block(128,256),
            *block(256,512),
            *block(512,1024),
            nn.Linear(1024,int(np.prod(self.img_shape))),
            nn.Tanh()
        )
    def forward(self,x):
        out = self.model(x)
        img = out.view(out.size(0),*self.img_shape)
        return img
class D_mlp(nn.Module):
    def __init__(self,img_shape):
        super(D_mlp, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)),512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256,1),
            nn.sigmoid()
        )
    def forward(self,x):
        x = x.view(x.size(0),-1)
        out = self.model(x)
        return out
loss_fn = nn.BCELoss()
generator = G_mlp()
discriminator = D_mlp()
cuda = True if torch.cuda.is_available() else False
if cuda():
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    loss_fn = loss_fn.cuda()
os.makedirs('../../data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=64, shuffle=True)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.001, betas=(0.9, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.9, 0.999))
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

for epoch in range(200):
    for i,(img,_) in enumerate(dataloader):
        valid = Tensor(img.size(0), 1).fill_(1.0)
        fake = Tensor(img.size(0), 1).fill_(0.0)
        real_img = img.type(Tensor)

        optimizer_G.zero_grad()
        z = Tensor(np.random.normal(0, 1, (img.shape[0], 100)))
        gen_img = generator(z)
        g_loss = loss_fn(discriminator(gen_img), valid)
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        real_loss = loss_fn(discriminator(real_img), valid)
        fake_loss = loss_fn(discriminator(gen_img.detach()), fake)
        loss = (real_loss + fake_loss) / 2
        loss.backward()
        optimizer_D.step()
        print("[Epoch %d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, i, len(dataloader),
                                                                         loss.item(), g_loss.item()))
        batches_done = epoch * len(dataloader) + i
        if batches_done % sample_interval == 0:
            save_image(gen_img.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)

