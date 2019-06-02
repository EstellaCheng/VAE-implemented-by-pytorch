import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

# training
BATCH_SIZE = 100
trainset = datasets.MNIST('./data/', train=True, download=True,
                          transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=2)
# test
testset = datasets.MNIST('./data/', train=False, download=False,
                         transform=transforms.ToTensor())

testloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                         shuffle=True, num_workers=2)


def show_images(images):
    images = torchvision.utils.make_grid(images)
    show_image(images[0])


def show_image(img):
    plt.imshow(img, cmap='gray')
    plt.show()


dataiter = iter(trainloader)
images, labels = dataiter.next()
show_images(images)


class VAE(nn.Module):
    def __init__(self, latent_variable_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2m = nn.Linear(400, latent_variable_dim)  # use for mean 均值
        self.fc2s = nn.Linear(400, latent_variable_dim)  # use for standard deviation 方差

        # 根据均值和方差学到隐向量z
        # 再根据隐向量z解码得到图片
        self.fc3 = nn.Linear(latent_variable_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    # 重参数化
    def reparameterize(self, log_var, mu):
        s = torch.exp(0.5 * log_var)
        eps = torch.rand_like(s)  # generate a iid standard normal same shape as s
        return eps.mul(s).add_(mu)  # z=mu+eps*s

    def forward(self, input):
        x = input.view(-1, 784)  # 将图片展开成一维向量
        x = torch.relu(self.fc1(x))
        log_s = self.fc2s(x)  # 得到log方差
        m = self.fc2m(x)  # 得到均值m
        z = self.reparameterize(log_s, m)  # 重参数化，得到Z

        x = self.decode(z)

        return x, m, log_s

    def decode(self, z):
        x = torch.relu(self.fc3(z))
        x = torch.sigmoid(self.fc4(x))
        return x


def loss(input_image, recon_image, mu, log_var):
    CE = F.binary_cross_entropy(recon_image, input_image.view(-1, 784), reduction='sum')  # 交叉熵
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())  # KL散度

    return KLD + CE


## train
print("start training....")
vae = VAE(5)
optimizer = optim.Adam(vae.parameters(), lr=0.0001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loss = []
for epoch in range(5):
    for i, data in enumerate(trainloader, 0):
        images, labels = data
        images = images.to(device)
        optimizer.zero_grad()
        recon_image, s, mu = vae(images)
        l = loss(images, recon_image, mu, s)
        l.backward()
        train_loss.append(l.item() / len(images))
        optimizer.step()
plt.plot(train_loss)
plt.show()

# test
print("start testing....")
with torch.no_grad():
    for i, data in enumerate(testloader, 0):
        images, labels = data
        images = images.to(device)
        recon_image, s, mu = vae(images)
        recon_image_ = recon_image.view(BATCH_SIZE, 1, 28, 28)
        if i % 100 == 0:
            show_images(recon_image_)

# generate，just use the decoder part and pass some realization to latent variable Z to generate some images
print("start generating...")
with torch.no_grad():
    # attention please, the dimension of Z must match the dimension of latent_variable in VAE
    z = [[0.2, 0.5, 0.2, 0.2, 0.2], [0, 1, 1, 1, 1], [1, 1, 1, 1, 1], [-1, -1, -1, -1, -1],
         [-0.9, -0.9, -0.9, -0.9, -0.9], [-0.2, -0.2, -0.2, -0.2, -0.2]]
    sample_images = vae.decode(torch.FloatTensor(z))
    sample_images_ = sample_images.view(len(z), 1, 28, 28)
    print(sample_images_.size())
    show_images(sample_images_)
