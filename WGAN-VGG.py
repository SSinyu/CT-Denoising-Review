
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.models import vgg19


class DCMsDataset(Dataset):
    def __init__(self, input_path, target_path, transfrom=None):
        self.input_path = input_path
        self.target_path = target_path
        self.input_dir = sorted(os.listdir(input_path))
        self.target_dir = sorted(os.listdir(target_path))
        self.transfrom = transfrom

    def __len__(self):
        # input and target have same length
        return len(self.input_dir)

    def __getitem__(self, idx):
        input_name = os.path.join(self.input_path, self.input_dir[idx])
        target_name = os.path.join(self.target_path, self.target_dir[idx])
        input_img = np.load(input_name)
        target_img = np.load(target_name)
        #sample = {'input':input_img, 'target':target_img}
        sample = (input_img, target_img)

        if self.transfrom:
            sample = self.transfrom(sample)

        return sample



class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        out = self.feature_extractor(img)
        return out



class Generator_CNN(nn.Module):
    def __init__(self):
        super(Generator_CNN, self).__init__()
        self.g_conv_n32s1_f = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.g_conv_n32s1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.g_conv_n1s1 = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, img):
        out = self.relu(self.g_conv_n32s1_f(img))
        for _ in range(6):
            out = self.relu(self.g_conv_n32s1(out))
        out = self.relu(self.g_conv_n1s1(out))
        return out



class Discriminator_CNN(nn.Module):
    def __init__(self, input_size=55):
        super(Discriminator_CNN, self).__init__()

        def discriminator_block(in_filters, out_filters, stride):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, padding=0)]
            layers.append(nn.LeakyReLU())
            return layers

        layers = []
        for in_filters, out_filters, stride in [(1,64,1), (64,64,2), (64,128,1), (128,128,2), (128,256,1), (256,256,2)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride))

        self.cnn = nn.Sequential(*layers)
        self.leaky = nn.LeakyReLU()
        self.fc1 = nn.Linear(256*5*5, 1024)
        self.fc2 = nn.Linear(1024,1)

    def forward(self, img):
        out = self.cnn(img)
        out = out.view(-1, 256*5*5)
        out = self.fc1(out)
        out = self.leaky(out)
        out = self.fc2(out)
        return out




#### training ####
LEARNING_RATE = 1e-3
LEARNING_RATE_ = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 20
OUT_CHANNELS = 96

# patch datasets
input_img_dir = '/home/datascience/PycharmProjects/CT/patch/input/'
target_img_dir = '/home/datascience/PycharmProjects/CT/patch/target/'
#print(os.listdir(input_img_dir)[:3])
#print(os.listdir(target_img_dir)[:3])

dcm = DCMsDataset(input_img_dir, target_img_dir)
dcmloader = DataLoader(dcm, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dcm, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


generator = Generator_CNN()
discriminator = Discriminator_CNN(input_size=64)
feature_extractor = FeatureExtractor()

if torch.cuda.device_count() > 1:
    print("Use {} GPUs".format(torch.cuda.device_count()), "="*9)
    generator = nn.DataParallel(generator)
    discriminator = nn.DataParallel(discriminator)
    feature_extractor = nn.DataParallel(feature_extractor)

generator.to(device)
discriminator.to(device)
feature_extractor.to(device)


criterion_GAN = nn.MSELoss()
criterion_perceptual = nn.L1Loss()


optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5,0.9))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-5, betas=(0.5,0.9))

patch = (BATCH_SIZE, 1)

Tensor = torch.cuda.FloatTensor
valid = Variable(Tensor(np.ones(patch)), requires_grad=False)
fake = Variable(Tensor(np.zeros(patch)), requires_grad=False)

total_step = len(dcmloader)
for epoch in range(10):
    for i, (inputs, targets) in enumerate(dcmloader):
        input_img = torch.tensor(inputs, requires_grad=True).unsqueeze(1).to(device)
        target_img = torch.tensor(targets).unsqueeze(1).to(device)

        # Generator
        optimizer_G.zero_grad()

        gen = generator(input_img)

        gen_valid = discriminator(gen)
        loss_GAN = criterion_GAN(gen_valid, valid)

        gen_dup = gen.repeat(1,3,1,1)
        target_dup = target_img.repeat(1,3,1,1)
        gen_features = feature_extractor(gen_dup)
        real_features = Variable(feature_extractor(target_dup), requires_grad=False)
        loss_perceptual = criterion_perceptual(gen_features, real_features)

        loss_G = loss_GAN + (0.1 * loss_perceptual)
        loss_G.backward()
        optimizer_G.step()

        # Discriminator
        optimizer_D.zero_grad()

        loss_real = criterion_GAN(discriminator(target_img), valid)
        loss_fake = criterion_GAN(discriminator(input_img.detach()), fake)

        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()

        if i % 10 == 0:
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch+1, 10, i, len(dcmloader), loss_D.item(), loss_G.item()))

        if (epoch+1) % 10 == 0:
            torch.save(generator.state_dict(), "WGAN_VGG_{}ep.ckpt".format(epoch+1))
            torch.save(discriminator.state_dict(), "WGAN_VGG_{}ep.ckpt".format(epoch + 1))
