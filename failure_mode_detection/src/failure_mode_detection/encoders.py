

import torch

from layers import CausalConv1D, conv2d, Flatten
from utilities import init_weights

##Force encoder
class WrenchEncoder(torch.nn.Module):

    def __init__(self, z_dim = 64, initialize_weights = True, device="cuda"):
        super().__init__()
        self.z_dim = z_dim
        self.device = device

        torch.device(self.device)

        self.wrench_encoder = torch.nn.Sequential(
            CausalConv1D(6,1024,kernel_size=2, stride=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(1024,512,kernel_size=2,stride=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(512,256,kernel_size=2,stride=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(256,128,kernel_size=2,stride=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            CausalConv1D(128, self.z_dim,kernel_size=2,stride=1),   #The original was 2* z_dim
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Flatten(), # Image grid to single feature vector
            torch.nn.Linear(z_dim, z_dim)
        ).to(self.device)



        if(initialize_weights):
            init_weights(self.modules())

        
    def forward(self, wrench_vector):
        return self.wrench_encoder(wrench_vector.to(self.device)).unsqueeze(2)



#Image Encoder
class ImageEncoder(torch.nn.Module):
    def __init__(self, z_dim = 64, input_channels = 3, c_dim = 16, initailize_weights=True, device="cuda"):
        """
        Image encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        self.z_dim = z_dim
        self.device = device
        torch.device(self.device)
        self.img_conv1 = conv2d(input_channels, c_dim, kernel_size=7, stride=2, device=device)
        self.img_conv2 = conv2d(c_dim, 32, kernel_size=5, stride=2, device=device)
        self.img_conv3 = conv2d(32, 64, kernel_size=5, stride=2, device=device)
        self.img_conv4 = conv2d(64, 64, stride=2, device=device)
        self.img_conv5 = conv2d(64, 128, stride=2)
        self.img_conv6 = conv2d(128, self.z_dim, stride=2, device=device)
        self.linear_layer = torch.nn.Linear(4 * self.z_dim, self.z_dim).to(device)
        
        self.flatten = Flatten().to(device)
 
        if initailize_weights:
            init_weights(self.modules())
        

    def forward(self, image):
        # image encoding layers
        image = image.to(self.device)
        out_img_conv1 = self.img_conv1(image)
        out_img_conv2 = self.img_conv2(out_img_conv1)
        out_img_conv3 = self.img_conv3(out_img_conv2)
        out_img_conv4 = self.img_conv4(out_img_conv3)
        out_img_conv5 = self.img_conv5(out_img_conv4)
        out_img_conv6 = self.img_conv6(out_img_conv5)
        
        # # image embedding parameters
        flattened = self.flatten(out_img_conv6)
        z_image = self.linear_layer(flattened).unsqueeze(2)
        
        return z_image


