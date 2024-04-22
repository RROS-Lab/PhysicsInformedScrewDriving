"""_summary_This code has been adopted from the multimodal fusion repository https://github.com/stanford-iprl-lab/multimodal_representation
_ Further Modifications have been made to suit the IROS 2023 Problem
"""


import torch
from layers import conv2d, Flatten, CausalConv1D
from utilities import init_weights



##Robot Pose Encoder
class PoseEncoder(torch.nn.Module):
    def __init__(self, z_dim = 64, initailize_weights=True, device="cuda"):
        """_summary_: Encodes the endeffector pose

        Args:
            z_dim (_type_): _description_ Dimension of the latent space variable
            initailize_weights (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        torch.device(device)
        self.z_dim = z_dim
        self.device = device
        self.pose_encoder = torch.nn.Sequential(
            torch.nn.Linear(6, 512),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(128, self.z_dim),
            torch.nn.LeakyReLU(0.1, inplace=True),
        ).to(device)


        ##Add the code for a decoder too
    
        if initailize_weights:
            init_weights(self.modules())

    def forward(self, current_pose):
        return self.pose_encoder(current_pose.to(self.device))



class OrientationEncoder(torch.nn.Module):
    def __init__(self, z_dim = 64, initailize_weights=True, device="cuda"):
        """_summary_: Encodes the endeffector pose

        Args:
            z_dim (_type_): _description_ Dimension of the latent space variable
            initailize_weights (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        torch.device(device)
        self.z_dim = z_dim
        self.device = device
        self.pose_encoder = torch.nn.Sequential(
            torch.nn.Linear(3, 100),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(100, 80),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(80, 40),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(40, self.z_dim),
            torch.nn.LeakyReLU(0.1, inplace=True),
        ).to(device)


        ##Add the code for a decoder too
    
        if initailize_weights:
            init_weights(self.modules())

    def forward(self, current_pose):
        return self.pose_encoder(current_pose.to(self.device))



class WrenchEncoderCausal(torch.nn.Module):

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
            torch.nn.Linear(1620, z_dim)
        ).to(self.device)

        if(initialize_weights):
            init_weights(self.modules())

        
    def forward(self, wrench_vector):
        return self.wrench_encoder(wrench_vector.to(self.device))


class WrenchEncoder(torch.nn.Module):
    def __init__(self, z_dim = 64, initailize_weights=True, device="cuda"):
        """_summary_: Encodes the endeffector pose

        Args:
            z_dim (_type_): _description_ Dimension of the latent space variable
            initailize_weights (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        torch.device(device)
        self.z_dim = z_dim
        self.device = device
        self.wrench_encoder = torch.nn.Sequential(
            torch.nn.Linear(6, 100),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(100, 80),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(80, 40),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(40, self.z_dim),
            torch.nn.LeakyReLU(0.1, inplace=True),
        ).to(device)


        ##Add the code for a decoder too
    
        if initailize_weights:
            init_weights(self.modules())

    def forward(self, current_pose):
        return self.wrench_encoder(current_pose.to(self.device))



#Image Encoder
class ImageEncoder(torch.nn.Module):
    def __init__(self, z_dim = 64, input_channels = 3, c_dim = 16,initailize_weights=True, device="cuda"):
        """
        Image encoder taken from Making Sense of Vision and Touch
        """
        super().__init__()
        self.z_dim = z_dim
        self.device = device
        torch.device(self.device)
        self.img_conv1 = conv2d(input_channels, c_dim, kernel_size=7, stride=2, device=device)  #<---- Do not change the 16,3 here
        self.img_conv2 = conv2d(1024, 512, kernel_size=5, stride=2, device=device)
        self.img_conv3 = conv2d(512, 256, kernel_size=5, stride=2, device=device)
        self.img_conv4 = conv2d(256, 128, stride=2, device=device)
        self.img_conv5 = conv2d(128, 2*self.z_dim, stride=2, device=device)
        self.linear_layer = torch.nn.Linear(2 * self.z_dim, self.z_dim).to(device)
        
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
        
    
        # image embedding parameters
        flattened = self.flatten(out_img_conv5)
        z_image = self.linear_layer(flattened).unsqueeze(2)
        
        return z_image


