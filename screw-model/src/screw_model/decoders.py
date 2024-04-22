import torch
from utilities import *
from layers import *

class ImageDecoder(torch.nn.Module):
    def __init__(self, z_dim=256, input_channels = 3, c_dim = 16,initailize_weights=True, device = 'cuda'):
        """
        Decodes the Image.
        """
        super().__init__()
        self.z_dim = z_dim
        self.device = device
        self.linear_decoder_layer = torch.nn.Sequential(torch.nn.Linear(self.z_dim, 2*self.z_dim), torch.nn.LeakyReLU(0.1, inplace=True)).to(self.device)
        self.img_deconv6 = transposed_conv2d(128, 64)
        self.img_deconv5 = transposed_conv2d(64, 32)
        self.img_deconv4 = transposed_conv2d(162, 32)
        self.img_deconv3 = transposed_conv2d(32, 16)
        self.img_deconv2 = transposed_conv2d(c_dim, input_channels)
       
        
        if initailize_weights:
            init_weights(self.modules())

    def forward(self, z_features):
        """
        Decodes the image features.
        Args:
            z space of the images: action conditioned z (output of fusion + action network)
            img_out_convs: outputs of the image encoders (skip connections)
        """
        # image encoding layers
        z_features = z_features.to(self.device)
        linear_output = self.linear_decoder_layer(z_features)
        unflatenned_output = linear_output.reshape(linear_output.shape[0],-1,8,8)

        out_img_deconv1 = self.img_deconv6(unflatenned_output)
        out_img_deconv2 = self.img_deconv5(out_img_deconv1)
        out_img_deconv3 = self.img_deconv4(out_img_deconv2)
        out_img_deconv4 = self.img_deconv3(out_img_deconv3)
        out_img_deconv5 = self.img_deconv2(out_img_deconv4)
        decoder_output = torch.sigmoid(out_img_deconv5)
        

        return decoder_output



class PoseDecoder(torch.nn.Module):
    def __init__(self, z_dim = 256, pose_dim=6, initailize_weights=True, device = "cuda"):
        """
        Decodes the Pose of the Robot
        """
        super().__init__()
        self.z_dim = z_dim
        self.device = device
        self.pose_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim, 128),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(64, 32),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(32, pose_dim),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, pose_features):
        return self.pose_decoder(pose_features)


class OrientationDecoder(torch.nn.Module):
    def __init__(self, z_dim = 64, orientation_dim=3, initailize_weights=True, device = "cuda"):
        """
        Decodes the Pose of the Robot
        """
        super().__init__()
        self.z_dim = z_dim
        self.device = device
        self.orientation_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim, 32),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(32, 64),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(64, 100),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(100, orientation_dim),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, pose_features):
        return self.orientation_decoder(pose_features)

class WrenchDecoder(torch.nn.Module):
    def __init__(self, z_dim = 64, orientation_dim=6, initailize_weights=True, device = "cuda"):
        """
        Decodes the Pose of the Robot
        """
        super().__init__()
        self.z_dim = z_dim
        self.device = device
        self.orientation_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.z_dim, 32),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(32, 64),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(64, 100),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(100, orientation_dim),
        )

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, pose_features):
        return self.orientation_decoder(pose_features)


class WrenchDecoderCausal(torch.nn.Module):

    def __init__(self, z_dim = 256, wrench_dim=6, initailize_weights=True, device = "cuda"):
        super().__init__()
        self.z_dim = z_dim
        self.device = device

        self.wrench_out1 = CausalConvTransposed1D(self.z_dim,128,kernel_size=1, stride=2)
        self.wrench_out2 = CausalConvTransposed1D(128,64,kernel_size=1, stride=2)
        self.wrench_out3 = CausalConvTransposed1D(64,32,kernel_size=1, stride=2)
        self.wrench_out4 = CausalConvTransposed1D(32,24,kernel_size=1, stride=2)
        self.wrench_out5 = CausalConvTransposed1D(24,wrench_dim,kernel_size=1, stride=2)
        


        if initailize_weights:
            init_weights(self.modules())

    
    def forward(self, wrench_feature):
        
        wrench_feature = wrench_feature.to(self.device)
        wrench_output_1 = self.wrench_out1(wrench_feature)
        wrench_output_2 = self.wrench_out2(wrench_output_1)
        wrench_output_3 = self.wrench_out3(wrench_output_2)
        wrench_output_4 = self.wrench_out4(wrench_output_3)
        wrench_output_5 = self.wrench_out5(wrench_output_4)


        decoder_output = torch.tanh(wrench_output_5)

        return decoder_output


class WrenchDecoderV2(torch.nn.Module):

    def __init__(self, z_dim = 256, wrench_dim=6, initailize_weights=True, device = "cuda"):
        super().__init__()
        self.z_dim = z_dim
        self.device = device
        
        self.wrench_out1 = CausalConvTransposed1D(self.z_dim,128,kernel_size=1, stride=1)
        self.wrench_out2 = CausalConvTransposed1D(128,256,kernel_size=1, stride=1)
        self.wrench_out3 = CausalConvTransposed1D(256,512,kernel_size=1, stride=1)
        self.wrench_out4 = CausalConvTransposed1D(512,1024,kernel_size=1, stride=1)
        self.wrench_out5 = CausalConvTransposed1D(1024,wrench_dim,kernel_size=1, stride=1)
        self.activation = torch.nn.LeakyReLU(0.1, inplace=True)


        if initailize_weights:
            init_weights(self.modules())

    
    def forward(self, wrench_feature):
        
        wrench_feature = wrench_feature.to(self.device)
        wrench_output_1 = self.wrench_out1(wrench_feature)
        wrench_output_2 = self.wrench_out2(wrench_output_1)
        wrench_output_3 = self.wrench_out3(wrench_output_2)
        wrench_output_4 = self.wrench_out4(wrench_output_3)
        wrench_output_5 = self.wrench_out5(wrench_output_4)
        
        decoder_output = torch.tanh(wrench_output_5)

        return decoder_output