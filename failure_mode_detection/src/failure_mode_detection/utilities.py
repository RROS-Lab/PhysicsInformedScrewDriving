import torch
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt

def init_weights(modules):
    """
    Weight initialization from original SensorFusion Code
    """
    for m in modules:
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
            torch.nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        



def scaled_image(image, output_size = (128,128), save_flag = False):
    """_summary_

    Args:
        image (_type_): _description_ should be an OpenCV format frame object from Realsense
        output_size (tuple, optional): _description_. Defaults to (128,128).
        rgb_scale (int, optional): _description_. Defaults to 255.

    Returns:
        _type_: _description_
    """
    ####Implement Scaling Code Here###
    new_image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
    if(save_flag):
        print("Saving Image")
        current_timestamp = datetime.datetime.timestamp(datetime.datetime.now())
        cv2.imwrite(str(current_timestamp)+".jpg", new_image)
        # cv2.imshow("Image", new_image)
        # cv2.waitKey(0)
    return new_image


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, : target.size(2), : target.size(3)]


def transposed_conv2d(in_planes, out_planes):
    return torch.nn.Sequential(
        torch.nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False
        ),
        torch.nn.LeakyReLU(0.1, inplace=True),
    )


def predict_flow(in_planes):
    return torch.nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)




def find_mean_std_dev(X):
    mean = 0
    std_dev = 0

    mean = np.mean(X)
    std_dev = np.std(X)

    print(mean,std_dev)

    lower = mean - std_dev
    upper = mean + std_dev

    # print(len(X),upper-lower,lower,upper,((lower<X)&(X<upper)).sum())
    # print("")
    return [lower, upper]

