import torch
import cv2
import datetime
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel
from matplotlib.patches import Ellipse

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


def torch_binom(n, k):
    mask = n.detach() >= k.detach()
    n = mask * n
    k = mask * k
    a = torch.lgamma(n + 1) - torch.lgamma((n - k) + 1) - torch.lgamma(k + 1)
    return torch.exp(a) * mask

def fit_ellipse_skimage(X1,y1, display_flag = True):
    """_summary_

    Args:
        X1 (_type_): px value of the screw tip motion
        y1 (_type_): py value of the screw tip motion
        display_flag (bool, optional): Will plot the input data with the ground truth ellipse. Defaults to True.

    Returns:
        list: Parameters of the ellipse
    """
    [x_mean, x_std] = find_mean_std_dev(X1)
    [y_mean, y_std] = find_mean_std_dev(y1)

    new_x = X1[np.where(np.logical_and(X1>=(x_mean-x_std), X1<=(x_mean+x_std)))]
    new_y = y1[np.where(np.logical_and(y1>=(y_mean-y_std), y1<=(y_mean+y_std)))]
    points = np.column_stack((new_x,new_y))
    
    x = points[:,0]
    y = points[:,1]
    ell = EllipseModel()
    ell.estimate(points)
    
    xc,yc,a,b,theta = ell.params
    plt.style.context('seaborn-whitegrid')
    if(display_flag):
        print("Ellipse Params: ", ell.params)
        fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
        # axs[0].scatter(x,y)

        axs.scatter(x, y,s=10,color='blue')
        axs.scatter(xc, yc, color='red', s=100)
        axs.set_xlim(x.min(), x.max())
        axs.set_ylim(y.min(), y.max())
        axs.set_aspect('equal',adjustable='datalim',anchor='C')
        axs.grid(axis='x', color='0.95')
        axs.grid(axis='y', color='0.95')
        plt.title('Screw Tip Motion Depiction')

        ell_patch = Ellipse((xc, yc), 2*a, 2*b, theta*180/np.pi, edgecolor='red', facecolor='none',lw=3)
        plt.show()
        
    return [xc,yc,a,b,theta]



def fit_ellipse_skimage_test(X1,y1, predicted_vals,label, display_flag = True):

    [x_mean, x_std] = find_mean_std_dev(X1)
    [y_mean, y_std] = find_mean_std_dev(y1)

    new_x = X1[np.where(np.logical_and(X1>=(x_mean-x_std), X1<=(x_mean+x_std)))]
    new_y = y1[np.where(np.logical_and(y1>=(y_mean-y_std), y1<=(y_mean+y_std)))]
    points = np.column_stack((new_x,new_y))
    
    x = points[:,0]
    y = points[:,1]
    ell = EllipseModel()
    ell.estimate(points)
    
    xc,yc,a,b,theta = ell.params
    print("Predicted Values: ", predicted_vals)

    ell2 = Ellipse(xy=(xc,yc), width=2*predicted_vals[0][0], height=2*predicted_vals[0][1], angle=(predicted_vals[0][2]*180/np.pi),edgecolor='blue', facecolor='none', label='Predicted Area',alpha=0.4, fill=True,lw=5)
    
    plt.style.context('seaborn-whitegrid')
   
    if(display_flag):
        
        fig, axs = plt.subplots(1, 1, sharex=True, sharey=True)
        # axs.scatter(x,y)
        axs.set_xlabel('Pixel Value in X')
        axs.set_ylabel('Pixel Value in Y')
        axs.set_alpha(0.2)
       
        
        
        # axs.scatter(x, y, s=10)
        # axs.scatter(xc, yc, color='red', s=100)
        axs.set_xlim(x.min(), x.max())
        axs.set_ylim(y.min(), y.max())

        ell_patch = Ellipse((xc, yc), 2*a, 2*b, theta*180/np.pi, edgecolor='green', facecolor='none', alpha=0.4, label='True Area', fill=True,lw=5)

        axs.add_patch(ell_patch)
        axs.add_artist(ell2)
        plt.legend(markerscale=0.5)
        plt.grid(axis='x', color='0.95')
        plt.grid(axis='y', color='0.95')
        plt.title('Area Coverage Model Performance')
        plt.show()
        
    return [xc,yc,a,b,theta]


def find_mean_std_dev(X):
    mean = 0
    std_dev = 0

    mean = np.mean(X)
    std_dev = np.std(X)

    print(mean,std_dev)

    lower = mean - std_dev
    upper = mean + std_dev

    return [lower, upper]

