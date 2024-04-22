#!/usr/bin/env python3
from torch.utils.data import Dataset
from derivative import dxdt
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils, transforms
import torch
from PIL import Image
import time

class FailureDetectionDataLoader(Dataset):

    def __init__(self, path_to_directory, timehorizon = 30, current_transform=None):
        """_summary_

        Args:
            path_to_directory (str): path to the directory where the dataset is present
        """
        super().__init__()
        
        self.X = []
        self.y = []
        self.path_to_directory = path_to_directory
        self.time_horizon = timehorizon
        self.robot_data_filename = "robot_data.csv"
        self.image_data_directory = "images"
        self.data_transforms = current_transform


        self.wrench_normalization = [30,30,30,10,10,10]
        self.preprare_data()

    def preprare_data(self):
        subdirs = [x[1] for x in os.walk(self.path_to_directory)]
        subdirs = subdirs[0]

        print("\nLoading New Dataset")
        current_data_count = 0

        label1_count = 0
        label2_count = 0
        for current_directory in subdirs:
            print("Current Directory Count: ", current_data_count+1)
            robot_filename  = self.path_to_directory + "/" + str(current_directory) + "/" +self.robot_data_filename
            image_filename  = self.path_to_directory + "/" + str(current_directory) + "/" + self.image_data_directory
            label_filename = self.path_to_directory + "/" + str(current_directory) + "/label.csv"
            robot_data = pd.read_csv(robot_filename).to_dict('list')
            image_data = []
            label = pd.read_csv(label_filename).to_dict('list')
            
            if(len(robot_data['timestamp'])< self.time_horizon):
                print("Data has less number of points that chosen time horizon, skipping: ", current_directory)
                continue
            
            for image in os.listdir(image_filename):
                img = Image.open(str(os.path.join(image_filename,image)))
                if(self.data_transforms != None):
                    img = self.data_transforms(img)
                    
                if img is not None:
                    image_data.append(img)
                else:
                    print("NONE Image Found")
                
            for ids in range(len(robot_data['timestamp'])//self.time_horizon):
                start_index = ids*self.time_horizon
                end_index = start_index + self.time_horizon
                
                wrench_data = np.column_stack((np.asarray(robot_data['Fx'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[0],np.asarray(robot_data['Fy'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[1],np.asarray(robot_data['Fz'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[2],
                                                        np.asarray(robot_data['Tx'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[3],np.asarray(robot_data['Ty'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[4],np.asarray(robot_data['Tz'][start_index:end_index], dtype=np.float32)/self.wrench_normalization[5]))
                wrench_data = torch.from_numpy(wrench_data)
                
                                        
                current_image_data = torch.cat(image_data[start_index:end_index])
                current_image_data = torch.reshape(current_image_data, (self.time_horizon,int(current_image_data.shape[0]/self.time_horizon), current_image_data.shape[1],current_image_data.shape[2]))
                Current_X = [wrench_data,current_image_data]

                #Generating one-hot encoding vector
                Current_label = torch.zeros(2)
                Current_label[int(label["label"][0])-1] = 1
                
                self.X.append(Current_X)
                self.y.append(Current_label)
                
                if(int(label["label"][0]) == 1):
                    label1_count += 1
                else:
                    label2_count += 1
        
            current_data_count += 1
            
            if(current_data_count>5):
                break


        print("Done Loading Data with label 1 count: ", label1_count, " and label 2 count: ", label2_count)
            
        

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        X = self.X[index]
        y = self.y[index]

        return [X,y]


