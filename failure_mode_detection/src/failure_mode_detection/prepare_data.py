import rospy
import numpy as np
import copy
import os
import pandas as pd
import cv2
import pathlib
import csv
from cv_bridge import CvBridge

class DataPreparer():

    def __init__(self, robot_data_filename, image_data_dir, final_dir):

        self.raw_image_data = []
        self.raw_robot_data = []


        #Data Collection Parameters
        self.time_aligned_image_data = []
        self.time_aligned_robot_data = []
        



        ##Filenames
        self.robot_data_filename = robot_data_filename
        self.image_dir = image_data_dir
        self.processed_dir = final_dir


        #Dataprocessing params
        self.time_diff_robot_to_camera = -326
        self.force_cuttoff = 25


        self.bridge = CvBridge()


        self.data_collection_time = 13      #Time in seconds for data collection and screwing



        return

    
    def generate_time_aligned_data(self):
        
        if(not self.data_processing_completed):
            print("Raw data Processing is not complete yet")
            return

        force_z_abs = np.abs(self.raw_robot_data[:,9])

        previous_delta = np.inf
        for i in range(0,len(force_z_abs)-1):
            if(force_z_abs[i] < self.force_cuttoff):
                continue
            
            delta_f = force_z_abs[i] - force_z_abs[i+1]

            if(delta_f>0):
                start_robot_index = i + 10
                break


        
        starting_robot_timestamp = self.raw_robot_data[start_robot_index,0]
        ending_robot_timestamp = starting_robot_timestamp + ((self.data_collection_time-5)*1000)
        
        peak_found = False
        current_loop_count = copy.deepcopy(int(start_robot_index))
        previous_delta = np.inf
        
        while(not peak_found):
            current_loop_count += int(1)
            delta_t = np.abs((self.raw_robot_data[current_loop_count-2,0] - ending_robot_timestamp))
            
            if(delta_t>previous_delta):
                end_robot_index = current_loop_count
                ending_robot_timestamp = self.raw_robot_data[current_loop_count-2,0]
                
                peak_found = True

            previous_delta = copy.deepcopy(delta_t)
        
        
        self.clipped_robot_data = self.raw_robot_data[start_robot_index:end_robot_index]
        print("Size of clipped robot data: ", len(self.clipped_robot_data))        

        starting_image_timestamp = int(starting_robot_timestamp + self.time_diff_robot_to_camera)
        start_image_index = 0
        previous_delta = np.inf
        
        current_image_cnt = 0
        for image in self.raw_image_data:
            # current_timestamp = int((image.header.stamp.secs)*1000 + (image.header.stamp.nsecs/1e6))
            current_timestamp = image[0]
            delta_t = np.abs((current_timestamp - starting_image_timestamp))

            if(delta_t>previous_delta):
                start_image_index = current_image_cnt - 1
                starting_image_timestamp = current_timestamp
                break
            
            current_image_cnt += 1
            previous_delta = copy.deepcopy(delta_t)

        ending_image_timestamp = starting_image_timestamp + int((self.data_collection_time-5)*1000)
        end_image_index = 0
        previous_delta = np.inf
        for image_index in range(start_image_index, len(self.raw_image_data)):
            # current_timestamp = int((image.header.stamp.secs)*1000 + (image.header.stamp.nsecs/1e6))
            current_timestamp = self.raw_image_data[image_index][0]
            
            delta_t = np.abs((current_timestamp - ending_image_timestamp))

            if(delta_t>previous_delta):
                end_image_index = image_index - 1
                ending_image_timestamp = current_timestamp
                break
            
            previous_delta = copy.deepcopy(delta_t)

        # print("Start Image Timestamp: ", starting_image_timestamp, "; End Image Timestamp: ", ending_image_timestamp)
        # print("Start Robot Timestamp: ", starting_robot_timestamp, "; End Robot Timestamp: ", ending_robot_timestamp)

        self.clipped_image_data = self.raw_image_data[start_image_index:end_image_index]
        self.time_aligned_frame_ids = [start_image_index, end_image_index]

        # print("Length of robot data: ", len(self.clipped_robot_data))
        # print("Length of image data: ", len(self.clipped_image_data))

        
        if(len(self.clipped_image_data)>=len(self.clipped_robot_data)): 
            for current_robot_data in self.clipped_robot_data:
                print("Need to Implement this")
                raise NotImplementedError
        else:
            robot_index = 0
            for current_image in self.clipped_image_data:
                # current_image_timestamp = int((current_image.header.stamp.secs)*1000 + (current_image.header.stamp.nsecs/1e6))
                current_image_timestamp = current_image[0]
                previous_delta = np.inf
                for robot_data_id in range(robot_index, len(self.clipped_robot_data)):
                    delta_t = np.abs((current_image_timestamp - self.clipped_robot_data[robot_data_id,0]))

                    if(delta_t>previous_delta):
                        self.time_aligned_image_data.append(current_image)
                        self.time_aligned_robot_data.append(self.clipped_robot_data[robot_data_id])
                        robot_index = robot_data_id + 1
                        break
                    
                    previous_delta = copy.deepcopy(delta_t)

        print("Lenght of time aligned robot data: ", len(self.time_aligned_robot_data))
        print("Lenght of time aligned iamge data: ", len(self.time_aligned_image_data))
        
    def save_current_data(self, current_saving_timestamp, current_label):  
        print("Saving data")
        
        
        path_to_folder = self.processed_dir
        directory_path = os.path.join(path_to_folder, current_saving_timestamp)
        os.mkdir(directory_path,mode=0o777)

        filepath = str(directory_path) + "/" +"robot_data.csv"
        writer = csv.writer(open(filepath, 'w',encoding='UTF8', newline=''))
        writer.writerow(["timestamp","X","Y","Z", "A", "B", "C", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz","Vx", "Vy", "Vz", "Wx","Wy","Wz"])
        
        for state in self.time_aligned_robot_data:
            position_data = state[1:7]
            wrench_data = state[7:13]
            velocity_data = state[13:19]
            
            writer.writerow([state[0],position_data[0],position_data[1],position_data[2],position_data[3],position_data[4], position_data[5],
                            wrench_data[0],wrench_data[1],wrench_data[2],wrench_data[3],wrench_data[4],wrench_data[5],
                            velocity_data[0],velocity_data[1],velocity_data[2],velocity_data[3],velocity_data[4],velocity_data[5]])

            
        print("Robot Data Saved in File: ", filepath)

        print("saving_image_data")
        
        image_directory = os.path.join(str(directory_path), "images")
        os.mkdir(image_directory,mode=0o777)
        
        for image in self.time_aligned_image_data:
            current_image_timestamp = image[0]
            image_filepath = str(image_directory)+"/"+str(current_image_timestamp)+".jpg"
            cv2.imwrite(image_filepath, self.bridge.imgmsg_to_cv2(image[1][0]))

        print("Done saving image data in: ", image_directory)


        print("Current Label: ", current_label)
        filepath = str(directory_path) + "/" +"label.csv"
        writer = csv.writer(open(filepath, 'w',encoding='UTF8', newline=''))
        writer.writerow(["label"])
        writer.writerow([current_label])
        
        print("Done saving the label")
        

        return






    def save_data_in_new_folder(self, path_to_directory):

        subdirs = [x[1] for x in os.walk(path_to_directory)]
        subdirs = subdirs[0]
        current_dir_count = 0
        self.data_processing_completed = True
            
        for current_directory in subdirs:
            print("Current Directory: ", current_directory)
            
            self.raw_robot_data = []
            self.raw_image_data = []
            self.time_aligned_image_data = []
            self.time_aligned_robot_data = []
            
            robot_filename  = path_to_directory + "/" + str(current_directory) + "/" +self.robot_data_filename
            image_folder  = path_to_directory + "/" + str(current_directory) + "/" + self.image_dir
            label_filename = path_to_directory + "/" + str(current_directory) + "/" + "label.csv"
            self.raw_robot_data = pd.read_csv(robot_filename).to_numpy()
            current_label = pd.read_csv(label_filename).to_dict()
            
            for image_filename in os.listdir(image_folder):
                img = cv2.imread(os.path.join(image_folder,image_filename))
                if img is not None:
                    image_filename = image_filename.replace('.jpg','')
                    image_filename = int(image_filename)
                    
                    img_msg = [image_filename, [self.bridge.cv2_to_imgmsg(img)]]
                    self.raw_image_data.append(img_msg)

            self.raw_image_data.sort(key=lambda raw_image_data:raw_image_data[0])
            self.generate_time_aligned_data()


            saving_timestamp = str(current_directory).partition('_')[0]
            print("Saving Timestamp: ", saving_timestamp)
            print("label: ", current_label["label"])
            self.save_current_data(saving_timestamp, 2)

            current_dir_count += 1
        
            # if(current_dir_count == 1):
            #     break


            

        return

    





def main():
    
    rospy.init_node("prepare_data")
    robot_file_name = "raw_robot_data.csv"
    image_dir = "raw_images"


    package_dir = str(pathlib.Path.cwd().resolve().parents[1])
    case1_dir = package_dir + '/dataset/raw_dataset/Case2/to_be_processed'
    case1_final_dir = package_dir + '/dataset/case2_processed/'

    data_preparer_obj = DataPreparer(robot_file_name, image_dir, case1_final_dir)

    data_preparer_obj.save_data_in_new_folder(case1_dir)



if __name__== "__main__":
  main()  