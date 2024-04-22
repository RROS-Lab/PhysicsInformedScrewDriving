import torch
from model import Screw_Model
from dataloaders import DataLoader, ScrewModel_Data_Preparer
import wandb
import argparse
import json
import time
import matplotlib.pyplot as plt
from scipy import integrate
import os
import datetime
import numpy as np

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

class ModelTrainer():

    def __init__(self, training_params, sindy_params):
        
        self.device = training_params['device']
        torch.device(self.device)   
        self.sindy_params = sindy_params
        print("Defining Screw Model V2")
        self.model = Screw_Model(self.sindy_params)
    
        ##Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=training_params['learning_rate'], weight_decay=training_params['weight_decay'])


        ##Training Hyperparameters
        self.batch_size = training_params["batch_size"]
        self.num_epochs = training_params["epochs"]
        self.time_horizon = sindy_params["time_horizon"]
        self.learning_rate = training_params["learning_rate"]
        self.weight_decay = training_params["weight_decay"]
        self.log_freq = training_params["log_freq"]
        self.sequential_threshold_freq = training_params["sequential_thresholding_freq"]
        self.sequential_threshold_param = training_params["sequential_thresholding_param"]
        self.sindy_loss_weight = training_params['sindy_loss_weight']

        ###Defining losses
        self.orientation_reconstruction = torch.nn.MSELoss()
        self.wrench_reconstruction = torch.nn.MSELoss()
        self.sindy_loss = torch.nn.MSELoss()


        ###Dataset Preparation
        self.package_dir = os.path.join(MAIN_DIR,'screw-model')
        train_data_dir = os.path.join(self.package_dir, "dataset", training_params["training_dir"])
        self.train_data_prep_obj = ScrewModel_Data_Preparer(train_data_dir,self.time_horizon)
        
        dev_data_dir =  os.path.join(self.package_dir, "dataset", training_params["dev_dir"])
        self.dev_data_prep_obj = ScrewModel_Data_Preparer(dev_data_dir,self.time_horizon)
        
        test_data_dir =  os.path.join(self.package_dir, "dataset", training_params["test_dir"])
        self.test_data_prep_obj = ScrewModel_Data_Preparer(test_data_dir,self.time_horizon)

        ##Loading Params
        self.load_checkpoint = training_params['load_checkpoint']
        current_checkpoint_filepath = training_params['preload_model_path']

        if(self.load_checkpoint):
            model_filename = os.path.join(self.package_dir, "models", current_checkpoint_filepath)
            self.load_model(filename=model_filename)

        self.wandb = training_params["wandb"]
        if(self.wandb):
            self.setup_wandb_loggers()
 


    def train(self):
        
        ##Saving parameters
        current_saving_timestamp = datetime.datetime.now()
        self.checkpoint_filepath =  os.path.join(self.package_dir, "models", str(current_saving_timestamp)+"_sindy_model","checkpoints")
        os.makedirs(self.checkpoint_filepath ,mode=0o777)
        self.model_filepath = os.path.join(self.package_dir, "models", str(current_saving_timestamp)+"_sindy_model", "model")
        os.makedirs(self.model_filepath ,mode=0o777)

        ###Initializing Dataloaders for training
        train_data_loader = DataLoader(self.train_data_prep_obj, batch_size=self.batch_size, shuffle=True)
        print("Total Number of Training batches: ", len(train_data_loader.dataset))
        ###Initializing Dataloaders for development data
        dev_data_loader = DataLoader(self.dev_data_prep_obj)

        self.model.to(torch.device("cuda"))
        self.model.get_latent_params = True
        
        for epoch in range(self.num_epochs):
            print("Initiating Training for Epoch: ", epoch + 1)

            self.model.train()
            
            current_epoch_train_loss = 0
            current_epoch_sindy_train_loss = 0
            current_epoch_wrench_train_loss = 0
            current_epoch_orientation_train_loss = 0

            current_epoch_dev_loss = 0
            current_epoch_sindy_dev_loss = 0
            current_epoch_wrench_dev_loss = 0
            current_epoch_orientation_dev_loss = 0
    
            
            for batch_id, (X,x_dot) in enumerate(train_data_loader):
                
                self.optimizer.zero_grad()
                start_time = time.time()
                [predicted_x_dot, orientation_decoded, wrench_decoded] = self.model(X[0])
                
                predicted_x_dot = predicted_x_dot.to(torch.device("cuda"))
                end_time = time.time()
                
                orientation = X[0][:,:,2:5]
                orientation = orientation.to(torch.device(self.device))
                wrench= X[0][:,:,5:11]
                wrench = wrench.to(torch.device(self.device))
                
                ##Computing loss
                x_dot = x_dot.to(torch.device(self.device))
                sindy_train_loss = self.sindy_loss_weight*self.sindy_loss(x_dot, predicted_x_dot)
                current_epoch_sindy_train_loss += sindy_train_loss.item()
                
                orientation_train_reconstruction = self.orientation_reconstruction(orientation,orientation_decoded)*1000
                current_epoch_orientation_train_loss += orientation_train_reconstruction.item()
                
                wrench_train_reconstruction = self.wrench_reconstruction(wrench, wrench_decoded)*1000
                current_epoch_wrench_train_loss += wrench_train_reconstruction.item()
                
                train_loss = sindy_train_loss + orientation_train_reconstruction + wrench_train_reconstruction
                train_loss.backward()
                
                self.optimizer.step()
                current_epoch_train_loss += train_loss.item()
                

            current_epoch_sindy_train_loss = current_epoch_sindy_train_loss/len(train_data_loader.dataset)
            current_epoch_orientation_train_loss = current_epoch_orientation_train_loss/len(train_data_loader.dataset)
            current_epoch_wrench_train_loss = current_epoch_wrench_train_loss/len(train_data_loader.dataset)
            current_epoch_train_loss = current_epoch_train_loss/len(train_data_loader.dataset)
            print("Training Losses at epoch: ", epoch + 1, " are as follows: ")
            print("1. Sindy Loss: ", current_epoch_sindy_train_loss)
            print("2. Orientation Reconstruction Loss: ", current_epoch_orientation_train_loss)
            print("3. Wrench Reconstruction Loss: ", current_epoch_wrench_train_loss)
            print("4. Total Training Loss: ", current_epoch_train_loss, "\n")

            self.model.eval()

            for dev_batch_id, (X_dev,x_dot_dev) in enumerate(dev_data_loader):
                # X_dev = X_dev.to(torch.device(self.device))
                X_dev_in = X_dev[0].to(torch.device(self.device))
                x_dot_dev = x_dot_dev.to(torch.device(self.device))

                orientation_dev = X_dev_in[:,:,2:5]
                orientation_dev = orientation_dev.to(self.device)
                
                wrench_dev = X_dev_in[:,:,5:11]
                wrench_dev = wrench_dev.to(self.device)

                [dev_predicted_x_dot, dev_orientation_decoded, dev_wrench_decoded] = self.model(X_dev_in)
                
                ##Computing development loss
                sindy_dev_loss = self.sindy_loss_weight*self.sindy_loss(x_dot_dev, dev_predicted_x_dot)
                orientation_dev_reconstruction = self.orientation_reconstruction(orientation_dev,dev_orientation_decoded)
                wrench_dev_reconstruction = self.wrench_reconstruction(wrench_dev, dev_wrench_decoded)

                dev_loss = sindy_dev_loss + orientation_dev_reconstruction + wrench_dev_reconstruction
                current_epoch_dev_loss = dev_loss.item()
                current_epoch_sindy_dev_loss = sindy_dev_loss.item()
                current_epoch_wrench_dev_loss = wrench_dev_reconstruction.item()
                current_epoch_orientation_dev_loss = orientation_dev_reconstruction.item()

                print("Development Losses at batch:" , dev_batch_id, ", for epoch: ", epoch + 1, " are as follows: ")
                print("1. Sindy Loss: ", current_epoch_sindy_dev_loss)
                print("2. Orientation Reconstruction Loss: ", current_epoch_orientation_dev_loss)
                print("3. Wrench Reconstruction Loss: ", current_epoch_wrench_dev_loss)
                print("4. Total Development Loss: ", current_epoch_dev_loss, "\n")


            if(((epoch+1) % self.log_freq) == 0):

                if(self.wandb):
                    wandb.log({'train_loss':current_epoch_train_loss,'train_sindy_loss':current_epoch_sindy_train_loss, 'train_orientation_reconstruction': current_epoch_orientation_train_loss,\
                            'train_wrench_reconstruction': current_epoch_wrench_train_loss,\
                            'dev_loss':current_epoch_dev_loss,'dev_sindy_loss':current_epoch_sindy_dev_loss, 'dev_orientation_reconstruction': current_epoch_orientation_dev_loss,\
                            'dev_wrench_reconstruction': current_epoch_wrench_dev_loss})
                
                self.save_checkpoint(epoch, train_loss,"model_"+str(epoch)+".pt")

            
            if(self.sequential_threshold_freq != 0):
                if((epoch+1)%self.sequential_threshold_freq == 0):
                    self.model.SinDy.coefficient_mask[np.abs(self.model.SinDy.model_params["sindy_coefficients"].cpu().detach().numpy())<self.sequential_threshold_param] = 0.0
                    
            

        self.save_model()

        return


    def evaluate(self):
        
        plt.style.context('seaborn-whitegrid')
      
        for id,(X_test, x_dot_test) in enumerate(DataLoader(self.train_data_prep_obj)): 
        
            self.model.eval()
            print("Sindy Coefficients: ", self.model.SinDy.model_params['sindy_coefficients'])
            self.model.get_latent_params = False
            tests_predicted_x_dot = self.model(X_test[0])
            
            
            tests_predicted_x_dot = tests_predicted_x_dot.detach().cpu().numpy()
            tests_predicted_x_dot = np.reshape(tests_predicted_x_dot,(self.time_horizon,2))
            
            x_dot_test = x_dot_test.detach().cpu().numpy()
            x_dot_test = np.reshape(x_dot_test,(self.time_horizon,2))
            
            t = X_test[1].flatten().detach().cpu().numpy()
            
            r_dot_og = x_dot_test[:,0]
            theta_dot_og = x_dot_test[:,1]


            r_dot_pred = tests_predicted_x_dot[:,0]
            theta_dot_pred = tests_predicted_x_dot[:,1]
            
            r0 = X_test[2].flatten().detach().cpu().numpy()[0]
            theta0 = X_test[3].flatten().detach().cpu().numpy()[0]
            r_int_og = integrate.cumtrapz(r_dot_og, t, initial=r0)
            r_int_og = r_int_og/np.linalg.norm(r_int_og)
            
            theta_int_og = integrate.cumtrapz(theta_dot_og, t, initial=theta0)
            
            r_int_pred = integrate.cumtrapz(r_dot_pred, t, initial=r0)
            r_int_pred = r_int_pred/np.linalg.norm(r_int_pred)
            
            theta_int_pred = integrate.cumtrapz(theta_dot_pred, t, initial=theta0)
            theta_int_pred = theta_int_pred/np.pi
            
            plt.figure(1)
            _, ax1 = plt.subplots(1,1)
            ax1.plot(t,r_int_og,'g', label='Orignal')
            ax1.set_title('R predicted vs R original')
            ax1.grid(axis='x', color='0.98')
            ax1.grid(axis='y', color='0.98')
            ax1.plot(t,r_int_pred,'b', label='Predicted')
            ax1.legend()
            
            plt.figure(2)
            _, ax4 = plt.subplots(1,1)
            ax4.plot(t,theta_int_og,'g',label='Original')
            
            ax4.set_title('Theta predicted vs Theta original')
            ax4.grid(axis='x', color='0.98')
            ax4.grid(axis='y', color='0.98')
            ax4.plot(t,theta_int_pred,'b',label='Predicted')
            ax4.legend()
            
            plt.show()


        
        return

    
    def setup_wandb_loggers(self):
        

        config = dict(learning_rate = self.learning_rate, weight_decay=self.weight_decay, batch_size = self.batch_size, time_horizon=self.time_horizon)

        wandb.init(project='screw_model',config=config)
        wandb.watch(self.model, log_freq=self.log_freq)
        


    def save_checkpoint(self, current_epoch, curent_losses, filename="model.pt"):
        filepath = self.checkpoint_filepath + filename
        torch.save({'epoch':current_epoch, 'loss':curent_losses,'model_state_dict':self.model.state_dict(), 'optimizer_state_dict':self.optimizer.state_dict()}, filepath)


    def save_model(self, filename="screw_mean_model.model"):
        filepath = self.model_filepath + filename
        torch.save({'model_state_dict':self.model.state_dict(), 'optimizer_state_dict':self.optimizer.state_dict()}, filepath)


    def load_model(self, filename):

        print("Loading the model stored in : ", filename)
        checkpoint = torch.load(filename)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



def main():
    
    parser = argparse.ArgumentParser(description='Screw model training arguments.')
    
    ###Training Hyperparameters
    parser.add_argument('-l', '--learning_rate', type=float, help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, help='mini-batch size')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to train for')
    parser.add_argument('-a', '--weight_decay', type=int, help='Weight decary factor for L1 Regularization of Sindy model')
    parser.add_argument('-S', '--sequential_thresholding_freq', type=int, help='Frequency of sequential thresholding of Sindy model params. Set to 0 if no sequential thresholding')
    parser.add_argument('-u', '--sequential_thresholding_param', type=float, help='Value of sequential thresholding of Sindy model params to set mask to 0')
    

    ###Dataset Params
    parser.add_argument('-T', '--training_dir', type=str, help='Name of the training directory in dataset folder')
    parser.add_argument('-D', '--dev_dir', type=str, help='Name of the development directory in dataset folder')
    parser.add_argument('-R', '--test_dir', type=str, help='Name of the testing directory in dataset folder')
    
    ###Logging Params
    parser.add_argument('-L', '--log_freq', type=int, help='Frequency at which the logging happens for wandb')
    parser.add_argument('-c', '--load_checkpoint', type=bool, help='Flag to start training from a previous checkpoint')
    parser.add_argument('-k', '--preload_model_path', type=int, help='Filename for the checkpoint')
    parser.add_argument('-n''--wandb', type=bool, help='Use wand or not')

    ##Device params
    parser.add_argument('-d', '--device', default='cuda', type=str, help='Set this argument to cuda if GPU capability needs to be enabled')


    ##Screw Autoencoder Params
    parser.add_argument('-w', '--wrench_dim', default=64,type=int, help='Latent space dim for wrench')
    parser.add_argument('-p', '--orientation_dim', default=64, type=int, help='Latent space dim for orientation')
    parser.add_argument('-i', '--image_dim', default=64, type=int, help='Latent space dim for image data')


    #SinDy Params
    parser.add_argument('-t', '--time_horizon', type=int, help='The number time points to consider in a dataset for SinDy model')
    parser.add_argument('-P', '--poly_order', type=int, help='The number order of the polynomial for SinDy model')
    parser.add_argument('-M', '--model_order', default=1, type=int, help='The order of the SinDy model, x dx ddx')
    parser.add_argument('-s', '--include_sine', default=True, type=bool, help='Flag to include sine terms for SinDy model')
    parser.add_argument('-x', '--state_space_dim', default=2, type=int, help='This is the dimension of X for SinDy model. Default is 2 for (x,y) pixel values')
    parser.add_argument('-m', '--sindy_loss_weight', default=1e-4, type=float, help='Sindy Loss Scaling Factor')


    ##Input JSON setup file
    parser.add_argument('-f', '--training_param_file', required=True, type=str, help='Training Params Filename')
    

    args = parser.parse_args()
    arg_parse_dict = vars(args)

    package_dir = os.path.join(MAIN_DIR, 'screw-model')
    config_dir = os.path.join(package_dir, 'config')
    config_filename = os.path.join(config_dir, args.training_param_file)
    ##Defining Network Architecture Specific Params
    with open(config_filename, 'rb') as file:
        training_params_dict = json.load(file)

    arg_parse_dict.update(training_params_dict)

    sindy_params = {}
    sindy_params['device'] = training_params_dict['device']
    sindy_params['time_horizon'] = training_params_dict['time_horizon']
    sindy_params['model_order'] = training_params_dict['model_order']
    sindy_params['include_sine'] = training_params_dict['include_sine']
    sindy_params['poly_order'] = training_params_dict['poly_order']
    sindy_params['state_space_dim'] = training_params_dict['state_space_dim']
    sindy_params['latent_dim'] = training_params_dict['state_space_dim'] + training_params_dict['orientation_dim'] + training_params_dict['wrench_dim']#+training_params_dict["stiffness_damping_dim"]
    sindy_params['sindy_loss_weight'] = training_params_dict['sindy_loss_weight']
    sindy_params["wrench_dim"] = training_params_dict["wrench_dim"]
    sindy_params["orientation_dim"] = training_params_dict["orientation_dim"]

    model_trainer_obj = ModelTrainer(training_params_dict, sindy_params)

    model_trainer_obj.train()
    model_trainer_obj.evaluate()

    return
    

if __name__ == '__main__':
    main()


