import torch
from model import ScrewMeanMotion_Model
from utilities import init_weights
from dataloaders import ScrewMeanModel_Data_Preparer, DataLoader, utils
import wandb
import argparse
import json
import time
import os
import datetime

MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


class ScrewMeanModelTrainer():

    def __init__(self, training_params):
        
        self.device = training_params['device']
        torch.device(self.device)   
        self.model = ScrewMeanMotion_Model(training_params)
        init_weights(self.model.modules())
    
        ##Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=training_params['learning_rate'], weight_decay=training_params['weight_decay'])
        

        ##Training Hyperparameters
        self.batch_size = training_params["batch_size"]
        self.num_epochs = training_params["epochs"]
        self.time_horizon = training_params["time_horizon"]
        self.learning_rate = training_params["learning_rate"]
        self.weight_decay = training_params["weight_decay"]
        self.log_freq = training_params["log_freq"]
        
        ###Defining losses
        self.range_loss = torch.nn.MSELoss()


        ###Dataset Preparation
        self.package_dir = os.path.join(MAIN_DIR,'screw-model')
        train_data_dir = os.path.join(self.package_dir, "dataset", training_params["training_dir"])
        self.train_data_prep_obj = ScrewMeanModel_Data_Preparer(train_data_dir,self.time_horizon)
        
        dev_data_dir =  os.path.join(self.package_dir, "dataset", training_params["dev_dir"])
        self.dev_data_prep_obj = ScrewMeanModel_Data_Preparer(dev_data_dir,self.time_horizon)
        
        test_data_dir =  os.path.join(self.package_dir, "dataset", training_params["test_dir"])
        self.test_data_prep_obj = ScrewMeanModel_Data_Preparer(test_data_dir,self.time_horizon)

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
        self.checkpoint_filepath =  os.path.join(self.package_dir, "models", str(current_saving_timestamp)+"_mean_model","checkpoints")
        os.makedirs(self.checkpoint_filepath ,mode=0o777)
        self.model_filepath = os.path.join(self.package_dir, "models", str(current_saving_timestamp)+"_mean_model", "model")
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
            current_epoch_range_loss = 0
            
            current_epoch_range_dev_loss = 0
            
            
            for batch_id, (X,y) in enumerate(train_data_loader):
                print("Current Batch: ", batch_id)
                self.optimizer.zero_grad()
                start_time = time.time()
                predicted_y = self.model(X[0])
                end_time = time.time()
                print("Model Inference Time: ", end_time-start_time)
                
                ##Computing loss
                y = y.to(torch.device(self.device))
                range_train_loss = self.range_loss(y,predicted_y)*1000
                current_epoch_range_loss += range_train_loss.item()
                
                range_train_loss.backward()
                
                self.optimizer.step()
                current_epoch_train_loss += range_train_loss.item()
                

            current_epoch_range_loss = current_epoch_range_loss/len(train_data_loader.dataset)
            current_epoch_train_loss = current_epoch_train_loss/len(train_data_loader.dataset)

            print("Training Losses at epoch: ", epoch + 1," are as follows: ")
            print("1. Range Training Loss: ", current_epoch_range_loss)
            print("2. Total Training Loss: ", current_epoch_train_loss, "\n")

            self.model.eval()

            for dev_batch_id, (X_dev,y_dev) in enumerate(dev_data_loader):
                # X_dev = X_dev.to(torch.device(self.device))
                y_dev = y_dev.to(torch.device(self.device))

                dev_predicted_y = self.model(X_dev[0])
                dev_predicted_y = dev_predicted_y.to(torch.device("cuda"))
                
                ##Computing development loss
                range_dev_loss = self.range_loss(y_dev,dev_predicted_y)
                current_epoch_range_dev_loss = range_dev_loss.item()
                
                print("Development Losses at epoch: ", epoch + 1,", for batch: ",dev_batch_id)
                print("1. Range Loss: ", current_epoch_range_dev_loss)
                

            
    
                
            self.save_checkpoint(epoch, range_train_loss,"model_"+str(epoch)+".pt")        
            

        self.save_model()

        return


    def evaluate(self):
        test_data_loader = DataLoader(self.test_data_prep_obj)
        self.model.eval()
        self.model.get_latent_params = False
            

        print("Training Data Performance")
        for id,(X_train, y_train) in enumerate(DataLoader(self.train_data_prep_obj)): 
            train_predicted_y = self.model(X_train[0])
            train_predicted_y = train_predicted_y.to(torch.device("cuda"))
            y_train = y_train.to(torch.device("cuda"))
            utils.fit_ellipse_skimage_test(X_train[1].detach().cpu().numpy(),X_train[2].detach().cpu().numpy()[0],train_predicted_y.detach().cpu().numpy()[0], True)

        print("Development Data Performance: ")
        for id,(X_dev, y_dev) in enumerate(DataLoader(self.dev_data_prep_obj)): 
            dev_predicted_y = self.model(X_dev[0])
            dev_predicted_y = dev_predicted_y.to(torch.device("cuda"))
            y_dev = y_dev.to(torch.device("cuda"))
            utils.fit_ellipse_skimage_test(X_dev[1].detach().cpu().numpy(),X_dev[2].detach().cpu().numpy()[0],dev_predicted_y.detach().cpu().numpy()[0], True)

        print("Testing Data Performance")    
        for id,(X_test, y_test) in enumerate(test_data_loader): 
            tests_predicted_y = self.model(X_test[0])
            tests_predicted_y = tests_predicted_y.to(torch.device("cuda"))
            y_test = y_test.to(torch.device("cuda"))
            utils.fit_ellipse_skimage_test(X_test[1].detach().cpu().numpy(),X_test[2].detach().cpu().numpy()[0],tests_predicted_y.detach().cpu().numpy()[0], True)
        
        
        return

    
    def setup_wandb_loggers(self):

        config = dict(learning_rate = self.learning_rate, weight_decay=self.weight_decay, batch_size = self.batch_size, time_horizon=self.time_horizon)

        wandb.init(project='screw_model',config=config)
        wandb.watch(self.model, log_freq=self.log_freq)
        


    def save_checkpoint(self, current_epoch, curent_losses, filename="model.pt"):
        filepath = os.path.join(self.checkpoint_filepath, filename)
        torch.save({'epoch':current_epoch, 'loss':curent_losses,'model_state_dict':self.model.state_dict(), 'optimizer_state_dict':self.optimizer.state_dict()}, filepath)


    def save_model(self, filename="screw_mean_model.pt"):
        filepath = os.path.join(self.model_filepath,filename)
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
    parser.add_argument('-p', '--orientation_dim', default=64, type=int, help='dim for orientation')
    parser.add_argument('-i', '--image_dim', default=64, type=int, help='Latent space dim for image data')


    #Params
    parser.add_argument('-t', '--time_horizon', type=int, help='The number time points to consider in a dataset')
    

    parser.add_argument('-K', '--stiffness_dim', type=int, help='Dimension of Stiffness Matrix')
    parser.add_argument('-C', '--damping_dim', type=int, help='Dimension of Damping Matrix')
    

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

    
    model_trainer_obj = ScrewMeanModelTrainer(training_params_dict)

    model_trainer_obj.train()
    model_trainer_obj.evaluate()

    return
    

if __name__ == '__main__':
    main()


