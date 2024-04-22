import torch
from model import FailureModeDetector
from dataloaders import FailureDetectionDataLoader, transforms
import wandb
import argparse
import json
import os
from torch.utils.data import DataLoader
import datetime
from sklearn.metrics import confusion_matrix
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

class ModelTrainer():

    def __init__(self, training_params):
        
        self.device = training_params['device']
        torch.device(self.device)   
        
        self.model = FailureModeDetector(training_params)
       
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
        self.loss = torch.nn.BCEWithLogitsLoss()
        # self.loss = torch.nn.SoftMarginLoss()

        
        #Image Data Transforms
        self.data_transforms = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])


        ###Dataset Preparation
        self.package_dir = os.path.join(MAIN_DIR,'failure_mode_detection')
        train_data_dir = os.path.join(self.package_dir, "dataset",training_params["training_dir"])
        self.train_data_prep_obj = FailureDetectionDataLoader(train_data_dir,self.time_horizon,self.data_transforms)
        
        dev_data_dir =  os.path.join(self.package_dir, "dataset", training_params["dev_dir"])
        self.dev_data_prep_obj = FailureDetectionDataLoader(dev_data_dir,self.time_horizon, self.data_transforms)
        
        test_data_dir =  os.path.join(self.package_dir, "dataset", training_params["test_dir"])
        self.test_data_prep_obj = FailureDetectionDataLoader(test_data_dir,self.time_horizon, self.data_transforms)

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
        self.checkpoint_filepath =  os.path.join(self.package_dir, "models", str(current_saving_timestamp),"checkpoints")
        os.makedirs(self.checkpoint_filepath ,mode=0o777)
        self.model_filepath = os.path.join(self.package_dir, "models", str(current_saving_timestamp), "model")
        os.makedirs(self.model_filepath ,mode=0o777)

        ###Initializing Dataloaders for training
        train_data_loader = DataLoader(self.train_data_prep_obj, batch_size=self.batch_size, shuffle=True)
        print("Total Number of Training batches: ", len(train_data_loader.dataset))
        ###Initializing Dataloaders for development data
        dev_data_loader = DataLoader(self.dev_data_prep_obj)

        self.model.to(torch.device("cuda"))
        
        for epoch in range(self.num_epochs):
            print("Initiating Training for Epoch: ", epoch + 1)

            self.model.train()
            
            current_epoch_train_loss = 0
            current_epoch_dev_loss = 0
            
            
            for batch_id, (X_train,train_labels) in enumerate(train_data_loader):
                print("Current Batch: ", batch_id)
                
                train_labels = train_labels.to(torch.device(self.device))
                self.optimizer.zero_grad()
                predicted_train_logits = self.model(X_train)
                predicted_train_logits = predicted_train_logits.to(torch.device(self.device))
                print(predicted_train_logits)
                
                predicted_train_labels = torch.sigmoid(predicted_train_logits)
                print(train_labels, predicted_train_labels)
                self.compute_confusion_matrix(train_labels, predicted_train_labels)
                
                ##Computing loss
                train_loss = self.loss(predicted_train_labels,train_labels)*2e2
                train_loss.backward()
                
                self.optimizer.step()
                current_epoch_train_loss += train_loss.item()
                

            current_epoch_train_loss = current_epoch_train_loss/len(train_data_loader.dataset)
            print("Training Losses at epoch: ", epoch + 1, " is: ", current_epoch_train_loss)
            
            self.model.eval()

            for dev_batch_id, (X_dev,dev_labels) in enumerate(dev_data_loader):
                
                dev_labels = dev_labels.to(torch.device(self.device))
                dev_labels_predicted = self.model(X_dev)
                dev_labels_predicted = dev_labels_predicted.to(torch.device(self.device))
                
                self.compute_confusion_matrix(y_true=dev_labels, y_pred=dev_labels_predicted)
                
                ##Computing development loss
                dev_loss = self.loss(dev_labels_predicted,dev_labels)
                current_epoch_dev_loss += dev_loss.item()

        
            current_epoch_dev_loss = current_epoch_dev_loss/len(dev_data_loader.dataset)

            print("Development Losses at epoch:", epoch + 1, " is: ", current_epoch_dev_loss)


            if(((epoch+1) % self.log_freq) == 0):

                if(self.wandb):
                    wandb.log({'train_loss':current_epoch_train_loss, 'dev_loss':current_epoch_dev_loss})
                
                self.save_checkpoint(epoch, train_loss,"model_"+str(epoch)+".pt")
            

        self.save_model()

        return


    def evaluate(self):

        test_data_loader = DataLoader(self.test_data_prep_obj, batch_size=76, shuffle=False)
        self.model.eval()

        print("value of Evaluation for Testing Dataset")
        
        for test_batch_id, (X_test,test_labels) in enumerate(test_data_loader):
            print("Value of test batch id: ", test_batch_id)        
            test_labels = test_labels.to(torch.device(self.device))
            test_labels_predicted = self.model(X_test)
            test_labels_predicted = test_labels_predicted.to(torch.device(self.device))

            self.compute_confusion_matrix(y_true=test_labels, y_pred=test_labels_predicted)
                
        
        return

    
    def setup_wandb_loggers(self):
        

        config = dict(learning_rate = self.learning_rate, weight_decay=self.weight_decay, batch_size = self.batch_size, time_horizon=self.time_horizon)

        wandb.init(project='screw_model',config=config)
        wandb.watch(self.model, log_freq=self.log_freq)
        


    def save_checkpoint(self, current_epoch, curent_losses, filename="model.pt"):
        filepath = self.checkpoint_filepath + filename
        torch.save({'epoch':current_epoch, 'loss':curent_losses,'model_state_dict':self.model.state_dict(), 'optimizer_state_dict':self.optimizer.state_dict()}, filepath)


    def save_model(self, filename="failure_model.pt"):
        filepath = os.path.join(self.model_filepath,filename)
        torch.save({'model_state_dict':self.model.state_dict(), 'optimizer_state_dict':self.optimizer.state_dict()}, filepath)


    def load_model(self, filename):

        print("Loading the model stored in : ", filename)
        checkpoint = torch.load(filename)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    
    def compute_confusion_matrix(self, y_true, y_pred):

        confusion_mat = []
        print("Current Truth: ", torch.max(y_true,1)[1])
        print("Current Predictions: ", torch.max(y_pred,1)[1])
        y_true = torch.argmax(y_true,1).detach().cpu().numpy()
        y_pred = torch.argmax(y_pred,1).detach().cpu().numpy()
            
        confusion_mat = confusion_matrix(y_true, y_pred)
        print("Confusion Matrix: ", confusion_mat, "\n")
        
        return



def main():
    
    parser = argparse.ArgumentParser(description='Screw model training arguments.')
    
    ###Training Hyperparameters
    parser.add_argument('-l', '--learning_rate', type=float, help='learning rate')
    parser.add_argument('-b', '--batch_size', type=int, help='mini-batch size')
    parser.add_argument('-e', '--epochs', type=int, help='number of epochs to train for')
    parser.add_argument('-a', '--weight_decay', type=int, help='Weight decary factor for L1 Regularization')
    parser.add_argument('-C', '--num_classes', type=int, help='The total number of types of failures')
    parser.add_argument('-g', '--gru_dim', type=int, help='Hidden dimension of the GRU layer')
    parser.add_argument('-p', '--gru_layers', type=int, help='Number of Layers in GRU')
    parser.add_argument('-A', '--gru_dropout', type=float, help='Dropout Rate for GRU')

    

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


    ##Encoder Params
    parser.add_argument('-w', '--wrench_dim', default=64,type=int, help='Latent space dim for wrench')
    parser.add_argument('-i', '--image_dim', default=64, type=int, help='Latent space dim for image data')
    parser.add_argument('-t', '--time_horizon', type=int, help='The number time points to consider in a dataset')
    

    ##Input JSON setup file
    parser.add_argument('-f', '--training_param_file', required=True, type=str, help='Training Params Filename')
    

    args = parser.parse_args()
    arg_parse_dict = vars(args)

    package_dir = os.path.join(MAIN_DIR, 'failure_mode_detection')
    config_dir = os.path.join(package_dir, 'config')
    config_filename = os.path.join(config_dir, args.training_param_file)
    ##Defining Network Architecture Specific Params
    with open(config_filename, 'rb') as file:
        training_params_dict = json.load(file)

    arg_parse_dict.update(training_params_dict)

    model_trainer_obj = ModelTrainer(training_params_dict)
    model_trainer_obj.train()
    model_trainer_obj.evaluate()

    return
    

if __name__ == '__main__':
    main()


