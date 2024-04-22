import torch
from encoders import WrenchEncoder, ImageEncoder
from decoders import init_weights



class FailureModeDetector(torch.nn.Module):


    def __init__(self, model_params):
        super(FailureModeDetector,self).__init__()
        torch.device("cuda")

        self.wrench_z_dim = model_params["wrench_dim"]
        self.image_z_dim = model_params["image_dim"]
        self.gru_z_dim =  model_params["gru_dim"]

        self.time_horizon = model_params["time_horizon"]

        self.wrench_encoder = WrenchEncoder(self.wrench_z_dim)
        self.image_encoder = ImageEncoder(self.image_z_dim)

        gru_dropout = model_params["gru_dropout"]

        input_dim = self.wrench_z_dim + self.image_z_dim
        
        self.gru_layers = model_params["gru_layers"]
        num_classes = model_params["num_classes"]

        # GRU layers
        self.gru = torch.nn.GRU(input_dim, self.gru_z_dim , self.gru_layers, batch_first=True, dropout=gru_dropout).to(torch.device("cuda"))

        
        self.fc = torch.nn.Sequential(torch.nn.Linear(self.gru_z_dim, int(self.gru_z_dim/2)),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(int(self.gru_z_dim/2), num_classes)).to(torch.device("cuda"))

        
        init_weights(self.modules()) 

    def forward(self, x):
        
        wrench_batch = x[0][:,:,:]
        images = x[1][:]
        
        initial_wrench_shape = wrench_batch.shape
        wrench_batch = torch.reshape(wrench_batch,(wrench_batch.shape[0]*wrench_batch.shape[1],wrench_batch.shape[2],1))
        wrench_latent = self.wrench_encoder(wrench_batch)
        # wrench_latent = torch.reshape(wrench_latent, (initial_wrench_shape[0],1,initial_wrench_shape[1]*self.wrench_z_dim))
        wrench_latent = torch.reshape(wrench_latent, (initial_wrench_shape[0],initial_wrench_shape[1],self.wrench_z_dim))
        
        
        initial_image_shape = images.shape
        images = torch.reshape(images,(images.shape[0]*images.shape[1],images.shape[2],images.shape[3],images.shape[4]))
        image_latent = self.image_encoder(images)
        
        image_latent = torch.reshape(image_latent,(initial_image_shape[0],initial_wrench_shape[1],self.image_z_dim))
        
        
        X_Latent = torch.cat((wrench_latent,image_latent), dim=2).to(torch.device("cuda"))
        
        h0 = torch.zeros(self.gru_layers, initial_wrench_shape[0], self.gru_z_dim).requires_grad_().to(torch.device("cuda"))

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(X_Latent, h0.detach())
        
        out = out[:, -1, :]     ##Taking the output of the last GRU cell
        
        ###This line was added to concatenate
        out = self.fc(out)
        
        return out