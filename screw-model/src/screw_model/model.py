from sindy_utils import library_size
from encoders import WrenchEncoder, OrientationEncoder
from decoders import OrientationDecoder, WrenchDecoder
import torch
from sklearn.preprocessing import PolynomialFeatures


class Screw_Model(torch.nn.Module):

    def __init__(self, sindy_params):
        super().__init__()

        self.wrench_z_dim = sindy_params["wrench_dim"]
        self.orientation_z_dim = sindy_params["orientation_dim"]

        ##Initializing the pose encoder and decoders
        self.orientation_encoder = OrientationEncoder(sindy_params["orientation_dim"])
        self.orientation_decoder = OrientationDecoder(sindy_params["orientation_dim"])
        
        ###Initializing the wrench encoder and decoders
        self.wrench_encoder = WrenchEncoder(sindy_params["wrench_dim"])
        self.wrench_decoder = WrenchDecoder(sindy_params["wrench_dim"])


        ##Initializing the Sindy model
        self.SinDy  = SinDY(sindy_params).to("cuda")

        self.get_latent_params = True
    

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            output: During training time returns the latent space params as well as the x_dot for loss computation
                    During inference time returns x_dot or x_dot and latent space if (get_latent_params is set to True)
        """

        x_sindy_batch = x[:,:,:2]         ##First two elements are x,y
        orientation_batch = x[:,:,2:5]           ## Next six elements are ABC
        wrench_batch = x[:,:,5:11]          ## Last six elements are Wrench Fx,Fy,Fz,Tx,Ty,Tz
        stiffness_batch = x[:,:,11:17]
        damping_batch = x[:,:,17:23]
        
        x_sindy_batch = x_sindy_batch.to(torch.device("cuda"))
        stiffness_batch = stiffness_batch.to(torch.device("cuda"))
        damping_batch = damping_batch.to(torch.device("cuda"))
        wrench_batch = wrench_batch.to(torch.device("cuda"))
        orientation_batch = orientation_batch.to(torch.device("cuda"))
        
        orientation_latent = self.orientation_encoder(orientation_batch)
        orientation_decoded = self.orientation_decoder(orientation_latent)
        
        initial_wrench_shape = wrench_batch.shape
        wrench_batch_reshaped = torch.reshape(wrench_batch,(wrench_batch.shape[0]*wrench_batch.shape[1],wrench_batch.shape[2]))
        wrench_latent = self.wrench_encoder(wrench_batch_reshaped)
        wrench_decoded = self.wrench_decoder(wrench_latent)
        
        wrench_latent = torch.reshape(wrench_latent, (initial_wrench_shape[0],initial_wrench_shape[1],self.wrench_z_dim))
        wrench_decoded = torch.reshape(wrench_decoded, initial_wrench_shape)
        
        X_U = torch.cat((x_sindy_batch,orientation_latent,wrench_latent), dim=2).to(torch.device("cuda"))
        
        x_dot_batch = self.SinDy(X_U)
        
        if(self.training or (self.get_latent_params)):
            output = [x_dot_batch, orientation_decoded, wrench_decoded]

        else:
          output = x_dot_batch

        
        return output





class ScrewMeanMotion_Model(torch.nn.Module):

    def __init__(self, input_params):
        super().__init__()

        ###Initializing the pose encoder and decoders
        torch.device("cuda")
        
        input_layer_dim = input_params["orientation_dim"] + input_params["stiffness_dim"] + input_params["damping_dim"]
        self.get_latent_params = True

        self.screw_model = torch.nn.Sequential(
            torch.nn.Linear(input_layer_dim, input_layer_dim*4),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(input_layer_dim*4, input_layer_dim*6),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(input_layer_dim*6, input_layer_dim*4),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(4*input_layer_dim, 2*input_layer_dim),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(2*input_layer_dim, input_layer_dim),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Linear(input_layer_dim, 3),
        ).to("cuda")
    

    def forward(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            output: During training time returns the latent space params as well as the x_dot for loss computation
                    During inference time returns x_dot or x_dot and latent space if (get_latent_params is set to True)
        """

        ##2 here is just a random entry chosen for computing value
        ## It is important that these values are kept more or less constant
        
        orientation_batch = x[:,2,0:3]           ## Next six elements are ABC
        stiffness_batch = x[:,2,3:9]
        damping_batch = x[:,2,9:15]

        stiffness_batch = stiffness_batch.to(torch.device("cuda"))
        damping_batch = damping_batch.to(torch.device("cuda"))
        orientation_batch = orientation_batch.to(torch.device("cuda"))
        
        stiffness_batch = stiffness_batch.to(torch.device("cuda"))
        damping_batch = damping_batch.to(torch.device("cuda"))

        X = torch.cat((orientation_batch,stiffness_batch,damping_batch), dim=1).to(torch.device("cuda"))
        
        output = self.screw_model(X)
        # output = torch.sigmoid(output)
        output = torch.reshape(output,(output.shape[0],1,output.shape[1]))
        
        
        return output



##SinDY Model
class SinDY(torch.nn.Module):

    def __init__(self, params):
        """_summary_

        Args:
            model_order (int, optional): The model order decides the derivatives to predict. Defaults to 1.
            poly_oder (int, optional): Max polynomial order tow hich we build SinDY system. Defaults to 2.
            include_sine (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.device = params['device']
        
        ###Defining dimensions
        self.poly_order = params['poly_order']
        self.state_var_dim = params['state_space_dim']            ##This is the dimension of the control variables in this case it is [px,py]
        self.latent_dim = params['latent_dim']
        # self.latent_dim = 5
        self.include_sine = params['include_sine']
        
        
        self.library_dim = library_size(self.latent_dim, self.poly_order, use_sine=self.include_sine)
        
        self.model_params = torch.nn.ParameterDict({'sindy_coefficients':torch.nn.Parameter(torch.randn(self.library_dim, self.state_var_dim))})
        torch.nn.init.xavier_uniform_(self.model_params['sindy_coefficients'])
        ##Training script will update this mask
        self.coefficient_mask = torch.ones((self.library_dim, self.state_var_dim), device=self.device, dtype=torch.float32)
        self.Theta = PolynomialFeatures(self.poly_order)
        
    def forward(self, x):
        """_summary_

        Args:
            x (_type_): Input to the SinDy Algo. In this case since Theta(X,U), x = [X,U]

        Returns:
            _type_: _description_
        """
        if(len(x.shape) !=3 ):
            print("Ensure that if batch size is 1 send in a tensor of (1,library_dim, latent_dim)")
            return x
        
        x_dot_batch = torch.Tensor(size=(x.shape[0],x.shape[1],self.state_var_dim))
        
        for i in range(x_dot_batch.shape[0]):
            x_input = x[i].detach().cpu().numpy()
            theta = torch.from_numpy(self.Theta.fit_transform(x_input))
            theta = theta.to(torch.device(self.device))
            if(self.include_sine):
                theta = torch.hstack((theta,torch.sin(x[i])))
            x_dot_batch[i] = torch.matmul(theta,self.coefficient_mask*self.model_params['sindy_coefficients'])

        
        return x_dot_batch.to("cuda")