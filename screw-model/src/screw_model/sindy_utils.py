import numpy as np
from scipy.special import binom
from scipy.integrate import odeint
import torch


def sindy_library_order2(X, dX, poly_order, include_sine=False):
    m,n = X.shape
    l = library_size(2*n, poly_order, include_sine, True)
    library = np.ones((m,l))
    index = 1

    X_combined = np.concatenate((X, dX), axis=1)

    for i in range(2*n):
        library[:,index] = X_combined[:,i]
        index += 1

    if poly_order > 1:
        for i in range(2*n):
            for j in range(i,2*n):
                library[:,index] = X_combined[:,i]*X_combined[:,j]
                index += 1

    if poly_order > 2:
        for i in range(2*n):
            for j in range(i,2*n):
                for k in range(j,2*n):
                    library[:,index] = X_combined[:,i]*X_combined[:,j]*X_combined[:,k]
                    index += 1

    if poly_order > 3:
        for i in range(2*n):
            for j in range(i,2*n):
                for k in range(j,2*n):
                    for q in range(k,2*n):
                        library[:,index] = X_combined[:,i]*X_combined[:,j]*X_combined[:,k]*X_combined[:,q]
                        index += 1
                    
    if poly_order > 4:
        for i in range(2*n):
            for j in range(i,2*n):
                for k in range(j,2*n):
                    for q in range(k,2*n):
                        for r in range(q,2*n):
                            library[:,index] = X_combined[:,i]*X_combined[:,j]*X_combined[:,k]*X_combined[:,q]*X_combined[:,r]
                            index += 1

    if include_sine:
        for i in range(2*n):
            library[:,index] = np.sin(X_combined[:,i])
            index += 1

    return library


def sindy_fit(RHS, LHS, coefficient_threshold):
    m,n = LHS.shape
    Xi = np.linalg.lstsq(RHS,LHS, rcond=None)[0]
    
    for k in range(10):
        small_inds = (np.abs(Xi) < coefficient_threshold)
        Xi[small_inds] = 0
        for i in range(n):
            big_inds = ~small_inds[:,i]
            if np.where(big_inds)[0].size == 0:
                continue
            Xi[big_inds,i] = np.linalg.lstsq(RHS[:,big_inds], LHS[:,i], rcond=None)[0]
    return Xi


def sindy_simulate(x0, t, Xi, poly_order, include_sine):
    m = t.size
    n = x0.size
    f = lambda x,t : np.dot(sindy_library(np.array(x).reshape((1,n)), poly_order, include_sine), Xi).reshape((n,))

    x = odeint(f, x0, t)
    return x


def sindy_simulate_order2(x0, dx0, t, Xi, poly_order, include_sine):
    m = t.size
    n = 2*x0.size
    l = Xi.shape[0]

    Xi_order1 = np.zeros((l,n))
    for i in range(n//2):
        Xi_order1[2*(i+1),i] = 1.
        Xi_order1[:,i+n//2] = Xi[:,i]
    
    x = sindy_simulate(np.concatenate((x0,dx0)), t, Xi_order1, poly_order, include_sine)
    return x




def library_size(n, poly_order, use_sine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if use_sine:
        l += n
    if not include_constant:
        l -= 1
    return l


def sindy_library(X, poly_order, include_sine=False):
    m,n = X.shape
    l = library_size(n, poly_order, include_sine, True)
    library = np.ones((m,l))
    index = 1

    for i in range(n):
        library[:,index] = X[:,i]
        index += 1

    if poly_order > 1:
        for i in range(n):
            for j in range(i,n):
                library[:,index] = X[:,i]*X[:,j]
                index += 1

    if poly_order > 2:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    library[:,index] = X[:,i]*X[:,j]*X[:,k]
                    index += 1

    if poly_order > 3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]
                        index += 1
                    
    if poly_order > 4:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        for r in range(q,n):
                            library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]*X[:,r]
                            index += 1

    if include_sine:
        for i in range(n):
            library[:,index] = np.sin(X[:,i])
            index += 1

    return library



def sindy_library_torch(z, poly_order, include_sine=False):
    """
    Build the SINDy library.
    Arguments:
        z - 2D tensor array of the snapshots on which to build the library. Shape is number of
        time points by the number of state variables.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.
    Returns:
        2D tensor array containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """

    if(not torch.is_tensor(z)):
        print("Tensor not found in z, converting to tensor")
        z = torch.Tensor(z)
    
    # if(torch.has_cuda()):
    latent_dim = z.shape[1]
    
    library = [torch.ones(z.shape[0],device="cuda")]

    for i in range(latent_dim):
        library.append(z[:,i])

    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(torch.multiply(z[:,i], z[:,j]))

    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i]*z[:,j]*z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q])

    if include_sine:
        for i in range(latent_dim):
            library.append(torch.sin(z[:,i]))

    return torch.stack(library, axis=1)



def sindy_library_torch_order2(z, dz, latent_dim, poly_order, include_sine=False):
    """
    Build the SINDy library for a second order system. This is essentially the same as for a first
    order system, but library terms are also built for the derivatives.
    """

    if(not torch.is_tensor(z)):
        print("Tensor not found in z, converting to tensor")
        z = torch.Tensor(z)
    
    z.to("cuda")

    library = [torch.ones(z.shape[0])]

    z_combined = torch.concat([z, dz], 1)

    for i in range(2*latent_dim):
        library.append(z_combined[:,i])

    if poly_order > 1:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                library.append(torch.multiply(z_combined[:,i], z_combined[:,j]))

    if poly_order > 2:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k])

    if poly_order > 3:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    for p in range(k,2*latent_dim):
                        library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p])

    if poly_order > 4:
        for i in range(2*latent_dim):
            for j in range(i,2*latent_dim):
                for k in range(j,2*latent_dim):
                    for p in range(k,2*latent_dim):
                        for q in range(p,2*latent_dim):
                            library.append(z_combined[:,i]*z_combined[:,j]*z_combined[:,k]*z_combined[:,p]*z_combined[:,q])

    if include_sine:
        for i in range(2*latent_dim):
            library.append(torch.sin(z_combined[:,i]))

    return torch.stack(library, axis=1)


def z_derivative(input, dx, weights, biases, activation='elu'):
    """
    Compute the first order time derivatives by propagating through the network.
    Arguments:
        input - 2D Tensor array, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        weights - List of Tensor arrays containing the network weights
        biases - List of Tensor arrays containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.
    Returns:
        dz - Tensor array, first order time derivatives of the network output.
    """
    dz = dx
    if activation == 'elu':
        for i in range(len(weights)-1):
            input = torch.matmul(input, weights[i]) + biases[i]
            dz = torch.multiply(torch.minimum(torch.exp(input),1.0),
                                  torch.matmul(dz, weights[i]))
            input = torch.nn.ELU(input)
        dz = torch.matmul(dz, weights[-1])
    elif activation == 'relu':
        for i in range(len(weights)-1):
            input = torch.matmul(input, weights[i]) + biases[i]
            dz = torch.multiply(torch.float(input>0), torch.matmul(dz, weights[i]))
            input = torch.nn.ReLU(input)
        dz = torch.matmul(dz, weights[-1])
    elif activation == 'sigmoid':
        for i in range(len(weights)-1):
            input = torch.matmul(input, weights[i]) + biases[i]
            input = torch.sigmoid(input)
            dz = torch.multiply(torch.multiply(input, 1-input), torch.matmul(dz, weights[i]))
        dz = torch.matmul(dz, weights[-1])
    else:
        for i in range(len(weights)-1):
            dz = torch.matmul(dz, weights[i])
        dz = torch.matmul(dz, weights[-1])
    
    return dz



def z_derivative_order2(input, dx, ddx, weights, biases, activation='elu'):
    """
    Compute the first and second order time derivatives by propagating through the network.
    Arguments:
        input - 2D tensor array, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        ddx - Second order time derivatives of the input to the network.
        weights - List of tensor arrays containing the network weights
        biases - List of tensor arrays containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.
    Returns:
        dz - tensor array, first order time derivatives of the network output.
        ddz - tensor array, second order time derivatives of the network output.
    """
    dz = dx
    ddz = ddx
    if activation == 'elu':
        for i in range(len(weights)-1):
            input = torch.matmul(input, weights[i]) + biases[i]
            dz_prev = torch.matmul(dz, weights[i])
            elu_derivative = torch.minimum(torch.exp(input),1.0)
            elu_derivative2 = torch.multiply(torch.exp(input), torch.float(input<0))
            dz = torch.multiply(elu_derivative, dz_prev)
            ddz = torch.multiply(elu_derivative2, torch.square(dz_prev)) \
                  + torch.multiply(elu_derivative, torch.matmul(ddz, weights[i]))
            input = torch.nn.ELU(input)
        dz = torch.matmul(dz, weights[-1])
        ddz = torch.matmul(ddz, weights[-1])
    elif activation == 'relu':
        # NOTE: currently having trouble assessing accuracy of 2nd derivative due to discontinuity
        for i in range(len(weights)-1):
            input = torch.matmul(input, weights[i]) + biases[i]
            relu_derivative = torch.float(input>0)
            dz = torch.multiply(relu_derivative, torch.matmul(dz, weights[i]))
            ddz = torch.multiply(relu_derivative, torch.matmul(ddz, weights[i]))
            input = torch.nn.ReLU(input)
        dz = torch.matmul(dz, weights[-1])
        ddz = torch.matmul(ddz, weights[-1])
    elif activation == 'sigmoid':
        for i in range(len(weights)-1):
            input = torch.matmul(input, weights[i]) + biases[i]
            input = torch.sigmoid(input)
            dz_prev = torch.matmul(dz, weights[i])
            sigmoid_derivative = torch.multiply(input, 1-input)
            sigmoid_derivative2 = torch.multiply(sigmoid_derivative, 1 - 2*input)
            dz = torch.multiply(sigmoid_derivative, dz_prev)
            ddz = torch.multiply(sigmoid_derivative2, torch.square(dz_prev)) \
                  + torch.multiply(sigmoid_derivative, torch.matmul(ddz, weights[i]))
        dz = torch.matmul(dz, weights[-1])
        ddz = torch.matmul(ddz, weights[-1])
    else:
        for i in range(len(weights)-1):
            dz = torch.matmul(dz, weights[i])
            ddz = torch.matmul(ddz, weights[i])
        dz = torch.matmul(dz, weights[-1])
        ddz = torch.matmul(ddz, weights[-1])
    return dz,ddz