import torch
import torch.nn as nn
import torch.nn.functional as F


# Simple MLP layer used in the SSM scaffolding later on, can be modified
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # Define the model using nn.Sequential
        self.model = nn.Sequential(nn.Linear(input_size, hidden_size, bias=False),  # First layer
                                   nn.SiLU(),  # Activation after the first layer
                                   nn.Linear(hidden_size, hidden_size, bias=False),  # Hidden layer
                                   nn.ReLU(),  # Activation after hidden layer
                                   nn.Linear(hidden_size, output_size, bias=False)  # Output layer (no activation)
                                   )

    def forward(self, x):
        if x.dim() == 3:
            # x is of shape (batch_size, sequence_length, input_size)
            batch_size, seq_length, input_size = x.size()

            # Flatten the batch and sequence dimensions for the MLP
            x = x.reshape(-1, input_size)

            # Apply the MLP to each feature vector
            x = self.model(x)  # Shape: (batch_size * sequence_length, output_size)

            # Reshape back to (batch_size, sequence_length, output_size)
            output_size = x.size(-1)
            x = x.reshape(batch_size, seq_length, output_size)
        else:
            # If x is not 3D, just apply the MLP directly
            x = self.model(x)

        return x


class HamiltonianSIE(nn.Module):
    # Hamiltonian neural network
    # General ODE: \dot{y} = J(y,t) K(t) \tanh( K^T(t) y(t) + b(t) )
    # Constraints:
    #   J(y,t) = J_1 = [ 0 I ; -I 0 ]
    # Discretization method: Semi-Implicit Euler
    def __init__(self, n_layers, nf=4, t_end=0.5, random=True, bias=True):
        super().__init__()

        self.n_layers = n_layers
        self.h = t_end / self.n_layers
        self.act = nn.Tanh()
        if not nf % 2 == 0:
            raise ValueError('Number of features need to be and even number -- Currently it is %i' % nf)
        self.nf = nf
        self.half_state_dim = nf//2
        if random:
            k1 = 0.1 * torch.randn(self.half_state_dim, self.half_state_dim).repeat(self.n_layers, 1, 1)
            k2 = 0.1 * torch.randn(self.half_state_dim, self.half_state_dim).repeat(self.n_layers, 1, 1)
            b1 = 0.1 * torch.randn(1, self.half_state_dim).repeat(self.n_layers, 1, 1)
            b2 = 0.1 * torch.randn(1, self.half_state_dim).repeat(self.n_layers, 1, 1)
        else:
            k1 = torch.eye(self.half_state_dim).repeat(self.n_layers, 1, 1)
            k2 = torch.eye(self.half_state_dim).repeat(self.n_layers, 1, 1)
            b1 = torch.zeros(self.n_layers, 1, self.half_state_dim)
            b2 = torch.zeros(self.n_layers, 1, self.half_state_dim)

        self.k1 = nn.Parameter(k1)
        self.k2 = nn.Parameter(k2)
        if bias:
            self.b1 = nn.Parameter(b1)
            self.b2 = nn.Parameter(b2)
        else:
            self.b1 = torch.zeros(self.n_layers, 1, self.half_state_dim)
            self.b2 = torch.zeros(self.n_layers, 1, self.half_state_dim)

    def forward(self, x0, ini=0, end=None):
        # the size of x0 is (sampleNumber, 1, nf)
        if end is None:
            end = self.n_layers
        # x = x0.clone()
        p, q = torch.split(x0.clone(), [self.half_state_dim, self.half_state_dim], dim=2)
        for j in range(ini, end):
            p = p - self.h * F.linear(self.act(F.linear(q, self.k2[j].transpose(0,1)) + self.b1[j]), self.k2[j])
            q = q + self.h * F.linear(self.act(F.linear(p, self.k1[j].transpose(0,1)) + self.b2[j]), self.k1[j])
        x = torch.cat([p, q], dim=2)
        return x


class FCNN(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, act=nn.Tanh):
        super(FCNN, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(dim_in, dim_hidden, bias=False), act(),
            # nn.Linear(hidden_dim, hidden_dim), act(),
            nn.Linear(dim_hidden, dim_out, bias=False)
        )

    def forward(self, x):
        return self.network(x)


class CouplingLayer(nn.Module):
    """
    An implementation of a coupling layer from RealNVP (https://arxiv.org/abs/1605.08803).
    """
    def __init__(self, dim_inputs, dim_hidden):
        super(CouplingLayer, self).__init__()

        self.dim_inputs = dim_inputs
        self.register_buffer('mask', torch.arange(0, dim_inputs) % 2)   # alternating inputs

        self.scale_net = FCNN(dim_in=dim_inputs, dim_out=dim_inputs, dim_hidden=dim_hidden)
        self.translate_net = FCNN(dim_in=dim_inputs, dim_out=dim_inputs, dim_hidden=dim_hidden)

        nn.init.normal_(self.translate_net.network[0].weight.data, std=0.1)
        nn.init.normal_(self.translate_net.network[2].weight.data, std=0.1)

        nn.init.normal_(self.scale_net.network[0].weight.data, std=0.1)
        nn.init.normal_(self.scale_net.network[2].weight.data, std=0.1)

    def forward(self, inputs, mode='direct'):
        mask = self.mask
        masked_inputs = inputs * mask

        log_s = self.scale_net(masked_inputs) * (1 - mask)
        t = self.translate_net(masked_inputs) * (1 - mask)

        if mode == 'direct':
            s = torch.exp(log_s)
            return inputs * s + t
        else:
            s = torch.exp(-log_s)
            return (inputs - t) * s
