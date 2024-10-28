import torch
import torch.nn as nn

class MLP_forward(nn.Module):
    """A simple MLP used for supervised learning
    In order to check the performance of the model
    and how many inputs are needed for learning-kry 24th,10,2024"""
    def __init__(self, layers):
        super(MLP_forward, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = nn.Sigmoid()

    def forward(self, x):
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x

class MLP_forward_fourier(nn.Module):
    """A simple MLP used for supervised learning
    In order to check the performance of the model
    and how many inputs are needed for learning-kry 24th,10,2024"""
    def __init__(self, layers,embedding_dim,input_dim,scale=10.0,device='cuda'):
        super(MLP_forward_fourier, self).__init__()
        self.B = torch.randn((embedding_dim // 2, input_dim)) * scale
        self.B = self.B.to(device)
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = nn.Sigmoid()

    def forward(self, x):

        x = 2.0 * 3.1415926 * x @ self.B.T
        x=torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        for i in range(len(self.layers) - 1):
            x = self.activation(self.layers[i](x))
        x = self.layers[-1](x)
        return x

class MLP_forward_embed(nn.Module):
    """A simple MLP used for supervised learning
    In order to check the performance of the model
    and how many inputs are needed for learning-kry 24th,10,2024"""
    def __init__(self, layers,embedding_dim,input_dim):
        super(MLP_forward_embed, self).__init__()
        self.embedding_dim=embedding_dim
        self.layers = nn.ModuleList()
        self.embedx1=nn.Linear(input_dim,embedding_dim)
        self.embedx2=nn.Linear(embedding_dim,embedding_dim*3)
        self.embedz1=nn.Linear(input_dim,embedding_dim)
        self.embedz2=nn.Linear(embedding_dim,embedding_dim*3)
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        B_x=self.embedx2(self.activation(self.embedx1(x))).unsqueeze(1)
        B_z=self.embedz2(self.activation(self.embedz1(x))).unsqueeze(1)
        B_x=B_x.reshape(B_x.shape[0],3,-1)
        B_z=B_z.reshape(B_z.shape[0],3,-1)  
        # x_pos=x[:,:3].unsqueeze(1)
        # z_pos=x[:,3:].unsqueeze(1)
        # x_pos
        B=torch.hstack([B_x,B_z])
        x=x.unsqueeze(1)
        x = 2.0 * 3.1415926 * x @ B
        x=x.squeeze(1)
        x=torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        for i in range(len(self.layers)):
            x = self.activation(self.layers[i](x))
        # x = self.layers[-1](x)
        return x

