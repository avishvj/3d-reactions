# from typing import ?
import torch
import torch.nn as nn
from numpy import exp, sqrt
from numpy.random import normal

class VAE(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # linear (size of input, 2d), size of input= max possible size i.e. largest mol
            nn.Linear(input_size, d**2),
            nn.ReLU(),
            nn.Linear(d ** 2, d * 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(d, d ** 2),
            nn.ReLU(),
            nn.Linear(d ** 2, input_size)
            # would use sigmoid here if input was between 0 and 1
        )

    def reparameterise(self, mean_z, log_var_z):
        if self.training:
            # eps = normal(loc=0, scale=1, size=(len(graphs.nodes), self.latent_dim=2d))
            # since variances only positive, computing log allows you to output full real range for encoder
            eps = normal(0, 1, size=(len(input_nodes), latent_dims))
            z = mean_z + eps * sqrt(exp(log_var_z))
            return z
        else:
            return mean_z

    def forward(self, x):
        # reshape input into a vector, then reshape using view(-1, batchsize=2, d)
        params_z = self.encoder(x.view(-1, input_size)).view(-1, 2, d)
        
        mean_z = params_z[:, 0, :]
        log_var_z = params_z[:, 1, :]
        z = self.reparameterise(mean_z, log_var_z)
        return self.decoder(z), mean_z, log_var_z 

model = VAE().to(device)

# setting optimiser
learning_rate = 1e-3
optimiser = torch.optim.Adam(model.parameters(), lr = learning_rate)

# reconstruction + KL divergence losses summed over all elements
def loss_function(z_hat, x, mean_z, log_var_z):
    # binary cross entropy between input and reconstruction
    BCE = nn.functional.binary_cross_entropy(x_hat, x.view(-1, 784), reduction='sum') 
    # kl divergence: var is linear, - log var is logarithmic, mean is squared 
    KLD = 0.5 * torch.sum(exp(log_var_z) - log_var_z - 1 + mean_z**2)
    return BCE + KLD

# training and testing the VAE
epochs = 5
codes = dict(mean=list(), log_var=list(), y=list())
for epoch in range(0, epochs+1):
    # training
    if epoch > 0:
        model.train()
        train_loss = 0
        for x, _ in train_loader:
            x = x.to(device)
            # === forward ===
            x_hat, mean, log_var = model(x)
            loss = loss_function(x_hat, x, mean, log_var)
            train_loss += loss.item()
            # === backward ===
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        # === log ===
        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
        
    # testing
    means, log_vars, labels = list(), list(), list()
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for x, y in test_loader:
            x = x.to(device)
            # === forward ===
            x_hat, mean, log_var = model(x)
            test_loss += loss_function(x_hat, x, mean, log_var).item()
            # === log ===
            means.append(mean.detach())
            log_vars.append(log_var.detach())
            labels.append(y.detach())
    # === log ===
    codes['mean'].append(torch.cat(means))
    codes['log_var'].append(torch.cat(log_vars))
    codes['y'].append(torch.cat(labels))
    test_loss /= len(test_loader.dataset)
    print(f'===> Test set loss: {test_loss:.4f}')
    display_images(x, x_hat, 1, f'Epoch {epoch}')


# generating a few samples
N = 16
z = torch.randn((N, d)).to(device)
sample = model.decoder(z)
display_images(None, sample, N//4, count=True)

# Choose starting and ending point for the interpolation -> shows original and reconstructed

A, B = 1, 14
sample = model.decoder(torch.stack((mean[A].data, mean[B].data), 0))
display_images(None, torch.stack(((
    x[A].data.view(-1),
    x[B].data.view(-1),
    sample.data[0],
    sample.data[1]
)), 0))
