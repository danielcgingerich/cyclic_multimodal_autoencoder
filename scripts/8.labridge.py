#!/usr/bin/env python
# coding: utf-8

# In[28]:


import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
from sklearn.preprocessing import StandardScaler

os.chdir('path/to/data/')

X0 = pd.read_csv('7.export_for_model___atac_train.csv', index_col = 0)
X0 = X0.sample(n=10000, random_state=42)
# X0 = pd.DataFrame(scaler.fit_transform(X0), index=X0.index, columns=X0.columns)

multiome_idx = list(X0.index)
random.shuffle(multiome_idx)
size = int(X0.shape[0] * 0.8)
train_idx = multiome_idx[:size]
holdout_idx = multiome_idx[size:]
X0_train = X0.loc[train_idx]
X0_holdout = X0.loc[holdout_idx]
X0_train = X0_train.values
X0_holdout = X0_holdout.values
X0_train_tensor = torch.tensor(X0_train, dtype = torch.float32)
X0_holdout_tensor = torch.tensor(X0_holdout, dtype = torch.float32)

X = pd.read_csv('7.export_for_model___atac_test.csv', index_col = 0)
# X = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
X = X.values
X_tensor = torch.tensor(X, dtype = torch.float32)

X_labels = pd.read_csv('7.export_for_model___labels_test.csv', index_col = 0)
X_labels = X_labels['cell_type'].astype('category')
X_labels_tensor = torch.tensor(X_labels.cat.codes, dtype=torch.long)

paired_labels = pd.read_csv('7.export_for_model___labels_train.csv', index_col = 0)
paired_labels_train = paired_labels.loc[train_idx]
paired_labels_holdout = paired_labels.loc[holdout_idx]
paired_labels_train = paired_labels_train['cell_type'].astype('category')
paired_labels_holdout = paired_labels_holdout['cell_type'].astype('category')
paired_labels_train_tensor = torch.tensor(paired_labels_train.cat.codes, dtype=torch.long)
paired_labels_holdout_tensor = torch.tensor(paired_labels_holdout.cat.codes, dtype=torch.long)


# In[29]:


os.chdir('C:/Users/danie/Desktop/african_american_multiome/data/19.apoe_ccres/')

Y0 = pd.read_csv('7.export_for_model___rna_train.csv', index_col = 0)
# Y0 = Y0.sample(n=10000, random_state=42)
# Y0 = pd.DataFrame(scaler.fit_transform(Y0), index=Y0.index, columns=Y0.columns)

Y0_train = Y0.loc[train_idx]
Y0_holdout = Y0.loc[holdout_idx]
Y0_train = Y0_train.values
Y0_holdout = Y0_holdout.values
Y0_train_tensor = torch.tensor(Y0_train, dtype = torch.float32)
Y0_holdout_tensor = torch.tensor(Y0_holdout, dtype = torch.float32)

Y = pd.read_csv('7.export_for_model___rna_test.csv', index_col = 0)
# Y = pd.DataFrame(scaler.fit_transform(Y), index=Y.index, columns=Y.columns)
Y = Y.values
Y_tensor = torch.tensor(Y, dtype = torch.float32)

Y_labels = pd.read_csv('7.export_for_model___labels_test.csv', index_col = 0)
Y_labels = Y_labels['cell_type'].astype('category')
Y_labels_tensor = torch.tensor(Y_labels.cat.codes, dtype=torch.long)


# In[30]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np

def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

# Encoder model
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, n_celltypes, hidden_dim=[50,40,30], output_dim=30):
        super().__init__()
        self.celltype_embed = nn.Embedding(n_celltypes, embed_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim + embed_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]), 
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]), 
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim[2], output_dim)
        self.fc_logvar = nn.Linear(hidden_dim[2], output_dim)

    def forward(self, x, labels):
        label_embeds = self.celltype_embed(labels)
        x_cat = torch.cat([x, label_embeds], dim=1)
        passing_tone = self.net(x_cat)
        mu = self.fc_mu(passing_tone)
        logvar = self.fc_logvar(passing_tone)
        return mu, logvar

# Decoder model
class Decoder(nn.Module):
    def __init__(self, input_dim, embed_dim, n_celltypes, hidden_dim=[30,40,50], output_dim=50):
        super().__init__()
        self.celltype_embed = nn.Embedding(n_celltypes, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + embed_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]), 
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]), 
            nn.ReLU(),
            nn.Linear(hidden_dim[2], output_dim) 
        )

    def forward(self, x, labels):
        label_embeds = self.celltype_embed(labels)
        x_cat = torch.cat([x, label_embeds], dim=1)
        return self.net(x_cat)


# MultiModal VAE model with two encoders and decoders
class MultiModalVAE(nn.Module):
    def __init__(self, rna_dim, atac_dim, embed_dim, n_celltypes, 
                 hidden_dim_encoder=[50,40,30], hidden_dim_decoder=[30,40,50], 
                 output_dim_encoder=30, output_dim_decoder=50):
        super().__init__()
        self.encoder_rna = Encoder(input_dim=rna_dim, embed_dim=embed_dim, n_celltypes=n_celltypes, 
                                   hidden_dim=hidden_dim_encoder, output_dim=output_dim_encoder)
        self.encoder_atac = Encoder(input_dim=atac_dim, embed_dim=embed_dim, n_celltypes=n_celltypes, 
                                   hidden_dim=hidden_dim_encoder, output_dim=output_dim_encoder)

        self.decoder_rna = Decoder(input_dim=output_dim_encoder, embed_dim=embed_dim, n_celltypes=n_celltypes, 
                                   hidden_dim=hidden_dim_decoder, output_dim=rna_dim)
        self.decoder_atac = Decoder(input_dim=output_dim_encoder, embed_dim=embed_dim, n_celltypes=n_celltypes, 
                                   hidden_dim=hidden_dim_decoder, output_dim=atac_dim)

    def forward(self, rna, atac, rna_labels, atac_labels):
        # Encode both modalities
        mu_rna, logvar_rna = self.encoder_rna(x=rna, labels=rna_labels)
        mu_atac, logvar_atac = self.encoder_atac(x=atac, labels=atac_labels)

        # Sample from the mixture latent distribution
        z_rna = reparameterize(mu_rna, logvar_rna)
        z_atac = reparameterize(mu_atac, logvar_atac)

        # Decode both modalities
        recon_rna = self.decoder_rna(x=mu_rna, labels=rna_labels)
        recon_rna_cycle = self.decoder_rna(x=mu_atac, labels=atac_labels)
        recon_atac = self.decoder_atac(x=mu_atac, labels=atac_labels)
        recon_atac_cycle = self.decoder_atac(x=mu_rna, labels=rna_labels)

        # Return everything needed for loss computation
        return recon_rna, recon_rna_cycle, recon_atac, recon_atac_cycle, mu_rna, logvar_rna, mu_atac, logvar_atac

# VAE loss function with reconstruction loss and KL divergence
def vae_loss(rna, recon_rna, recon_rna_cycle, 
             atac, recon_atac, recon_atac_cycle, 
             mu_rna, logvar_rna, 
             mu_atac, logvar_atac):
    # Reconstruction loss (MSE here; can swap for NB or BCE if needed)
    recon_loss_rna = F.mse_loss(recon_rna, rna, reduction='mean')
    recon_loss_rna_cycle = F.mse_loss(recon_rna_cycle, rna, reduction='mean')
    recon_loss_atac = F.mse_loss(recon_atac, atac, reduction='mean')
    recon_loss_atac_cycle = F.mse_loss(recon_atac_cycle, atac, reduction='mean')

    # KL divergences
    kl_rna = -0.5 * torch.sum(1 + logvar_rna - mu_rna.pow(2) - logvar_rna.exp())
    kl_atac = -0.5 * torch.sum(1 + logvar_atac - mu_atac.pow(2) - logvar_atac.exp())
    
    kl_rna = kl_rna / rna.size(0)
    kl_atac = kl_atac / atac.size(0)
    
    var_rna = logvar_rna.exp()
    var_atac = logvar_atac.exp()
    
    kl_rna_to_atac = 0.5 * torch.sum(
        logvar_atac - logvar_rna +
        (var_rna + (mu_rna - mu_atac).pow(2)) / var_atac - 1
    )

    kl_atac_to_rna = 0.5 * torch.sum(
        logvar_rna - logvar_atac +
        (var_atac + (mu_atac - mu_rna).pow(2)) / var_rna - 1
    )

    kl_integrated = (kl_rna_to_atac + kl_atac_to_rna) / (2 * rna.size(0))

    z_dist = F.mse_loss(mu_rna, mu_atac, reduction='mean')
    
    return recon_loss_rna, recon_loss_rna_cycle, recon_loss_atac, recon_loss_atac_cycle, kl_rna, kl_atac, kl_integrated, z_dist


# In[31]:


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims):
        super().__init__()
        self.T = nn.Parameter(torch.randn(in_features, out_features, kernel_dims))

    def forward(self, x):
        M = x.matmul(self.T.view(x.size(1), -1))  # (batch, out_features * kernel_dims)
        M = M.view(-1, self.T.size(1), self.T.size(2))  # (batch, out_features, kernel_dims)

        batch_size = x.size(0)
        out = []
        for i in range(batch_size):
            diff = M[i].unsqueeze(0) - M  # (batch, out_features, kernel_dims)
            diff_l1 = torch.sum(torch.abs(diff), dim=2)  # (batch, out_features)
            c = torch.exp(-diff_l1)  # (batch, out_features)
            out.append(torch.sum(c, dim=0))  # (out_features,)

        out = torch.stack(out)  # (batch, out_features)
        return torch.cat([x, out], dim=1)  # Append minibatch features

class scrim(nn.Module):
    def __init__(self, input_dim, embed_dim, n_celltypes, hidden_dim=[32,16,8]):
        super().__init__()
        self.celltype_embed = nn.Embedding(n_celltypes, embed_dim)
        self.fc1 = nn.Linear(input_dim + embed_dim, hidden_dim[0])
        self.relu1 = nn.ReLU()
        
        # Add minibatch discrimination after the first hidden layer
        out_features=16
        kernel_dims=3
        self.minibatch_disc = MinibatchDiscrimination(hidden_dim[0], out_features=out_features, kernel_dims=kernel_dims)
        
        # Adjust next layer's input size
        self.fc2 = nn.Linear(hidden_dim[0] + out_features, hidden_dim[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dim[2], 1)

    def forward(self, x, labels):
        subset_indices = torch.randperm(x.size(0))[:100]
        x_subset = x[subset_indices]
        labels_subset = labels[subset_indices]
        
        label_embeds = self.celltype_embed(labels_subset)
        x_cat = torch.cat([x_subset, label_embeds], dim=1)
        h = self.relu1(self.fc1(x_cat))
        h = self.minibatch_disc(h)
        h = self.relu2(self.fc2(h))
        h = self.relu3(self.fc3(h))
        return self.fc4(h)

class vanillaScrim(nn.Module):
    def __init__(self, input_dim, embed_dim, n_celltypes, hidden_dim=[32,16,8], output_dim=1):
        super().__init__()
        self.celltype_embed = nn.Embedding(n_celltypes, embed_dim)

        self.net = nn.Sequential(
            nn.Linear(input_dim + embed_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]), 
            nn.ReLU(),
            nn.Linear(hidden_dim[1], hidden_dim[2]), 
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim[2], output_dim)
        self.fc_logvar = nn.Linear(hidden_dim[2], output_dim)

    def forward(self, x, labels):
        label_embeds = self.celltype_embed(labels)
        x_cat = torch.cat([x, label_embeds], dim=1)
        passing_tone = self.net(x_cat)
        mu = self.fc_mu(passing_tone)
        return mu


# In[32]:


i = Y0_train_tensor.shape[1]
j = X0_train_tensor.shape[1]
print([i,j])
print(Y0_train_tensor.shape)
print(X0_train_tensor.shape)


# In[33]:


from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(Y0_train_tensor, X0_train_tensor, paired_labels_train_tensor)
dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)

# wrap code in
# for rna_batch, atac_batch, labels_batch in dataloader:


# In[34]:


# Assume you already have: train_dataloader and valid_dataloader
def cyclic_kl_weight(epoch, cycle_length=10, step_size=0.001, max_kl_weight=0.1):
    cycle_position = epoch % cycle_length
    max_kl = min(max_kl_weight, epoch * step_size)
    return (cycle_position / cycle_length) * max_kl



lambda_rna = 1
lambda_atac = 1
lambda_kl = 0
lambda_kl_int = 0 
lambda_dist = 10
lambda_gan = 1

Ex = Encoder(input_dim=j, embed_dim=8, n_celltypes=6, output_dim=30)
Ey = Encoder(input_dim=i, embed_dim=8, n_celltypes=6, output_dim=30)
Dx = Decoder(input_dim=30, embed_dim=8, n_celltypes=6, output_dim=j)
Dy = Decoder(input_dim=30, embed_dim=8, n_celltypes=6, output_dim=i)
critic = vanillaScrim(input_dim=30, embed_dim=8, n_celltypes=6)

optimizer_Ex = torch.optim.Adam(Ex.parameters(), lr=1e-4)
optimizer_Ey = torch.optim.Adam(Ey.parameters(), lr=1e-4)
optimizer_Dx = torch.optim.Adam(Dx.parameters(), lr=1e-4)
optimizer_Dy = torch.optim.Adam(Dy.parameters(), lr=1e-4)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=1e-4)

critic_loss = nn.BCEWithLogitsLoss()
recon_loss = nn.MSELoss()


# In[35]:


best_test_loss = np.inf
patience = 500
epochs_without_improvement = 0
max_epoch = 100e3
kl_step = 1/max_epoch * 0.5

epoch = 0
while epochs_without_improvement < patience and epoch < max_epoch:
#     model.train()
    
    for rna_batch, atac_batch, labels_batch in dataloader:
        
        
        # train vae
        optimizer_Ex.zero_grad()
        
        mu_atac, _ = Ex(x=atac_batch, labels=labels_batch)
        mu_rna, _ = Ey(x=rna_batch, labels=labels_batch)
        
        recon_atac = Dx(x=mu_atac, labels=labels_batch)
        recon_atac_cycle = Dx(x=mu_rna, labels=labels_batch)
        
        critic_atac = critic(mu_atac, labels_batch)
        
        ones = torch.ones((critic_atac.shape[0], 1), dtype=torch.float32)
        halves = torch.full((critic_atac.shape[0], 1), 0.5, dtype=torch.float32)
        zeros = torch.zeros((critic_atac.shape[0], 1), dtype=torch.float32)

        recon_atac_loss = recon_loss(recon_atac, atac_batch)
        recon_atac_cycle_loss = recon_loss(recon_atac_cycle, atac_batch)
        atac_gan_loss = critic_loss(critic_atac, halves)
        
        loss_Ex = lambda_atac * (recon_atac_loss + recon_atac_cycle_loss) + lambda_gan * atac_gan_loss
        loss_Ex.backward()
        optimizer_Ex.step()

        
        optimizer_Ey.zero_grad()
        
        mu_atac, _ = Ex(x=atac_batch, labels=labels_batch)
        mu_rna, _ = Ey(x=rna_batch, labels=labels_batch)
        
        recon_rna = Dy(x=mu_rna, labels=labels_batch)
        recon_rna_cycle = Dy(x=mu_atac, labels=labels_batch)
        
        critic_rna = critic(mu_rna, labels_batch) 

        recon_rna_loss = recon_loss(recon_rna, rna_batch)
        recon_rna_cycle_loss = recon_loss(recon_rna_cycle, rna_batch)
        rna_gan_loss = critic_loss(critic_rna, halves)
        
        loss_Ey = lambda_rna * (recon_rna_loss + recon_rna_cycle_loss) + lambda_gan * rna_gan_loss
        loss_Ey.backward()
        optimizer_Ey.step()

        optimizer_Dx.zero_grad()
        mu_atac, _ = Ex(x=atac_batch, labels=labels_batch)
        recon_atac = Dx(x=mu_atac, labels=labels_batch)
        recon_atac_loss = recon_loss(recon_atac, atac_batch)
        loss_Dx = lambda_atac * recon_atac_loss
        loss_Dx.backward()
        optimizer_Dx.step()
        
        
        optimizer_Dy.zero_grad()
        mu_rna, _ = Ey(x=rna_batch, labels=labels_batch)
        recon_rna = Dy(x=mu_rna, labels=labels_batch)
        recon_rna_loss = recon_loss(recon_rna, rna_batch)
        loss_Dy = lambda_rna * recon_rna_loss
        loss_Dy.backward()
        optimizer_Dy.step()
        
        
        # train critic: rna=1, atac=0
        optimizer_critic.zero_grad()
        
        rna_critic = critic(mu_rna.detach(), labels_batch)
        atac_critic = critic(mu_atac.detach(), labels_batch)
        loss_critic = critic_loss(rna_critic, ones) + critic_loss(atac_critic, zeros)
        
        loss_critic.backward()
        optimizer_critic.step()
        
    # holdout loss
#     model.eval()
    with torch.no_grad():
        mu_atac_holdout, _ = Ex(x=X0_holdout_tensor, labels=paired_labels_holdout_tensor)
        mu_rna_holdout, _ = Ey(x=Y0_holdout_tensor, labels=paired_labels_holdout_tensor)
        
        recon_atac_holdout = Dx(x=mu_atac_holdout, labels=paired_labels_holdout_tensor)
        recon_atac_cycle_holdout = Dx(x=mu_rna_holdout, labels=paired_labels_holdout_tensor)
        
        critic_atac_holdout = critic(mu_atac_holdout, paired_labels_holdout_tensor)
        ones = torch.ones((critic_atac_holdout.shape[0], 1), dtype=torch.float32)
        halves = torch.full((critic_atac_holdout.shape[0], 1), 0.5, dtype=torch.float32)
        zeros = torch.zeros((critic_atac_holdout.shape[0], 1), dtype=torch.float32)

        recon_atac_loss_holdout = recon_loss(recon_atac_holdout, X0_holdout_tensor)
        recon_atac_cycle_loss_holdout = recon_loss(recon_atac_cycle_holdout, X0_holdout_tensor)
        atac_gan_loss_holdout = critic_loss(critic_atac_holdout, halves)
        loss_Ex_holdout = lambda_atac * (recon_atac_loss_holdout + recon_atac_cycle_loss_holdout) + lambda_gan * atac_gan_loss_holdout

        
        recon_rna_holdout = Dy(x=mu_rna_holdout, labels=paired_labels_holdout_tensor)
        recon_rna_cycle_holdout = Dy(x=mu_atac_holdout, labels=paired_labels_holdout_tensor)
        
        critic_rna_holdout = critic(mu_rna_holdout, paired_labels_holdout_tensor) 

        recon_rna_loss_holdout = recon_loss(recon_rna_holdout, Y0_holdout_tensor)
        recon_rna_cycle_loss_holdout = recon_loss(recon_rna_cycle_holdout, Y0_holdout_tensor)
        rna_gan_loss_holdout = critic_loss(critic_rna_holdout, halves)
        loss_Ey_holdout = lambda_rna * (recon_rna_loss_holdout + recon_rna_cycle_loss_holdout) + lambda_gan * rna_gan_loss_holdout
        
        test_loss = loss_Ex_holdout + loss_Ey_holdout
        
    if (epoch+1) % 100 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}]")
        print(f"RNA Loss: {(lambda_rna * recon_rna_loss):.4f}, ATAC Loss: {(lambda_atac * recon_atac_loss):.4f}, z loss: {(lambda_dist * z_dist):.4f}")
        print(f"RNA Cycle Loss: {(lambda_rna * recon_rna_cycle_loss):.4f}, ATAC Cycle Loss: {(lambda_atac * recon_atac_cycle_loss):.4f}")
        print(f"GAN Loss: {(lambda_gan * (atac_gan_loss + rna_gan_loss)):.4f}")
        print(f"RNA Holdout Loss: {(lambda_rna * loss_Ey_holdout):.4f}, ATAC Holdout Loss: {(lambda_atac * loss_Ex_holdout):.4f}, Z loss holdout: {(lambda_dist * z_dist_holdout):.4f}")
        print(f"Patience counter: {epochs_without_improvement}")

    if test_loss < best_test_loss:
        best_test_loss = test_loss
        best_state_dict = model.state_dict()
        epochs_without_improvement = 0
        # Optional: save best model
        # torch.save(model.state_dict(), 'best_model.pt')
    else:
        epochs_without_improvement += 1

    epoch += 1

print("Stopped: Validation loss did not improve.")


# In[9]:


os.chdir('C:/Users/danie/Desktop/african_american_multiome/data/19.apoe_ccres/')
model.load_state_dict(best_state_dict)
torch.save(model, '8.laspacebridge___trained_model.pt')


# In[11]:


os.chdir('C:/Users/danie/Desktop/african_american_multiome/data/19.apoe_ccres/')
model = torch.load('8.laspacebridge___trained_model.pt')
recon_rna, recon_rna_cycle, recon_atac, recon_atac_cycle, mu_rna, logvar_rna, mu_atac, logvar_atac = model(rna=Y_tensor, atac=X_tensor, rna_labels=X_labels_tensor, atac_labels=Y_labels_tensor)


# In[12]:


os.chdir('C:/Users/danie/Desktop/african_american_multiome/data/19.apoe_ccres/')

# Convert to numpy arrays
mu_rna_np = mu_rna.detach().cpu().numpy()
mu_atac_np = mu_atac.detach().cpu().numpy()

# Create column names
rna_cols = [f"mu_rna_{i}" for i in range(mu_rna_np.shape[1])]
atac_cols = [f"mu_atac_{i}" for i in range(mu_atac_np.shape[1])]

# Create DataFrames
df_rna = pd.DataFrame(mu_rna_np, columns=rna_cols)
df_atac = pd.DataFrame(mu_atac_np, columns=atac_cols)

# Concatenate and save
df_combined = pd.concat([df_rna, df_atac], axis=1)
df_combined.to_csv("8.laspacebridge___joint_embeds.csv", index=False)

