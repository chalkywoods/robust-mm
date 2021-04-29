import torch
import numpy as np
import random
import tqdm
import itertools

from .models import DeepGCCA
from .wgcca import WeightedGCCA
from .loss_objectives import GCCA_loss

class DGCCA:
  def __init__(self, layers, out_dim, device='cpu', use_all_singular_values=False):
    self.device = device
    self.modalities = len(layers)
    self.out_dim = out_dim
    self.input_shapes = [layer[0] for layer in layers]
    self.checkpoint = False
    layer_sizes = [layer[1:] + [out_dim] for layer in layers]
    self.model = DeepGCCA(layer_sizes, self.input_shapes, out_dim, use_all_singular_values, device).double().to(device)

  def train(self, data, epochs=100, batch_size=128, lr=1e-4, criterion=GCCA_loss, cca_dim=1, cca_hidden_dim=1000, incremental=False):
    if not self.checkpoint:
      self.train_reduction(epochs=epochs, data=data, batch_size=batch_size, lr=lr, criterion=criterion)
    self.train_linear_gcca(data, cca_dim, cca_hidden_dim, incremental=incremental)

  def train_reduction(self, epochs=100, data=None, batch_size=128, lr=1e-4, criterion=GCCA_loss):
    #optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.5)
    optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    data = [modality.to(self.device) for modality in data]

    self.model.train()
    minibatch_indices = []
    for i in range(0, int(np.ceil(data[0].shape[0]/batch_size))):
      sidx = i*batch_size
      eidx = sidx + batch_size
      minibatch_indices.append( (sidx, eidx) )
    
    random.shuffle(minibatch_indices)
    nans = 0
    for epoch in range(epochs):
      total_loss = 0
      with tqdm.tqdm(total=len(minibatch_indices)) as pbar_train:
        for batch, (start, stop) in enumerate(minibatch_indices):
          optimizer.zero_grad()
          out = self.model([modality[start:stop] for modality in data])
          try:
            loss = criterion(out)
          except AssertionError:
            print('\n')
            print(sum([torch.isnan(modality[start:stop]).sum().item() for modality in data]))
            print([torch.isnan(mod_out).sum().item() for mod_out in out])
            raise
          total_loss += loss
          total_loss_avg = total_loss / ((batch + 1))
          loss.backward()
          optimizer.step()
          pbar_train.update(1)
          pbar_train.set_description('Epoch {:3}/{}, Loss: {:.4f}'.format(epoch + 1, epochs, total_loss_avg))
      scheduler.step()
      random.shuffle(minibatch_indices)

  def train_linear_gcca(self, data, cca_dim=1, cca_hidden_dim=1000, batch_size=128, incremental=False):
    self.cca_dim = cca_dim
    self.wgcca = WeightedGCCA(self.modalities, [self.out_dim]*self.modalities, cca_dim, [1.e-12]*self.modalities, cca_hidden_dim, viewWts=None, verbose=True)
    self.model.eval()
    with tqdm.tqdm(total=int(data[0].shape[0]/batch_size)+1) as pbar_embed:
      train_out = []
      pbar_embed.set_description('Embedding training set:')

      for i in range(0, data[0].shape[0], batch_size):
        batch_data = torch.stack([mod[i:i+batch_size,:] for mod in data]).to(self.device)
        train_out.append(np.array([mod.detach().numpy() for mod in self.model(batch_data)]))
        del batch_data
        pbar_embed.update(1)
      #train_out = [out.detach().numpy() for out in self.model(data)]
      
    train_out = np.concatenate(train_out, axis=1)
    train_out = [mod for mod in train_out]
    self.wgcca.learn(train_out, incremental=incremental)

  def get_corrs(self, data, embedding_dim=None, window=None, weighting=np.mean, return_corrs=False):
    self.model.eval()
    data_reduced = [reduced.detach().numpy() for reduced in self.model(data)]
    return self.get_linear_corrs(data_reduced, embedding_dim, window=window, weighting=weighting, return_corrs=return_corrs)

  def get_linear_corrs(self, data, embedding_dim=None, window=None, weighting=np.mean, return_corrs=False):
    embedding_dim = self.cca_dim if embedding_dim is None else embedding_dim
    if return_corrs:
      all_corrs = np.zeros((data[0].shape[0] - window, self.modalities, self.modalities))
    else:
      corrs = np.zeros((self.modalities, self.modalities))
      if window is not None:
        stds = np.zeros((self.modalities, self.modalities))
    length = data[0].shape[0]
    embeddings = []
    for i in range(self.modalities):
      if return_corrs:
        all_corrs[:,i,i] = 1
      else:
        corrs[i,i] = 1
      embeddings.append(self.wgcca.apply(data, K=self.get_K(i, length)))
    for i, j in itertools.combinations(range(self.modalities), 2):
      if return_corrs:
        _, _, all_corrs[:,i,j] = window_corr(embeddings[i][:,0:embedding_dim], embeddings[j][:,0:embedding_dim], window, weighting=weighting)
      else:
        if window is None:
          if weighting == 'flat':
            corrs[i, j] = flat_corr(embeddings[i][:,0:embedding_dim], embeddings[j][:,0:embedding_dim])
          else:
            corrs[i, j] = combined_corr(embeddings[i][:,0:embedding_dim], embeddings[j][:,0:embedding_dim], weighting=weighting)
          corrs[j, i] = corrs[i, j]
        else:
          corrs[i, j], stds[i, j], _ = window_corr(embeddings[i][:,0:embedding_dim], embeddings[j][:,0:embedding_dim], window, weighting=weighting)
          corrs[j, i] = corrs[i, j]
          stds[j, i] = stds[i, j]
    if return_corrs:
      return all_corrs
    else:
      if window is None:
        return corrs
      else:
        return corrs, stds

  def get_K(self, modality, length):
    K = np.zeros((length, self.modalities))
    K[:,modality] = 1
    return K

  def load_checkpoint(self, path):
    self.model.load_state_dict(torch.load(path))
    self.checkpoint = True

  def save_checkpoint(self, path):
    torch.save(self.model.state_dict(), path)

  def get_all_embeddings(self, data):
    reduced_data = self.model(data)
    cca_embeddings = []
    for modality in range(self.modalities):
      K = self.get_K(modality, reduced_data[0].shape[0])
      cca_embeddings.append(self.wgcca.apply([mod.detach().numpy() for mod in reduced_data], K=K))
    return cca_embeddings


  def get_embedding(self, data, modality, batch_size=512):
    reduced_data = []
    for i in range(0, data.shape[0], batch_size):
      reduced_data.append(self.model.model_list[modality](data[i:i+batch_size,:]).detach().numpy())
    reduced_data = np.concatenate(reduced_data, axis=0)
    K = self.get_K(modality, reduced_data.shape[0])
    return self.wgcca.apply([reduced_data]*self.modalities, K=K)


def combined_corr(x, y, weighting=np.mean):
  dim = x.shape[1]
  corrmatrix = np.corrcoef(x, y, rowvar=False)
  corrs = []
  for i in range(dim):
      corrs.append(corrmatrix[i,i+dim])
  return weighting(np.nan_to_num(corrs))

def flat_corr(x, y):
  x = x.reshape((-1,))
  y = y.reshape((-1,))
  return np.corrcoef(x, y)[0,1]

def window_corr(x, y, window, stride=False, weighting=np.mean):
  corrs = []
  stride = stride if stride else 1
  for i in range(0, x.shape[0] - window, stride):
    if weighting == 'flat':
      corrs.append(flat_corr(x[i:i+window,:], y[i:i+window]))
    else:
      corrs.append(combined_corr(x[i:i+window,:], y[i:i+window], weighting=weighting))
  return np.mean(corrs), np.std(corrs), corrs

@staticmethod
def signed_rms(x):
  return np.sqrt(np.mean([y*np.abs(y) for y in x]))
