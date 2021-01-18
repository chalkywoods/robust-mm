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

  def train(self, epochs=100, data=None, batch_size=128, lr=1e-2, criterion=GCCA_loss, cca_dim=1, cca_hidden_dim=1000):
    if not self.checkpoint:
      optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=0.5)
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

      data = [modality.to(self.device) for modality in data]

      self.model.train()
      minibatch_indices = []
      for i in range(0, int(np.ceil(data[0].shape[0]/batch_size))):
        sidx = i*batch_size
        eidx = sidx + batch_size
        minibatch_indices.append( (sidx, eidx) )
      
      random.shuffle(minibatch_indices)

      for epoch in range(epochs):
        total_loss = 0
        with tqdm.tqdm(total=len(minibatch_indices)) as pbar_train:
          for batch, (start, stop) in enumerate(minibatch_indices):
            optimizer.zero_grad()
            out = self.model([modality[start:stop] for modality in data])
            loss = criterion(out)
            total_loss += loss
            total_loss_avg = total_loss / ((batch + 1))
            #if epoch % 1 == 0:
            #print('Epoch: {}, Loss:{}'.format(epoch, loss))
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar_train.update(1)
            pbar_train.set_description('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, total_loss_avg))
          random.shuffle(minibatch_indices)


    self.cca_dim = cca_dim
    self.wgcca = WeightedGCCA(self.modalities, [self.out_dim]*self.modalities, cca_dim, [1.e-12]*self.modalities, cca_hidden_dim, viewWts=None, verbose=True)

    self.model.eval()
    train_out = [out.detach().numpy() for out in self.model(data)]
    self.wgcca.learn(train_out)

  def get_corrs(self, data, embedding_dim=None):
    self.model.eval()
    data_reduced = [reduced.detach().numpy() for reduced in self.model(data)]
    return self.get_linear_corrs(data_reduced, embedding_dim)

  def get_linear_corrs(self, data, embedding_dim=None):
    embedding_dim = self.cca_dim if embedding_dim is None else embedding_dim
    corrs = np.zeros((self.modalities, self.modalities))
    length = data[0].shape[0]
    embeddings = []
    for i in range(self.modalities):
      corrs[i,i] = 1
      embeddings.append(self.wgcca.apply(data, K=self.get_K(i, length)))
    for i, j in itertools.combinations(range(self.modalities), 2):
      corrs[i, j] = sum_corr(embeddings[i][:,0:embedding_dim], embeddings[j][:,0:embedding_dim])
    return corrs

  def get_K(self, modality, length):
    K = np.zeros((length, self.modalities))
    K[:,modality] = 1
    return K

  def load_checkpoint(self, path):
    self.model.load_state_dict(torch.load(path))
    self.checkpoint = True

def sum_corr(x, y):
  dim = x.shape[1]
  corrmatrix = np.corrcoef(x, y, rowvar=False)
  corrs = 0
  for i in range(dim):
      corrs += corrmatrix[i,i+dim]
  return corrs/dim