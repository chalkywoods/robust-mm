import argparse
import numpy as np
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Subset
import tqdm
from importlib import reload
import matplotlib.pyplot as plt

import utils.mm_mnist
from utils.noise import GMMNoise
import dgcca.dgcca
import dgcca.anomaly_detection

parser = argparse.ArgumentParser(description='Evaluate cca anomaly detection')
parser.add_argument('--batch_size', type=int, default=128,
                    help='batch size')
parser.add_argument('--snr', type=float, default=1,
                    help='Signal to noise ratio for data corruption')
parser.add_argument('--gmm_components', type=int, default=5,
                    help='Number of components to use in the gaussian mixture models used for generating per-pixel noise')
parser.add_argument('--cca_dim', type=int, default=10,
                    help='Size of representation learned by DGCCA')
parser.add_argument('--window_size', type=int, default=20,
                    help='Number of samples to consider when detecting corruption in modalities')
parser.add_argument('--grace', type=int, default=0,
                    help='Number of individual false negatives to ignore when classifying modalities')
args = parser.parse_args()

ds_train = utils.mm_mnist.MM_MNIST('/mnt/fastdata/aca18hgw/mnist', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

ds_test = utils.mm_mnist.MM_MNIST('/mnt/fastdata/aca18hgw/mnist', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

split = [50000, 10000]

ds_train_cca = Subset(ds_train, range(0, split[0]))
ds_train_ad = Subset(ds_train, range(split[0], split[0] + split[1]))

dl_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch_size, shuffle=True)
dl_train_cca = torch.utils.data.DataLoader(ds_train_cca, batch_size=args.batch_size, shuffle=True)
dl_train_ad = torch.utils.data.DataLoader(ds_train_ad, batch_size=len(ds_train_ad), shuffle=False) # No shuffle as corrupt and clean data must be loaded in the same order
dl_test_classifier = torch.utils.data.DataLoader(ds_test, batch_size=args.batch_size)
dl_test_ad = torch.utils.data.DataLoader(ds_test, batch_size=len(ds_test))

classifier = utils.mm_mnist.MM_Classifier(4, (1,14,14), dropout=0.2, moddrop=0, activation=torch.tanh)
criterion = nn.CrossEntropyLoss()
classifier.load_state_dict(torch.load('output/mm_mnist.pth'))

def embed(loader=dl_train_cca, noise=None):
    classifier.eval()

    with tqdm.tqdm(total=len(loader)) as pbar:
        pbar.set_description('Embedding')
        out = []
        for batch, (data, _) in enumerate(loader):
            data = data if noise is None else torch.Tensor(noise(data))
            out.append([mod.detach() for mod in classifier(data, heads=True)])
            pbar.update(1)
    output = []
    for i in range(4):
        output.append(np.concatenate([batch[i] for batch in out]))
    return output

def eval(data, detector, noise, num_corrupt=0, window=20, grace=0):
    classifier.eval()
    tp_pred, tn_pred, fp_pred, fn_pred, tp_ind, tn_ind, fp_ind, fn_ind = (0,0,0,0,0,0,0,0)
    with tqdm.tqdm(total=len(range(0, len(data[0])-window, window))) as pbar:
        for i in range(0, len(data[0])-window, window):
            clean = np.ones(len(data))
            corrupt = np.random.default_rng().choice(len(data), size=num_corrupt, replace=False)
            clean[corrupt] = 0
            corrupt_data = noise([mod.numpy() for mod in data], corrupt)
            corrupt_data = [torch.FloatTensor(mod) for mod in corrupt_data]

            embedding = classifier(corrupt_data, heads=True)

            pred, individual = detector.detect_anomalies([mod[i:i+window,:].double() for mod in embedding], grace=grace, evaluating=True)

            for j in range(0,4):
                for k in range(j+1,4):
                    if j in corrupt or k in corrupt:
                        if not individual[j,k]:
                            tn_ind += 1
                        else:
                            fp_ind += 1
                    else:
                        if individual[j,k]:
                            tp_ind += 1
                        else:
                            fn_ind += 1
                        
            tp_pred += ((clean == pred) & (pred == True)).sum()
            tn_pred += ((clean == pred) & (pred == False)).sum()
            fp_pred += ((clean != pred) & (pred == True)).sum()
            fn_pred += ((clean != pred) & (pred == False)).sum()

            pbar.update(1)
            pbar.set_description('Prediction accuracy: {:.2%} | Individual accuracy: {:.2%} '.format((tp_pred+tn_pred)/(4*(1+(i/window))), (tp_ind+tn_ind)/(6*(1+(i/window)))))
    return np.array([[tp_pred, tn_pred, fp_pred, fn_pred],
                     [tp_ind, tn_ind, fp_ind, fn_ind]])

def pipeline(classifier, cca_dim, snr, gmm_components, window_size, grace, noise_gen=None):
    classifier.eval()
    cca = dgcca.dgcca.DGCCA([[125, 64]]*4, 32, device='cpu', use_all_singular_values=False)
    cca.load_checkpoint('output/mm_mnist_cca.pth')
    train_cca_embeddings = [torch.DoubleTensor(mod) for mod in embed(dl_train_cca)]
    cca.train(train_cca_embeddings, epochs=50, batch_size=128, cca_dim=cca_dim, cca_hidden_dim=1000)

    clean = [torch.DoubleTensor(mod) for mod in embed(dl_train_ad)]

    if noise_gen is None:
        noise_gen = utils.noise.GMMNoise(dl_train_ad, gmm_components) # Train GMM for each pixel to generate noise

    noise = lambda data: noise_gen.add_noise([mod.numpy() for mod in data], snr=snr) # noise and image weighted equally
    corrupt = [torch.DoubleTensor(mod) for mod in embed(dl_train_ad, noise)]

    detector = dgcca.anomaly_detection.CcaAnomalyDetector(cca)
    detector.train(clean, corrupt, stride='auto', window=window_size)

    test_noise = lambda data, modality: noise_gen.add_noise(data, snr=snr, modality=modality)

    res = []
    for data, label in dl_test_ad:
        data = [mod.double() for mod in data]
        for num_corrupt in [0,1,2]:
            res.append(eval(data, detector, test_noise, num_corrupt, window_size, grace))
    
    return res

results = pipeline(classifier, cca_dim=args.cca_dim, snr=args.snr, gmm_components=args.gmm_components, window_size=args.window_size, grace=args.grace)

print(np.stack(results))

with open('output/{}_{}_{}_{}_{}_{}.npy'.format(args.batch_size, args.snr, args.gmm_components, args.cca_dim, args.window_size, args.grace), 'wb') as f:
    np.save(f, np.stack(results))