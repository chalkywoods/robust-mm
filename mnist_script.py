import argparse
import numpy as np
import torch
import pickle
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Subset
import tqdm
from importlib import reload
import matplotlib.pyplot as plt

import utils.mm_mnist
import utils.evaluate
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
parser.add_argument('--noise_type', default='gmm',
                    help='Type of noise, gmm or gaussian')
parser.add_argument('--thresh_method', default='intersection',
                    help='Anomaly detector threshold method to use')
parser.add_argument('--data', default='../data',
                    help='Number of individual false negatives to ignore when classifying modalities')
parser.add_argument('--repeat', type=int, default=0,
                    help='Repeat number')
parser.add_argument('--seed', type=int, default=-1,
                    help='RNG seed. -1 for random seed')
parser.add_argument('--train_snr', type=float, default=-1.0,
                    help='snr to use for threshold training. -1 for global snr')
parser.add_argument('--ad_classifier', default='none',
                    help='function to use for rounding when estimating number of corrupt modalities. round, floor or ceiling, or none for proportional classification')
args = parser.parse_args()
if args.train_snr == -1:
  train_snr = args.snr
elif args.train_snr < 100:
  train_snr = args.train_snr
else:
  train_snr = 0.65
print('Using training snr of {}'.format(train_snr))

try:
  thresh_method = float(args.thresh_method)
  print('Using hard thresholds of {}'.format(threshold))
except:
  thresh_method = args.thresh_method
  print('Generating thresholds using {}'.format(thresh_method))

corruption_classifier = 'est_corrupt'
round_func = np.round
if args.ad_classifier == 'round':
  round_func = np.round
elif args.ad_classifier == 'floor':
  round_func = np.floor
elif args.ad_classifier == 'ceiling':
  round_func = np.ceil
elif args.ad_classifier == 'delta':
  corruption_classifier = 'delta'
elif args.ad_classifier == 'prob':
  corruption_classifier = 'prob'
else:
  corruption_classifier = 'proportional'

print('Using {} corruption classifier'.format(corruption_classifier))
if corruption_classifier == 'est_corrupt':
  print('    with rounding function {}'.format(args.ad_classifier))

ds_train = utils.mm_mnist.MM_MNIST(args.data, train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

ds_test = utils.mm_mnist.MM_MNIST(args.data, train=False, download=True,
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

classifier = utils.mm_mnist.MM_Classifier(4, (1,14,14), dropout=0.2, moddrop=0.1, head_type='cnn')
criterion = nn.CrossEntropyLoss()
classifier.load_state_dict(torch.load('output/mm_mnist_cnn_moddrop.pth'))

results = utils.evaluate.pipeline(classifier, dl_train_cca, dl_train_ad, dl_test_ad, cca_dim=args.cca_dim, 
                                  snr=args.snr, gmm_components=args.gmm_components, window_size=args.window_size, 
                                  grace=args.grace, cca_path='output/mm_mnist_cca_cnn_moddrop.pth', noise_type=args.noise_type, 
                                  thresh_method=thresh_method, train_snr=train_snr, round_func=round_func, corruption_classifier=corruption_classifier)

with open('output/cnn_moddrop_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(args.batch_size, args.snr, args.gmm_components, args.cca_dim, args.window_size, args.grace, args.noise_type, args.thresh_method, args.repeat, train_snr, args.ad_classifier), 'wb') as f:
    pickle.dump(results, f)