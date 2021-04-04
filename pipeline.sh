#!/bin/bash
#$ -l h_rt=01:00:00
# Email notifications to me
#$ -M sharc@hggwoods.com
# Email notifications if the job aborts
#$ -m a
#$ -N multi-autoencoder
#$ -wd /home/aca18hgw/robust-mm

module load apps/python/conda
module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176

source activate multimodal

python mnist_script.py --batch_size $1 --snr $2 --gmm_components $3 --cca_dim $4 --window_size $5 --grace $6
