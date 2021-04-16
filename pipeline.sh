#!/bin/bash
#$ -l h_rt=01:00:00
#$ -l rmem=4G
# Email notifications to me
#$ -M sharc@hggwoods.com
# Email notifications if the job aborts
#$ -m a
#$ -N multi-autoencoder
#$ -wd /home/aca18hgw/robust-mm

module load apps/python/conda
module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176

source activate multimodal
echo python mnist_script.py --batch_size $1 --snr $2 --gmm_components $3 --cca_dim $4 --window_size $5 --grace $6 --thresh_method $7 --data '/mnt/fastdata/aca18hgw/mnist' --noise_type $8 --repeat $9 --seed ${10} --train_snr ${11} --ad_classifier ${12}
python mnist_script.py --batch_size $1 --snr $2 --gmm_components $3 --cca_dim $4 --window_size $5 --grace $6 --thresh_method $7 --data '/mnt/fastdata/aca18hgw/mnist' --noise_type $8 --repeat $9 --seed ${10} --train_snr ${11} --ad_classifier ${12}
