# $1: partitions;
# $2: network architecture;
# $3: depth;
# $4: dataset, cifar10 or cifar100
# e.g. ./resnet_softmax.sh Test resnet 32 cifar100
#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 \
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=dynamic \
python vgg_RNP.py \
