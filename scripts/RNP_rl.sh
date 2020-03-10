# $1: partitions;
# $2: network architecture;
# $3: depth;
# $4: dataset, cifar10 or cifar100
# e.g. ./resnet_softmax.sh Test resnet 32 cifar100
#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 \
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=dy \
python main_RNP.py \
    -a $2 --depth $3 --dataset $4 --lr 0.1 --train-batch 64 --epochs 100 --schedule 25 50 75 --gamma 0.5 --wd 1e-4 \
    --greedyP $5 \
    $6
    #--epochs $1 \
