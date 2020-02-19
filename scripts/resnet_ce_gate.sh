# $1: partitions;
# $2: network architecture;
# $3: depth;
# $4: dataset, cifar10 or cifar100
# e.g. ./resnet_softmax.sh Test resnet 32 cifar100
#MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 \
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=spl_distill \
python main_ce_gate.py \
    -a $2 --depth $3 --dataset $4 --lr 0.01 --train-batch 64 --epochs 130 --schedule 60 100 --gamma 0.1 --wd 1e-4 \
    --ratio $5 --rscale $6 \
    -f --model-path $7 \
    $8 \
    $9 
    #--epochs $1
    #--epochs $1 \
