##
# $1: partitions;
# $2: gpus-id;
# $3: depth;
# $4: -e for evaluate, -r for resume;
# e.g. ./resnet_softmax.sh Test 1 32 -e
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 \
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=spl_distill \
python main_ce.py \
    -a $2 --dataset cifar100 --depth $3 --lr 0.001 --train-batch 64 --epochs 30 --schedule 15 25 --gamma 0.1 --wd 1e-4 \
    --model-path $4 \
    $5
