##
# $1: distillation, softmax, wce, cwce;
# $2: partitions;
# $3: gpus-id;
# $4: teacher depth;
# $5: student depth;
# $6: -e for evaluate, -r for resume;
# $7: specify-path: specify the path of model;
# e.g. ./run.sh distillation Test 1 32 32 -e
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 \
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=spl_distill \
python main_kd.py \
    --temperature 10 --teacher-path experiments/cifar100/resnet110_wd0.0001/model_best.pth.tar --teacher-depth 110 \
    -a resnet --dataset cifar100 --depth $3 --lr 0.1 --train-batch 64 --epochs 200 --schedule 80 130 161 --gamma 0.1 --wd 1e-4 \
    --loss-fun $2 \
    $4 \
