##
# $1: distillation, softmax, wce, cwce;
# $2: partitions;
# $3: gpus-id;
# $4: teacher depth;
# $5: student depth;
# $6: -e for evaluate, -r for resume;
# $7: specify-path: specify the path of model;
# e.g. ./run_kd.sh AD kd 10 32 8 cifar100 0.1
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 \
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 --job-name=spl_distill \
python main_kd.py \
    --temperature $3 --teacher-path experiments/$6/baseline/resnet$4_wd0.0001/model_best.pth.tar --teacher-depth $4 \
    --loss-fun $2 \
    -a resnet --depth $5 --dataset $6 --lr $7 --train-batch 64 --epochs 200 --schedule 80 130 161 --gamma 0.1 --wd 1e-4 \
    --pce-threshold $8 \
    $9 \
