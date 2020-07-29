CONFIG=$1
MODEL=$2

OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file $CONFIG \
    --eval-only \
    --num-gpus 8 \
    MODEL.WEIGHTS $MODEL