CONFIG1=$1
CONFIG2=$2

OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file $CONFIG1 \
    --num-gpus 8 --resume

OMP_NUM_THREADS=1 python tools/train_net.py \
    --config-file $CONFIG2 \
    --num-gpus 8 --resume