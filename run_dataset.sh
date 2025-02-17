conda activate anymatch
DATASETS=("wdc-computers" "wdc-shoes" "wdc-watches" "wdc-cameras" "dbgo" "music")

# CUDA_VISIBLE_DEVICES=3 python loo.py --leaved_dataset_name dbgo --base_model t5-base
for DATASET in "${DATASETS[@]}"
do
    export TORCH_USE_CUDA_DSA=1
    CUDA_VISIBLE_DEVICES=3 python -W ignore loo.py \
        --seed 42 \
        --tbs 16 \
        --vbs 32 \
        --leaved_dataset_name $DATASET \
        --serialization_mode mode1 \
        --train_data attr+row \
        --base_model gpt2 \
        --patience_start 20
done