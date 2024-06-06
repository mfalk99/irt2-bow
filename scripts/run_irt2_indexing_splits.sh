DATASETS=(
    irt2/tiny
    irt2/small
    irt2/medium
    irt2/large
)

SPLITS=(
    "validation"
    "test"
)

for dataset in ${!DATASETS[@]}; do
    for split in ${!SPLITS[@]}; do

        irt2-indexing \
            --dataset-name ${DATASETS[$dataset]} \
            --split ${SPLITS[$split]} \
            --re-create
                
    done
done