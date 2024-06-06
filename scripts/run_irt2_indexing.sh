DATASETS=(
    irt2/tiny
    irt2/small
    irt2/medium
    irt2/large
)

SPLITS=(
    training
)

for dataset in ${!DATASETS[@]}; do
    for split in ${!SPLITS[@]}; do

        index-by-splits \
                --dataset-name ${DATASETS[$dataset]} \
                --splits ${SPLITS[$splits]} \
                --re-create

        # with subsampling
        index-by-splits \
                --dataset-name ${DATASETS[$dataset]} \
                --splits ${SPLITS[$splits]} \
                --with-subsampling \
                --re-create
    done
done