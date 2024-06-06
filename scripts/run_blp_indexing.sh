DATASETS=(
    "blp/fb15k237"
    "blp/wn18rr"
)

SPLITS=(
    training
    training,validation
    training,validation,testing
)

for dataset in ${!DATASETS[@]}; do
    for splits in ${!SPLITS[@]}; do

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