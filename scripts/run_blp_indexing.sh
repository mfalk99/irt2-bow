# DATASETS=(
#     "blp/fb15k237"
#     "blp/wn18rr"
#     # "blp/wikidata5m"
# )

# SPLITS=(
#     validation
#     test
# )

# for dataset in ${!DATASETS[@]}; do
#     for split in ${!SPLITS[@]}; do

#         irt2-indexing \
#             --dataset-name ${DATASETS[$dataset]} \
#             --split ${SPLITS[$split]} \
#             --re-create
                
#     done
# done

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

        # index-by-splits \
        #     --dataset-name ${DATASETS[$dataset]} \
        #     --splits ${SPLITS[$splits]} \
        #     --re-create
        
        index-by-splits \
            --dataset-name ${DATASETS[$dataset]} \
            --splits ${SPLITS[$splits]} \
            --with-subsampling \
            --re-create
                
    done
done