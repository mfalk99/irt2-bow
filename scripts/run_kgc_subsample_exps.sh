DATASETS=(
    "blp/fb15k237"
    "blp/wn18rr"

    "irt2/tiny"
    "irt2/small"
    "irt2/medium"
    "irt2/large"
)

TASKS=(
    heads
    tails
)

SPLITS=(
    validation
    test
)

OUT_DIR="./";

for dataset in ${!DATASETS[@]}; do
    for split in ${!SPLITS[@]}; do
        
        # run linking
        for task in ${!TASKS[@]}; do

            irt2-linking \
                --task ${TASKS[$task]} \
                --dataset-name ${DATASETS[$dataset]} \
                --split ${SPLITS[$split]} \
                --with-subsampling \
                --out "${OUT_DIR}/${DATASETS[$dataset]}-${SPLITS[$split]}-${TASKS[$task]}.subsample.csv"

        done

        # create report
        eval-kgc \
            --dataset-name ${DATASETS[$dataset]} \
            --head-task "${OUT_DIR}/${DATASETS[$dataset]}-${SPLITS[$split]}-heads.subsample.csv" \
            --tail-task "${OUT_DIR}/${DATASETS[$dataset]}-${SPLITS[$split]}-tails.subsample.csv" \
            --split ${SPLITS[$split]} \
            --model bm25 \
            --with-subsampling \
            --out reports/${DATASETS[$dataset]}-${SPLITS[$split]}.subsample.yaml

    done
done