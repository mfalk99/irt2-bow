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

            bow-kgc \
                --task ${TASKS[$task]} \
                --dataset-name ${DATASETS[$dataset]} \
                --split ${SPLITS[$split]} \
                --out "${OUT_DIR}/${DATASETS[$dataset]}-${SPLITS[$split]}-${TASKS[$task]}.csv"

        done

        # create report
        eval-kgc \
            --dataset-name ${DATASETS[$dataset]} \
            --head-task "${OUT_DIR}/${DATASETS[$dataset]}-${SPLITS[$split]}-heads.csv" \
            --tail-task "${OUT_DIR}/${DATASETS[$dataset]}-${SPLITS[$split]}-tails.csv" \
            --split ${SPLITS[$split]} \
            --model bm25 \
            --out reports/${DATASETS[$dataset]}-${SPLITS[$split]}.yaml

    done
done