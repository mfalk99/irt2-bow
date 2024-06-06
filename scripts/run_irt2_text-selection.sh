DATASETS=(
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

OUT_DIR="./"

for dataset in ${!DATASETS[@]}; do
    for split in ${!SPLITS[@]}; do        
        for task in ${!TASKS[@]}; do

            irt2-text-selection \
                --task ${TASKS[$task]} \
                --dataset-name ${DATASETS[$dataset]} \
                --split ${SPLITS[$split]} \
                --out "${OUT_DIR}/${DATASETS[$dataset]}-${SPLITS[$split]}-${TASKS[$task]}-texts.pkl"

        done
    done
done