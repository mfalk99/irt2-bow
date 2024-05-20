DATASETS=(
    # "blp/fb15k237"
    # "blp/wikidata5m"
    # "blp/wn18rr"

    "irt2/tiny"
    # "irt2/small"
    # "irt2/medium"
    # "irt2/large"
)

VARIANTS=(
    # full
    original
)

TASKS=(
    heads
    tails
)

SPLITS=(
    # valid
    test
)

OUT_DIR="/mnt/data/dok/maurice/irt/irt2-bow/runs"

for dataset in ${!DATASETS[@]}; do
    for variant in ${!VARIANTS[@]}; do
        for split in ${!SPLITS[@]}; do
            for task in ${!TASKS[@]}; do

                irt2-linking \
                    --variant ${VARIANTS[$variant]} \
                    --task ${TASKS[$task]} \
                    --dataset-name ${DATASETS[$dataset]} \
                    --split ${SPLITS[$split]} \
                    --out "${OUT_DIR}/${DATASETS[$dataset]}-${SPLITS[$split]}-${VARIANTS[$variant]}-${TASKS[$task]}.yaml"

            done    
        done
    done
done