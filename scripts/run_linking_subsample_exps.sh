DATASETS=(
    "blp/fb15k237"
    # "blp/wikidata5m"
    "blp/wn18rr"

    "irt2/tiny"
    "irt2/small"
    "irt2/medium"
    "irt2/large"
)

VARIANTS=(
    full
    original
)

TASKS=(
    heads
    tails
)

SPLITS=(
    validation
    test
)

OUT_DIR="/mnt/data/dok/maurice/irt/irt2-bow/runs"

for dataset in ${!DATASETS[@]}; do
    for variant in ${!VARIANTS[@]}; do
        for split in ${!SPLITS[@]}; do
            
            # run linking
            for task in ${!TASKS[@]}; do

                irt2-linking \
                    --variant ${VARIANTS[$variant]} \
                    --task ${TASKS[$task]} \
                    --dataset-name ${DATASETS[$dataset]} \
                    --split ${SPLITS[$split]} \
                    --with-subsampling \
                    --out "${OUT_DIR}/${DATASETS[$dataset]}-${SPLITS[$split]}-${VARIANTS[$variant]}-${TASKS[$task]}-subsample.yaml"

            done

            # create report
            eval-kgc \
                --dataset-name ${DATASETS[$dataset]} \
                --head-task "${OUT_DIR}/${DATASETS[$dataset]}-${SPLITS[$split]}-${VARIANTS[$variant]}-heads-subsample.yaml" \
                --tail-task "${OUT_DIR}/${DATASETS[$dataset]}-${SPLITS[$split]}-${VARIANTS[$variant]}-tails-subsample.yaml" \
                --variant ${VARIANTS[$variant]} \
                --split ${SPLITS[$split]} \
                --with-subsampling \
                --model bm25 \
                --out reports/${DATASETS[$dataset]}-${SPLITS[$split]}-${VARIANTS[$variant]}-subsample.yaml

        done
    done
done