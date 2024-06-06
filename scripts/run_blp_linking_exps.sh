DATASETS=(
    # "blp/fb15k237"
    "blp/wn18rr"
    # "blp/wikidata5m"
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
    for split in ${!SPLITS[@]}; do
        
        # --- run FULL linking exps ---

        for task in ${!TASKS[@]}; do
            blp-linking \
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

        # --- run SUBSAMPLING linking exps ---

        # for task in ${!TASKS[@]}; do
        #     blp-linking \
        #         --task ${TASKS[$task]} \
        #         --dataset-name ${DATASETS[$dataset]} \
        #         --split ${SPLITS[$split]} \
        #         --with-subsampling \
        #         --out "${OUT_DIR}/${DATASETS[$dataset]}-${SPLITS[$split]}-${TASKS[$task]}.subsample.csv"
        # done

        # # create report
        # eval-kgc \
        #     --dataset-name ${DATASETS[$dataset]} \
        #     --head-task "${OUT_DIR}/${DATASETS[$dataset]}-${SPLITS[$split]}-heads.subsample.csv" \
        #     --tail-task "${OUT_DIR}/${DATASETS[$dataset]}-${SPLITS[$split]}-tails.subsample.csv" \
        #     --split ${SPLITS[$split]} \
        #     --with-subsampling \
        #     --model bm25 \
        #     --out reports/${DATASETS[$dataset]}-${SPLITS[$split]}.yaml

    done
done