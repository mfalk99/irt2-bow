DATASETS=(
    # "blp/fb15k237"
    # "blp/wikidata5m"
    # "blp/wn18rr"

    "irt2/tiny"
    "irt2/small"
    "irt2/medium"
    "irt2/large"
)

VARIANTS=(
    full
    original
)

for dataset in ${!DATASETS[@]}; do
    for variant in ${!VARIANTS[@]}; do

        irt2-indexing \
            --dataset-name ${DATASETS[$dataset]} \
            --variant ${VARIANTS[$variant]} \
            --re-create
                
    done
done