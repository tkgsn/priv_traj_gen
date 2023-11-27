cd ./competitors/privtrace

export DATASET=geolife_mm
export MAX_SIZE=0

total_epsilons=(0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)

for total_epsilon in ${total_epsilons[@]}
do
    export TOTAL_EPSILON=$total_epsilon
    ./run.sh
done