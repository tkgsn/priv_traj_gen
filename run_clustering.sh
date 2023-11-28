cd ./competitors/clustering


export DATASET=geolife_mm
export MAX_SIZE=0
export N_BINS=30

# epsilons=(0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)
epsilons=(0)

for epsilon in ${epsilons[@]}
do
    export EPSILON=$epsilon
    ./run.sh
done