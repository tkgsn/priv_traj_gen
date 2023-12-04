cd ./competitors/privtrace

export DATASET=geolife
export MAX_SIZE=0
export N_BINS=30


# total_epsilons=(0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)
total_epsilons=(0)
fixed_divide_parameter=32

for total_epsilon in ${total_epsilons[@]}
do
    export TOTAL_EPSILON=$total_epsilon
    export FIXED_DIVIDE_PARAMETER=$fixed_divide_parameter
    ./run.sh
done