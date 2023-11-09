# register enviornment variables
source ./variables/configs.sh

container_name=prepare-geolife-test
environment_variables=(
    "DATASET=geolife_test"
    "SEED=0"
    "MAX_SIZE=0"
    "N_BINS=30"
    "TIME_THRESHOLD=10"
    "LOCATION_THRESHOLD=200"
)
cpu=10
memory=32
CMD="/bin/bash -c 'cd priv_traj_gen && ./prepare.sh && ./preprocess.sh'"