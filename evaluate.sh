model_dir=$MODEL_DIR
location_threshold=$LOCATION_THRESHOLD
time_threshold=$TIME_THRESHOLD
n_bins=$N_BINS
seed=$SEED

echo "hello"
python3 evaluation.py --model_dir $model_dir --location_threshold $location_threshold --time_threshold $time_threshold --n_bins $n_bins --seed $seed --server