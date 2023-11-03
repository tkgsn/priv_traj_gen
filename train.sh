dataset=chengdu
data_name=100
training_data_name=200_10_bin2_seed0
route_data_name=0_0_bin2_seed0

cuda_number=3
seed=0
patience=10
batch_size=0
noise_multiplier=1
clipping_bound=1
epsilon=1000
n_split=5
n_layers=1
hidden_dim=256
location_embedding_dim=64
memory_dim=100
memory_hidden_dim=64
learning_rate=1e-3
accountant_mode=rdp
dp_delta=1e-5
meta_network_load_path=None
physical_batch_size=20
n_epoch=1000
meta_n_iter=10
coef_location=1
coef_time=1
n_classes=10
global_clip=1
n_test_locations=30
meta_patience=1000
meta_dist=dirichlet
clustering=depth


# network_type=meta_network
network_type=fulllinear_quadtree
# network_type=markov1

activate=relu
# activate=leaky_relu

transition_type=first
# transition_type=marginal

# set the options
is_dp=False
remove_first_value=True
remove_duplicate=False
train_all_layers=True
consistent=True

declare -A arguments=(
    ["dataset"]=$dataset
    ["data_name"]=$data_name
    ["training_data_name"]=$training_data_name
    ["seed"]=$seed
    ["cuda_number"]=$cuda_number
    ["patience"]=$patience
    ["batch_size"]=$batch_size
    ["noise_multiplier"]=$noise_multiplier
    ["clipping_bound"]=$clipping_bound
    ["epsilon"]=$epsilon
    ["n_split"]=$n_split
    ["n_layers"]=$n_layers
    ["hidden_dim"]=$hidden_dim
    ["location_embedding_dim"]=$location_embedding_dim
    ["learning_rate"]=$learning_rate
    ["accountant_mode"]=$accountant_mode
    ["physical_batch_size"]=$physical_batch_size
    ["n_epoch"]=$n_epoch
    ["meta_n_iter"]=$meta_n_iter
    ["coef_location"]=$coef_location
    ["coef_time"]=$coef_time
    ["n_classes"]=$n_classes
    ["global_clip"]=$global_clip
    ["memory_dim"]=$memory_dim
    ["n_test_locations"]=$n_test_locations
    ["meta_patience"]=$meta_patience
    ["meta_network_load_path"]=$meta_network_load_path
    ["network_type"]=$network_type
    ["activate"]=$activate
    ["meta_dist"]=$meta_dist
    ["memory_hidden_dim"]=$memory_hidden_dim
    ["clustering"]=$clustering
    ["dp_delta"]=$dp_delta
    ["transition_type"]=$transition_type
    ["route_data_name"]=$route_data_name
)

declare -A options=(
    ["is_dp"]=$is_dp
    ["remove_first_value"]=$remove_first_value
    ["remove_duplicate"]=$remove_duplicate
    ["train_all_layers"]=$train_all_layers
    ["consistent"]=$consistent
)

# make the option parameter
option=""
for argument in "${!arguments[@]}"; do
    option="$option --$argument ${arguments[$argument]}"
done
for key in "${!options[@]}"; do
    if [ "${options[$key]}" = True ]; then
        option="$option --$key"
    fi
done


save_name=${network_type}_dp${is_dp}_meta${meta_n_iter}_dim${memory_dim}_${memory_hidden_dim}_${location_embedding_dim}_${hidden_dim}_btch${batch_size}_cl${clustering}_${epsilon}_tr${train_all_layers}_co${consistent}
python3 run.py --save_name $save_name $option