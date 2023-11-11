dataset=${DATASET}_mm

data_dir=$(jq -r '.data_dir' config.json)
graph_data_dir=${data_dir}/${dataset}/raw
FILE=fmm_config_omp.xml
sed -i "s|data_dir|$graph_data_dir|g" $FILE

fmm fmm_config_omp.xml