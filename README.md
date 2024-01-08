# priv_traj_gen

## Environment of the experiments

Azure Virtual machine

Size: Standard NC4as T4 v3 (4 vcpus, 28 GiB memory) (see https://learn.microsoft.com/ja-jp/azure/virtual-machines/nct4-v3-series for the details)

image: NVIDIA GPU-Optimized VMI -v22.06.0 -x64 Gen 1

## build

```
git clone https://github.com/tkgsn/priv_traj_gen
cd priv_traj_gen
docker build -t kyotohiemrnet.azurecr.io/hiemrnet_cu117 .
```
We used File shares of Azure for the data directory and mount on /mnt/data with the script made by Azure or you can just use your local directory if you don't use virtual machine
```
sudo mkdir -p /mnt/data
```

## experiments


### prepare dataset
```
docker run -v /mnt/data:/data -e DATASET=geolife kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./prepare.sh"
docker run -v /mnt/data:/data -e DATASET=peopleflow kyotohiemrnet.azurecr.io/hiemrnet_cu117_ /bin/bash -c "./prepare.sh"
docker run -v /mnt/data:/data -e DATASET=chengdu kyotohiemrnet.azurecr.io/hiemrnet_cu117_ /bin/bash -c "./prepare.sh"
```

### preprocess dataset
```
docker run -v /mnt/data:/data -e DATASET=geolife -e SEED=0 -e MAX_SIZE=0 -e N_BINS=30 -e T_THRESH=30 -e L_THRESH=200 -e TRUNCATE=0 kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./preprocess.sh"
docker run -v /mnt/data:/data -e DATASET=peopleflow -e SEED=0 -e MAX_SIZE=0 -e N_BINS=30 -e T_THRESH=30/60 -e L_THRESH=200 -e TRUNCATE=0 kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./preprocess.sh"
docker run -v /mnt/data:/data -e DATASET=chengdu -e SEED=0 -e MAX_SIZE=0 -e N_BINS=30 -e T_THRESH=30 -e L_THRESH=200 -e TRUNCATE=0 kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./preprocess.sh"
docker run -v /mnt/data:/data -e DATASET=random -e SEED=0 -e MAX_SIZE=0 -e N_BINS=30 -e T_THRESH=30 -e L_THRESH=200 -e TRUNCATE=0 kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./preprocess.sh"
docker run -v /mnt/data:/data -e DATASET=rotation -e SEED=0 -e MAX_SIZE=0 -e N_BINS=30 -e T_THRESH=30 -e L_THRESH=200 -e TRUNCATE=0 kyotohiemrnet.azurecr.io/hiemrnet_cu117 /bin/bash -c "./preprocess.sh"
```

```
cd exps
```
### Section 5.2
```
python3 comparison_baseline.py
python3 comparison_clustering.py
python3 comparison_hiemrnet.py
python3 comparison_privtrace.py
```
### Section 5.3
```
python3 road_comparison_hiemrnet.py
python3 road_comparison_mtnet.py
```
### Section 5.4.1
```
python3 ablation_geolife.py
```
### Section 5.4.2
```
python3 ablation_deconv.py
```
### Section 5.4.3
```
python3 ablation_hie.py
```
### Section 5.4.4
```
python3 ablation_peopleflow.py
```
### Section 5.5
```
python3 priv_budget_allocation.py
```
