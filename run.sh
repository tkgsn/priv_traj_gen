sudo docker

python3 hub.py +experiment=ablation/hrnet_multitask_pre evaluation_mode=True

sudo docker run --rm --gpus all -v ~/data:/data hrnet /bin/bash -c "git pull && python3 hub.py +experiment=ablation/hrnet_multitask_pre"