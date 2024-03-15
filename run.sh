sudo docker run --rm --gpus all -v ~/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/baseline"
sudo docker run --rm --gpus all -v ~/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/baseline evaluation_mode=True"

sudo docker run --rm --gpus all -v ~/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/pre"
sudo docker run --rm --gpus all -v ~/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/pre evaluation_mode=True"

sudo docker run --rm --gpus all -v ~/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet"
sudo docker run --rm --gpus all -v ~/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet evaluation_mode=True"

sudo docker run --rm --gpus all -v ~/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet_pre"
sudo docker run --rm --gpus all -v ~/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet_pre evaluation_mode=True"

sudo docker run --rm --gpus all -v ~/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet_multitask"
sudo docker run --rm --gpus all -v ~/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet_multitask evaluation_mode=True"

sudo docker run --rm --gpus all -v ~/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet_multitask_pre"
sudo docker run --rm --gpus all -v ~/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet_multitask_pre evaluation_mode=True"