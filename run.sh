#!/usr/bin/bash

dims=(16 32 64 128)

for dim in ${dims[@]}; do
    sudo docker run --rm --gpus all -v /home/takagi/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/baseline location_embedding_dim=$dim memory_hidden_dim=$dim"
    sudo docker run --rm --gpus all -v /home/takagi/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/baseline evaluation_mode=True location_embedding_dim=$dim memory_hidden_dim=$dim"

    sudo docker run --rm --gpus all -v /home/takagi/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/pre location_embedding_dim=$dim memory_hidden_dim=$dim"
    sudo docker run --rm --gpus all -v /home/takagi/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/pre evaluation_mode=True location_embedding_dim=$dim memory_hidden_dim=$dim"

    sudo docker run --rm --gpus all -v /home/takagi/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet location_embedding_dim=$dim memory_hidden_dim=$dim"
    sudo docker run --rm --gpus all -v /home/takagi/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet evaluation_mode=True location_embedding_dim=$dim memory_hidden_dim=$dim"

    sudo docker run --rm --gpus all -v /home/takagi/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet_pre location_embedding_dim=$dim memory_hidden_dim=$dim"
    sudo docker run --rm --gpus all -v /home/takagi/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet_pre evaluation_mode=True location_embedding_dim=$dim memory_hidden_dim=$dim"

    sudo docker run --rm --gpus all -v /home/takagi/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet_multitask location_embedding_dim=$dim memory_hidden_dim=$dim"
    sudo docker run --rm --gpus all -v /home/takagi/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet_multitask evaluation_mode=True location_embedding_dim=$dim memory_hidden_dim=$dim"
    sudo docker run --rm --gpus all -v /home/takagi/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet_multitask_cons evaluation_mode=True location_embedding_dim=$dim memory_hidden_dim=$dim"

    sudo docker run --rm --gpus all -v /home/takagi/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet_multitask_pre location_embedding_dim=$dim memory_hidden_dim=$dim"
    sudo docker run --rm --gpus all -v /home/takagi/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet_multitask_pre evaluation_mode=True location_embedding_dim=$dim memory_hidden_dim=$dim"
    sudo docker run --rm --gpus all -v /home/takagi/data:/data hrnet /bin/bash -c "rm requirements.txt && git pull && python3 hub.py +experiment=ablation/hrnet_multitask_pre_cons evaluation_mode=True location_embedding_dim=$dim memory_hidden_dim=$dim"
done