FROM python:3.8-bullseye
  
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client \
    jq \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/tkgsn/priv_traj_gen.git

WORKDIR /priv_traj_gen

COPY requirements.txt /priv_traj_gen
RUN pip3 install --no-cache-dir -r requirements.txt

# 
COPY id_rsa /root/.ssh/id_rsa
COPY id_rsa.pub /root/.ssh/id_rsa.pub
COPY config /root/.ssh/config

ENV DATASET chengdu
ENV MAXSIZE 10000
ENV SEED 0
ENV META_N_ITER 100
ENV EPOCH 1000
ENV N_BINS 30
ENV L_THRESH 200
ENV T_THRESH 10
ENV P_BATCH 100
ENV N_TEST 30
ENV DP True
ENV MULTI_TASK True

RUN chmod 600 /root/.ssh/id_rsa

CMD ["/bin/bash", "train.sh"]