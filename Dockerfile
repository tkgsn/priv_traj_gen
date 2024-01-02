FROM python:3.8-bullseye
  
RUN apt-get update && apt-get install -y --no-install-recommends \
    jq \
    openssh-client \
 && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/tkgsn/priv_traj_gen.git

WORKDIR /priv_traj_gen

COPY requirements.txt /priv_traj_gen
RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]