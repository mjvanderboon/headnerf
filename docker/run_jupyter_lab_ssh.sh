#!bin/bash
# Set up tensorboard. The base env has incompatible versions from the image? Creating new env to circumvent.
#conda create -n tensorboard python=3.8
#conda activate tensorboard
#pip install tensorboard
#tensorboard --logdir=/gitlab/projects/headnerf/uv_completion/lightning_logs --host 0.0.0.0 --port 6006 &

# Set up jupyterlab
cd /gitlab
/usr/sbin/sshd -D
#jupyter lab --allow-root --no-browser --port=8888 --ip=0.0.0.0
jupyter lab --no-browser --allow-root --port=8888 --ip=0.0.0.0 --config=/project/docker/jupyter_notebook_config.py

