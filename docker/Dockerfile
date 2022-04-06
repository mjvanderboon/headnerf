FROM nvcr.io/nvidia/pytorch:21.03-py3

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Zet multiverse aan bij weghalen #
RUN cat /etc/apt/sources.list | sed -e '/multiverse/ s/\#//' > /etc/apt/sources2.list
RUN mv /etc/apt/sources2.list /etc/apt/sources.list

# Install all apt-get packages. Merendeel nodig voor anaconda
RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 \
    git mercurial subversion libbz2-dev libz-dev libpng-dev

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# Install ssh server, used by PyCharm for remote execution
RUN apt-get update && \
    apt-get -y install openssh-server && \
    mkdir -p /run/sshd && \
    sed -i -e 's/^UsePAM/#UsePam/' /etc/ssh/sshd_config
EXPOSE 22
# Depending on the user-namespace you are running, you need to add an internal user to docker image
ARG USERID
ARG GROUPID
ARG USERNAME
ARG GROUPNAME
RUN grep -q '^'$USERNAME':' /etc/passwd ||\
    echo "$USERNAME:x:$USERID:$GROUPID::/root" >> /etc/passwd; \
    grep -q '^'$GROUPNAME':' /etc/group ||\
    echo "$GROUPNAME:x:$GROUPID:" >> /etc/group

WORKDIR /

# No need for different envs, so everything is installed in the base conda env
RUN conda install -c conda-forge/label/cf202003 nodejs

RUN conda install -c conda-forge jupyterlab
RUN conda install -c conda-forge jupytext

RUN jupyter labextension install jupyterlab-jupytext
#RUN jupyter labextension install @jupyterlab/git==0.22.1 # gives error about version mismatch with backend.
RUN pip install --upgrade jupyterlab-git==0.22.1
RUN jupyter serverextension enable --py jupyterlab_git

RUN	conda update --force conda
# Necessary to force CUDA compilation of pytorch3d
ENV FORCE_CUDA=1

# Install facescape dependencies
#Env file wasn't working b/c kornia conda install forces to reinstall pytorch==1.4.0, which doesn't work with 3090 GPU.
RUN conda install tensorboard pyyaml python-lmdb scipy tqdm imageio imageio-ffmpeg opencv -c conda-forge
RUN pip install kornia

# Install cv2 dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install freeglut3-dev -y

# Install DECA dependencies
RUN pip install chumpy face-alignment yacs ninja fvcore