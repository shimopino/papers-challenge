# https://github.com/NVIDIA/apex
ARG BASE_IMAGE
FROM $BASE_IMAGE

# install git
RUN apt-get update && apt-get install -y --no-install-recommends \
  git \
  curl \
  wget \
  zip \
  unzip \
  && apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/*

# make sure we don't overwrite some existing directory called "apex"
WORKDIR /tmp/unique_for_apex

# uninstall Apex if present, twice to make absolutely sure :)
RUN pip uninstall -y apex || :
RUN pip uninstall -y apex || :

# SHA is something the user can touch to force recreation of this Docker layer,
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR /tmp/unique_for_apex/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

# share UID from local host --> set the same authentication
ARG UID=1000
ARG USERNAME=devpapers
RUN useradd ${USERNAME} -u ${UID} -G sudo -s /bin/bash -m  && \
  echo ${USERNAME}' ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
  chown ${USERNAME}:${USERNAME} /home/${USERNAME}

USER ${USERNAME}
WORKDIR /home/${USERNAME}
ENV HOME /home/${USERNAME}

ENV PATH $PATH:/home/${USERNAME}/.local/bin/

RUN pip install --upgrade pip \
    && pip install --no-cache-dir mlflow torch-mimicry
