FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York
RUN apt-get update && apt-get install -y build-essential git wget curl vim software-properties-common lsb-release

RUN apt-get install -y ca-certificates gpg wget
RUN test -f /usr/share/doc/kitware-archive-keyring/copyright || wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null
RUN apt-get update && apt-get install -y kitware-archive-keyring
RUN apt-get update && apt-get install -y cmake

# use gcc-13 and g++-13 on ubuntu 22.04
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG USER=gdlog
ARG PASS="gdlog"
RUN useradd -m -s /bin/bash $USER && echo "$USER:$PASS" | chpasswd
USER gdlog


COPY --chown=gdlog:gdlog . /opt/gdlog
WORKDIR /opt/gdlog

RUN cmake -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo -Bbuild . && cd build && make -j
RUN chmod -R 757 /opt/gdlog


# ENTRYPOINT [ "/usr/bin/bash" ]
