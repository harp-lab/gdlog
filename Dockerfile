FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y build-essential git cmake wget curl vim

COPY . /gdlog
RUN rm -rf /gdlog/build
RUN cd /gdlog && cmake -Bbuild . && cd build && make -j8

WORKDIR /gdlog/build/bin
