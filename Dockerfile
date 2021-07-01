FROM continuumio/miniconda3:latest

RUN apt update && apt -y upgrade
RUN apt install -y build-essential wget doxygen
RUN conda install -y mkl mkl-include pybind11 numpy sphinx pytest scikit-learn
RUN conda install -y -c conda-forge faiss-cpu

RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.0-rc2/cmake-3.21.0-rc2-linux-x86_64.sh
RUN bash cmake-3.21.0-rc2-linux-x86_64.sh --skip-license --prefix=/usr/

ENV MKL_ROOT=/opt/conda

