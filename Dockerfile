FROM continuumio/miniconda3

ADD env.yml /tmp/env.yml
RUN conda env create -f /tmp/env.yml

RUN echo "conda activate $(head -1 /tmp/env.yml | cut -d' ' -f2)" >> ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 /tmp/env.yml | cut -d' ' -f2)/bin:$PATH

ENV CONDA_DEFAULT_ENV $(head -1 /tmp/env.yml | cut -d' ' -f2)