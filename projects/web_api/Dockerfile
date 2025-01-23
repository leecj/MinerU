# Use the official Ubuntu base image
FROM docker.m.daocloud.io/ubuntu:22.04

# ENV http_proxy http://127.0.0.1:7890
# ENV https_proxy http://127.0.0.1:7890

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

# ADD sources.list /etc/apt
# RUN apt-get clean

# Update the package list and install necessary packages
RUN apt-get update && \
    apt-get install -y \
        software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.10 \
        python3.10-venv \
        python3.10-distutils \
        python3-pip \
        wget \
        git \
        libgl1 \
        libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create a virtual environment for MinerU
RUN python3 -m venv /opt/mineru_venv
RUN pip3 config set global.index-url https://mirrors.aliyun.com/pypi/simple
# Activate the virtual environment and install necessary Python packages
RUN /bin/bash -c "source /opt/mineru_venv/bin/activate && \
    pip3 install --upgrade pip -i https://mirrors.aliyun.com/pypi/simple && \
    wget https://gitee.com/myhloli/MinerU/raw/master/docker/china/requirements.txt -O requirements.txt && \
    pip3 install -r requirements.txt --extra-index-url https://wheels.myhloli.com -i https://mirrors.aliyun.com/pypi/simple && \
    pip3 install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/"

# Copy the configuration file template and install magic-pdf latest
RUN /bin/bash -c "wget https://gitee.com/myhloli/MinerU/raw/master/magic-pdf.template.json && \
    cp magic-pdf.template.json /root/magic-pdf.json && \
    source /opt/mineru_venv/bin/activate && \
    pip3 install -U magic-pdf -i https://mirrors.aliyun.com/pypi/simple"

# Download models and update the configuration file
RUN /bin/bash -c "pip3 install modelscope  --no-cache-dir && \
    wget https://gitee.com/myhloli/MinerU/raw/master/scripts/download_models.py && \
    python3 download_models.py"

# Create a virtual environment for MinerU
#RUN python3 -m venv /opt/mineru_venv
#RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
## Activate the virtual environment and install necessary Python packages
#RUN /bin/bash -c "source /opt/mineru_venv/bin/activate && \
#    pip install --upgrade pip && \
#    pip install magic-pdf[full] --extra-index-url https://myhloli.github.io/wheels/ --no-cache-dir"


RUN /bin/bash -c "source /opt/mineru_venv/bin/activate && \
    pip3 install fastapi uvicorn python-multipart --no-cache-dir"

# Copy the configuration file template and set up the model directory
#COPY magic-pdf.template.json /root/magic-pdf.json
#ADD models /opt/models
#ADD .paddleocr /root/.paddleocr
#ADD app/code /root/code
#ADD app/server.py /root/server.py
WORKDIR /root

ADD ./app ./app


# Set the models directory in the configuration file (adjust the path as needed)
# RUN sed -i 's|/tmp/models|/opt/models|g' /root/magic-pdf.json

# Create the models directory
# RUN mkdir -p /opt/models

# Set the entry point to activate the virtual environment and run the command line tool
# ENTRYPOINT ["/bin/bash", "-c", "source /opt/mineru_venv/bin/activate && exec \"$@\" && python3 server.py", "--"]


# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run FastAPI using Uvicorn, pointing to server.py and binding to 0.0.0.0:8000
CMD ["/bin/bash", "-c", "source /opt/mineru_venv/bin/activate && uvicorn app.server:app --host 0.0.0.0 --port 8000"]