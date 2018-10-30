FROM tensorflow/tensorflow:1.10.0-py3

RUN             apt-get update \
             && apt-get install -y --no-install-recommends \
                    git \
                    wget \
             && pip3 install matplotlib \
             && pip3 install scikit-image flask python-dotenv tablib pytest

WORKDIR /protoc
RUN             apt-get install wget -y \
             && wget https://github.com/google/protobuf/releases/download/v3.3.0/protoc-3.3.0-linux-x86_64.zip \
             && unzip protoc-3.3.0-linux-x86_64.zip \
             && rm protoc-3.3.0-linux-x86_64.zip

WORKDIR /tensorflow
RUN             git clone --depth 1 https://github.com/tensorflow/models.git
RUN             cd models/research  \
             && cd slim \
             && pip3 install -e . \
             && cd .. \
             && /protoc/bin/protoc object_detection/protos/*.proto --python_out=. \
             && export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim \
             && cd delf \
             && /protoc/bin/protoc delf/protos/*.proto --python_out=. \
             && pip3 install -e . \
             && apt-get clean \
             && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

ENV PYTHONPATH=/tensorflow/models/research:/tensorflow/models/research/slim
ENV LC_ALL=C.UTF-8
EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0"]