#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

ARG base_img

FROM $base_img
WORKDIR /

# Reset to root to run installation tasks
USER 0

RUN mkdir ${SPARK_HOME}/python

# add llvm dependencies
RUN apk --no-cache add \
        bash \
	autoconf \
	automake \
	freetype-dev \
	g++ \
	gcc \
	cmake \
	make \
        libxml2-dev \
        python3-dev \
        ncurses-dev \
        openblas-dev \
        boost-dev \
        zlib-dev \
        flex \
        bison

RUN ln -sf `which python3` `which python` && \
    cp -p /usr/bin/pip3.6 /usr/bin/pip && \
    cp -p /usr/bin/python3-config /usr/bin/python-config

RUN python -m ensurepip && \
    # We remove ensurepip since it adds no functionality since pip is
    # installed on the image and it just takes up 1.6MB on the image
    rm -r /usr/lib/python*/ensurepip && \
    pip install --no-cache-dir --upgrade pip setuptools

RUN pip install --no-cache-dir numpy six pytest numpy cython
RUN pip install --no-cache-dir pandas

ARG ARROW_VERSION=0.12.0
ARG ARROW_SHA1=2ede75769e12df972f0acdfddd53ab15d11e0ac2
ARG ARROW_BUILD_TYPE=release

ENV ARROW_HOME=/usr/local \
    PARQUET_HOME=/usr/local

#Download and build apache-arrow
RUN mkdir /arrow \
    && apk add --no-cache curl \
    && curl -o /tmp/apache-arrow.tar.gz -SL https://github.com/apache/arrow/archive/apache-arrow-${ARROW_VERSION}.tar.gz \
    && echo "$ARROW_SHA1 *apache-arrow.tar.gz" | sha1sum /tmp/apache-arrow.tar.gz \
    && tar -xf /tmp/apache-arrow.tar.gz -C /arrow --strip-components 1 \
    && mkdir -p /arrow/cpp/build \
    && cd /arrow/cpp/build \
    && cmake -DCMAKE_BUILD_TYPE=$ARROW_BUILD_TYPE \
          -DCMAKE_INSTALL_LIBDIR=lib \
          -DCMAKE_INSTALL_PREFIX=$ARROW_HOME \
          -DARROW_PARQUET=on \
          -DARROW_PYTHON=on \
          -DARROW_PLASMA=on \
          -DARROW_BUILD_TESTS=OFF \
          .. \
    && make -j$(nproc) \
    && make install \
    && cd /arrow/python \
    && python setup.py build_ext --build-type=$ARROW_BUILD_TYPE --with-parquet \
    && python setup.py install \
    && rm -rf /arrow /tmp/apache-arrow.tar.gz

RUN pip install --no-cache-dir py4j
RUN pip install --no-cache-dir scipy
RUN pip install --no-cache-dir jinja2
RUN pip install --no-cache-dir cloudpickle
RUN pip install --no-cache-dir lz4

# configure LLVM using CMake
RUN wget http://releases.llvm.org/8.0.0/llvm-8.0.0.src.tar.xz && \
    tar xf llvm-8.0.0.src.tar.xz && \
    cd llvm-8.0.0.src && mkdir build && cd build && cmake .. \
    -G "Unix Makefiles" -DLLVM_TARGETS_TO_BUILD="X86" \
    -DCMAKE_BUILD_TYPE=MinSizeRel && \
    # build and install LLVM
    make -j$(nproc) && make install && \
    # cleanup
    cd ../../ && rm -r llvm-8.0.0.src llvm-8.0.0.src.tar.xz

RUN pip install --no-cache-dir numba

RUN pip install --no-cache-dir coffea

COPY python/pyspark ${SPARK_HOME}/python/pyspark
COPY python/lib ${SPARK_HOME}/python/lib

ENV PYSPARK_MAJOR_PYTHON_VERSION="3"
ENV PYTHONPATH=/opt/spark/python/lib/pyspark.zip:$PYTHONPATH

WORKDIR /opt/spark/work-dir
ENTRYPOINT [ "/opt/entrypoint.sh" ]

# Specify the User that the actual main process will run as
ARG spark_uid=185
USER ${spark_uid}
