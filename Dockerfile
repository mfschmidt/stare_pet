FROM ubuntu:jammy

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl wget git ca-certificates build-essential cmake \
        python3.10 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install ITK as a dependency of PVC
RUN cd /opt/ && \
    wget https://github.com/InsightSoftwareConsortium/ITK/releases/download/v5.2.1/InsightToolkit-5.2.1.tar.gz && \
    tar -xzf InsightToolkit-5.2.1.tar.gz && \
    mkdir /opt/build_itk && cd /opt/build_itk && \
    cmake ../InsightToolkit-5.2.1 -DITK_BUILD_DEFAULT_MODULES:BOOL=ON -DModule_ITKReview:BOOL=ON && \
    make && make install

# Install PVC
RUN cd /opt/ && \
    git clone https://github.com/UCL/PETPVC.git && \
    cd /opt/PETPVC && git checkout tags/v1.2.10 -b starepet && \
    mkdir /opt/build_petpvc && cd /opt/build_petpvc && \
    cmake /opt/PETPVC && make && make install

# STARE code is already downloaded/pulled to get this Dockerfile
COPY . /stare_pet

RUN pip install /stare_pet

ENTRYPOINT ["/usr/local/bin/stare"]
