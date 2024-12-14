FROM ubuntu:jammy

# Upgrading to ubuntu:noble also upgrades to python 3.12 and gcc 13.
# ITK cannot be built with gcc 13 until its version 5.4 is released.
# So stay with jammy until ITK 5.4, then test PetPVC with it.
# -Mike 6/10/2024

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl wget git ca-certificates build-essential cmake \
        python3.10 python3-pip python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    python3 -m venv /venv

# Install ITK as a dependency of PVC
# ITK <5.4.0 will not compile with gcc>13, which is default with ubuntu:noble
RUN cd /opt/ && \
    wget https://github.com/InsightSoftwareConsortium/ITK/releases/download/v5.2.1/InsightToolkit-5.2.1.tar.gz && \
    tar -xzf InsightToolkit-5.2.1.tar.gz && \
    mkdir -p /opt/build_itk && cd /opt/build_itk && \
    cmake ../InsightToolkit-5.2.1 -DITK_BUILD_DEFAULT_MODULES:BOOL=ON -DModule_ITKReview:BOOL=ON && \
    make && make install && \
    rm /opt/InsightToolkit-5.2.1.tar.gz

# Install PVC
RUN cd /opt/ && \
    git clone https://github.com/UCL/PETPVC.git && \
    cd /opt/PETPVC && git checkout tags/v1.2.11 -b starepet && \
    mkdir /opt/build_petpvc && cd /opt/build_petpvc && \
    cmake /opt/PETPVC && make && make install

# STARE code is already downloaded/pulled to get this Dockerfile
COPY . /stare_pet

RUN /venv/bin/pip install /stare_pet

ENTRYPOINT ["/venv/bin/python3", "/venv/bin/stare"]
