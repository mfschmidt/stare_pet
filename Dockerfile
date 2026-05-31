FROM ubuntu:resolute

# Upgrading to ubuntu:noble also upgrades to python 3.12 and gcc 13.
# ITK cannot be built with gcc 13 until its version 5.4 is released.
# So stay with jammy until ITK 5.4, then test PetPVC with it.
# -Mike 6/10/2024

# Compiling software with ubuntu:resolute, python3.12, and ITK 5.4.6 (latest)
# works fine. I've added two environment variables to avoid being
# asked about regions for time zones. I removed libgl1-mesa-glx as it's
# not available in resolute and things worked without it.
# -Mike 6/27/2026

ENV DEBIAN_FRONTEND="noninteractive"
ENV TZ="America/New_York"
RUN apt update && \
    apt install -y --no-install-recommends \
        curl wget git ca-certificates build-essential cmake \
        python3.14 python3-pip python3-venv xvfb && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install uv to handle virtual environments and python packages
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# Install ITK as a dependency of PVC
# ITK <5.4.0 will not compile with gcc>13, which is default with ubuntu:noble
# ITK 5.4.6 compiles fine with the default gcc 15 in ubuntu:resolute
RUN cd /opt/ && \
    wget https://github.com/InsightSoftwareConsortium/ITK/releases/download/v5.4.6/InsightToolkit-5.4.6.tar.gz && \
    tar -xzf InsightToolkit-5.4.6.tar.gz && \
    mkdir -p /opt/build_itk && cd /opt/build_itk && \
    cmake ../InsightToolkit-5.4.6 -DITK_BUILD_DEFAULT_MODULES:BOOL=ON -DModule_ITKReview:BOOL=ON && \
    make && make install && \
    rm /opt/InsightToolkit-5.4.6.tar.gz

# Install PVC
# The latest version since 12/10/2024 has been 1.2.12; upgraded from 1.2.11
# CMake is v4.2.3 on ubuntu:resolute; PETPVC wants >2.8, so all seems OK.
# But cmake doesn't want to compile without the -D flag.
RUN cd /opt/ && \
    git clone https://github.com/UCL/PETPVC.git && \
    cd /opt/PETPVC && git checkout tags/v1.2.12 -b starepet && \
    mkdir /opt/build_petpvc && cd /opt/build_petpvc && \
    cmake /opt/PETPVC -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && \
    make && make install

# STARE code is already downloaded/pulled to get this Dockerfile
COPY src /opt/stare_pet/src
COPY pyproject.toml README.md /opt/stare_pet/
# Or RUN git clone https://github.com/mfschmidt/stare_pet.git

WORKDIR /opt/stare_pet
RUN uv lock && uv sync --frozen --no-dev
ENV PATH="/opt/stare_pet/.venv/bin:$PATH"

# ENTRYPOINT ["/opt/stare_pet/.venv/bin/python3", "/opt/stare_pet/.venv/bin/stare_app.py"]
# ENTRYPOINT ["/root/.local/bin/uv", "--directory", "/opt/stare_pet", "run", "/opt/stare_pet/.venv/bin/stare"]
ENTRYPOINT ["python3", "/opt/stare_pet/.venv/bin/stare"]
# ENTRYPOINT ["python3", "/opt/stare_pet/src/stare_pet/stare_app.py"]
