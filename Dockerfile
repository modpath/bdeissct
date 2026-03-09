# FROM python:3.10-slim
FROM evolbioinfo/bdext:v0.1.71

# RUN mkdir /pasteur

# Install bdext
RUN cd /usr/local/ && pip3 uninstall -y bdext && pip3 install --no-cache-dir bdext==0.1.74 treesimulator==0.2.26

# The entrypoint runs bdeissct_infer with command line arguments
ENTRYPOINT ["bdeissct_infer"]