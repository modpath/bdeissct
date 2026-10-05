# FROM python:3.10-slim
FROM evolbioinfo/bdext:v0.2.0

# RUN mkdir /pasteur

# Install bdext
RUN cd /usr/local/ && pip3 uninstall -y bdext && pip3 install --no-cache-dir bdext==0.2.9

# The entrypoint runs command line with command line arguments
ENTRYPOINT ["/bin/bash"]