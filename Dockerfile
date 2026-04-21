# FROM python:3.10-slim
FROM evolbioinfo/bdext:v0.1.97

# RUN mkdir /pasteur

# Install bdext
RUN cd /usr/local/ && pip3 uninstall -y bdext && pip3 install --no-cache-dir bdext==0.1.98

# The entrypoint runs command line with command line arguments
ENTRYPOINT ["/bin/bash"]