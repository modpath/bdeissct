FROM python:3.10-slim

RUN mkdir /pasteur

# Install bdext
RUN cd /usr/local/ && pip3 install --no-cache-dir bdext==0.1.23

# The entrypoint runs bdpn_infer with command line arguments
ENTRYPOINT ["bdct_infer"]