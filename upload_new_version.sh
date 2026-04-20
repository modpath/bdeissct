rm -rf bdpn.egg-info build dist
python setup.py sdist bdist_wheel
twine upload dist/* && \
sudo docker build -t evolbioinfo/bdext:v0.1.96 -f Dockerfile . && sudo docker login && sudo docker push evolbioinfo/bdext:v0.1.96