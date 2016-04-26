FROM python:2.7

RUN apt-get update  && apt-get install -y --no-install-recommends \
    python-numpy \
    python-scipy \
&& apt-get clean \
&& apt-get autoclean \
&& rm -rf /var/lib/apt/lists/*

RUN mkdir pysite

ENV WD=/opt/pylearn
ENV PYTHONPATH=/usr/lib/python2.7/dist-packages:PYTHONPATH
ENV PYTHONUSERBASE=$WD/pysite

COPY . /opt/pylearn
WORKDIR /opt/pylearn

RUN pip install --user .
