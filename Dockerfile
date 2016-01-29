FROM terriajs/inference:baseos
RUN mkdir -p /usr/src/dora
COPY . /usr/src/dora/
RUN cd /usr/src/dora \
  && pip3 install -e ".[server]"


