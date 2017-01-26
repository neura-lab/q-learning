FROM tensorflow/tensorflow:0.12.1-gpu

RUN apt-get update
RUN apt-get -y install \
  libjpeg8-dev \
  libav-tools \
  python-numpy \
  python-scipy \
  python-pyglet \
  python-setuptools \
  libpq-dev \
  libjpeg-dev \
  curl \
  cmake \
  swig \
  python-opengl \
  libboost-all-dev \
  libsdl2-dev \
  wget \
  unzip \
  git \
  xpra

RUN pip install plotly
RUN pip install gym
RUN pip install 'gym[all]'

RUN pip install tensorbuilder>=0.3.6

RUN pip install sympy

WORKDIR /notebooks

CMD /run_jupyter.sh
