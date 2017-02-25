FROM tensorflow/tensorflow:1.0.0

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

RUN pip install phi

RUN pip install sympy
RUN echo "hola2"
RUN pip install coconut-develop --upgrade

WORKDIR /notebooks

CMD /run_jupyter.sh
