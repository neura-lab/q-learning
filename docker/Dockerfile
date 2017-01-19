FROM tensorflow/tensorflow:1.0.0-alpha

RUN apt-get update
RUN apt-get -y install libjpeg8-dev

RUN pip install plotly
RUN pip install sklearn
RUN pip install asq

RUN pip install tensorbuilder

WORKDIR /notebooks

CMD /run_jupyter.sh
