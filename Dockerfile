FROM python:3.6
USER root

RUN apt-get update
RUN apt-get -y install locales && \
    localedef -f UTF-8 -i ja_JP ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
ENV LC_ALL ja_JP.UTF-8
ENV TZ JST-9
ENV TERM xterm

RUN apt-get install less


RUN pip install -U pip setuptools

COPY ./requirements.txt /project
COPY ./karras2019stylegan-ffhq-1024x1024.pkl /project
COPY ./dnnlib /project
COPY ./metrics /project
COPY ./training /project
COPY ./config.py /project
COPY ./ic.py /project


RUN pip install -r requirements.txt

CMD ["python", "ic.py", "1"]