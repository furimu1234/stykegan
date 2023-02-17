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

COPY ./requirements.txt /project/
COPY ./dnnlib/ /project/
COPY ./metrics/ /project/
COPY ./training/ /project/
COPY ./config.py /project/
COPY ./ic.py /project/

WORKDIR /project/

RUN curl -L "https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ" -o data.pkl


RUN pip install -r requirements.txt

CMD ["python", "ic.py", "1"]