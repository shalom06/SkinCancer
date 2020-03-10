FROM tensorflow/tensorflow:latest-py3


#RUN apk add --no-cache python3-dev
#
#RUN echo 'manylinux1_compatible = True' > /usr/local/lib/python3.7/site-packages/_manylinux.py


RUN pip3 install --upgrade pip

MAINTAINER Your Name "shalommathews05@gmail.com"

RUN apt-get install -y libsm6 libxext6 libxrender-dev

# We copy just the requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

COPY . /app

EXPOSE 8080


CMD [ "python3","app.py" ]