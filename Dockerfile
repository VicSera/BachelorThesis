FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime
COPY ./requirements.txt /var/requirements.txt
RUN pip install -r /var/requirements.txt
COPY . perpetuumidi
WORKDIR perpetuumidi
EXPOSE 5000
ENTRYPOINT flask run --host=0.0.0.0