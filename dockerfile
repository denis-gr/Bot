FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
ADD . /app
RUN pip3 install -U -r requirements.txt
CMD python3 .
