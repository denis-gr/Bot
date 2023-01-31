FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
ADD . .
RUN pip3 install -r requirements.txt
CMD python3 .
