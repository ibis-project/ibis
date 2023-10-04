FROM python:3.11-alpine

RUN set -ex; \
  python -m pip install --upgrade pip; \
  pip install kafka-python;

ADD generate_source_data.py /

CMD ["python", "-u", "./generate_source_data.py"]
