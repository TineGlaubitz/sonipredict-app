FROM python:3.11.2-slim-buster
RUN  apt-get update && apt-get install -y gcc g++

RUN useradd sonipredict
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100 \
    POETRY_VERSION=1.1.4

COPY requirements.txt /requirements.txt
RUN pip install -r requirements.txt

WORKDIR /home/sonipredict

COPY sonipredict ./sonipredict
COPY sonipredict/assets ./sonipredict/assets

COPY logging.ini .

CMD gunicorn sonipredict.app:server -b=0.0.0.0:$PORT --log-config=logging.ini  --timeout 30 --keep-alive 30  --worker-tmp-dir /dev/shm 


