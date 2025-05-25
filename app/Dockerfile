FROM continuumio/miniconda3

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt

RUN conda create -n m3site python=3.11 dssp -c ostrokach -y
RUN conda run -n m3site pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
    PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    SYSTEM=spaces

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

CMD ["conda", "run", "--no-capture-output", "-n", "m3site", "python", "app.py"]
