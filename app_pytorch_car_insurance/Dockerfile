FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
LABEL maintainer="yunusemremidilli@gmail.com"

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

COPY ./requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
    rm -rf /tmp/requirements.txt

# # Configure Jupyter Lab to use dark theme
RUN mkdir -p /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/

RUN echo '{ \
  "theme": "JupyterLab Dark" \
}' > /root/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
