FROM pytorch/torchserve:0.5.3-cpu AS buildTORCH

USER root

RUN apt update -qqy && apt upgrade -qqy

COPY requirements.txt .

RUN pip install -r  requirements.txt

RUN chmod -R a+w /home/model-server

COPY dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]
