# Track 1 image (organizer ruling 27 Oct). torch 2.5.1 + torchvision come from the base image; requirements.txt has the rest.
# weights/ is not copied: the organizers mount it read-only at /pkg/weights. No network at run time.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
COPY requirements.txt /tmp/r.txt
RUN pip install --no-cache-dir -r /tmp/r.txt
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
COPY . /pkg
WORKDIR /pkg
