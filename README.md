# aimslabs: MaCVi 2027 EURS, Track 1 reproduction

Inference code for our Track 1 testing-phase entry (see `repro.yaml` for the submission ID, `METHOD.md` for the method,
data and licences). Made from the organizers' template `CeviKle/EURS-MaCVi2027-repro-template`.

```
run.py            main(model_dir, input_dir, output_dir, device): restored/<stem>.jpg + seg_masks/<stem>.png
src/              model code (restorers, segmenters, SAM 3 rules); restormer_arch.py is the official Restormer (MIT)
weights/          weights_link.yaml: one public zip (~11.6 GB), unpacked into weights/ (licence texts in its LICENSES/)
WEIGHTS_SHA256.txt  SHA-256 of every checkpoint inside that zip
Dockerfile        pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime + requirements.txt (organizer ruling, 27 Oct)
repro_check.py    the organizers' checker, unchanged
```

## Run

```bash
docker build -t aimslabs-t1 .
# weights: download the zip named in weights/weights_link.yaml and unzip it into weights/
docker run --rm --network none --gpus all -v "$PWD/weights":/pkg/weights:ro -v /path/to/degraded:/in:ro -v /tmp/out:/out \
    aimslabs-t1 python run.py --input /in --output /out
```

`input_dir` holds the released test images with their names. Only the trailing `_<type>_<level>` of each name is read
(organizer ruling, 27 Oct). Needs one 24 GB GPU (we used 16 GB) and under 10 GB of RAM. Run time and parity with our
leaderboard zip are in `METHOD.md`.

## License

Our code: MIT (`LICENSE`). `src/restormer_arch.py`: Restormer, MIT (`src/restormer_LICENSE.md`). Weights: see
`LICENSES/README.md` inside the weights archive. SAM 3 is redistributed unchanged under the SAM License, and DINOv3
derivatives under the DINOv3 License.
