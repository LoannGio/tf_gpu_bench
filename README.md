# Install:
Requirements: Python >= 3.6

Run:
* `virtualenv . -p python3`
* `source bin/activate`
* `pip install -r requirements.txt`

# Run

`python benchmark.py --gpu_name "<GPU> --gpu <GPU_ID>"`

* `--gpu_name <GPU>`: **mandatory**, the name of the gpu tested (e.g., "GTX 1080 TI", "RTX 2070")
* `--gpu <GPU_ID>`: **optionnal**, GPU ID for CUDA_VISIBLE_DEVICES option

Outputs a result file in ./res/

# Publish

Push your result file 