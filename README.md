# Install:
Coded and tested with Python3.8.5, tensorflow2.2.0 (GPU), CUDA10.1, CUDNN7.6.5

* `pip install -r requirements.txt`

# Run

`python benchmark.py --gpu_name "<GPU>" --gpu <GPU_ID>`

* `--gpu_name <GPU>`: **mandatory**, the name of the gpu tested (e.g., "GTX 1080 TI", "RTX 2070")
* `--gpu <GPU_ID>`: **optionnal**, GPU ID for CUDA_VISIBLE_DEVICES option (default=0)

Outputs a result file in ./res/
Comparison plot in ./plots/

# Publish

Push your results file/plots
