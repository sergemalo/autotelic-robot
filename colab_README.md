# step-by-step

1. Clone repo

2. Install dependencies
    - `%%bash
    apt-get update -q
    apt-get install -y -q cmake build-essential \
        libgl1-mesa-dev libgl1-mesa-glx libglew-dev \
        libosmesa6-dev patchelf ffmpeg`
    - `!pip install -r colab_requirements.txt`

3. Download dataset
    - `%%bash
    cd /content/autotelic-robot/src/libero
    echo "N" | python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial --use-huggingface`

4. setup macros
    - `%%bash
    python /usr/local/lib/python3.*/dist-packages/robosuite/scripts/setup_macros.py`

5. Run sanity checks
    - `!python check_pytorch.py`
    - `!python check_libero.py`