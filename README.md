# Learning to Look: Cognitive Attention Alignment with Vision–Language Models

This repo explores a scalable way to make CNNs “right for the right reasons.” Instead of hand-annotated saliency or concept labels, we use a vision–language model (WeCLIP+) to generate semantic, language-guided attention maps from short prompts (e.g., “digit”). We then train a simple CNN with an auxiliary loss that aligns its internal attention (via CAM) to these pseudo-maps, reducing shortcut reliance without any manual annotation.

The training schedule mirrors the paper: a two-phase procedure with an initial “learn-to-look” stage that optimizes only the attention alignment loss, followed by a reset of the optimizer/scheduler and joint optimization of classification + attention alignment, with a gentle ramp on the KL weight to keep attention prioritized. We evaluate on ColoredMNIST (color ↔ label correlation inverted at test) and DecoyMNIST (class-indicative corner patches). The approach attains state-of-the-art performance on ColoredMNIST and remains competitive on DecoyMNIST, while producing saliency that focuses on digit shape rather than color or patch artifacts.

In short: language-guided attention supervision, no human masks, better generalization. The framework is backbone-agnostic and easily extends beyond LeNet and CAM to other models and differentiable attribution methods.

### Requirements

We recommend using a `conda` environment.

1.  **Create and activate a conda environment:**
    ```sh
    conda create -n learntolook python=3.8
    conda activate learntolook
    ```

2.  **Install dependencies:**
    The required packages are listed in [requirements.txt](requirements.txt). Install them using pip:
    ```sh
    pip install -r requirements.txt
    ```
    *Note: `pydensecrf` may require special installation steps as noted in the requirements file.*

3. **Install Pretrained Weights (Required for WeCLIP+):**
    Download the CLIP ViT-B/16 checkpoint and place it at code/WeCLIPPlus/pretrained/ViT-B-16.pt:
    https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt


### File Structure

-   **`data/`**: Contains scripts to generate the datasets.
    -   [`color_mnist.py`](data/color_mnist.py): Generates the ColorMNIST dataset.
    -   [`decoy_mnist.py`](data/decoy_mnist.py): Generates the DecoyMNIST dataset.
    -   `saved/`: Default output directory for generated datasets (ignored by git).

-   **`code/`**: Contains source code for the models and data utilities.
    -   **`WeCLIPPlus/`**: The WeCLIP+ model implementation, used for generating pseudo-masks.
        -   `configs/`: Configuration files for the model.
        -   `move_data/`: Utility scripts to move and format data for the WeCLIP+ pipeline.
        -   `scripts/dist_clip_voc.py`: The main training script for WeCLIP+.
        -   `test_msc_flip_voc.py`: Script to generate segmentation masks from the trained WeCLIP+ model.
        -   `generate_psuedo_masks.py`: Orchestrates the first stage of the pipeline: setting up data, training WeCLIP+, and generating the pseudo-masks.

-   **`scripts/`**: High-level scripts to run the end-to-end pipeline.
    
    -   [`run_guided_CNN.py`](scripts/run_guided_CNN.py): Implements and trains the LeNet-style CNN, using the pseudo-masks as an attention-guidance signal.

### How to Run

The project runs in three main steps:

1.  **Generate a Dataset**
    First, create the dataset you want to experiment with. For example, to generate ColorMNIST:
    ```sh
    python data/color_mnist.py
    ```
    This will save the images to `data/saved/ColorMNIST_images/`.

2.  **Generate Pseudo-Masks**
    This step uses the WeCLIP+ model to generate segmentation masks for the training images. The [`generate_psuedo_masks.py`](code/WeCLIPPlus/generate_psuedo_masks.py) script automates this process.

    *   **Important:** Before running, you must edit the hardcoded paths inside the scripts in `code/WeCLIPPlus/generate_psuedo_masks/` and the config file at [`code/WeCLIPPlus/configs/voc_attn_reg.yaml`](code/WeCLIPPlus/configs/voc_attn_reg.yaml) to match your local environment.

    Once configured, run the script:
    ```sh
    python code/WeCLIPPlus/generate_psuedo_masks.py --setup-data
    ```
    This will train WeCLIP+ and save the resulting pseudo-masks in a `results/` directory inside `code/WeCLIPPlus/`.

3.  **Train the Guided CNN**
    Finally, train the simple CNN using the original images and the generated pseudo-masks.
    ```sh
    python scripts/run_guided_CNN.py data/saved/ColorMNIST_images/ code/WeCLIPPlus/results/val/prediction_cmap/
    ```
    The first argument is the path to the dataset, and the second is the path to the generated pseudo-masks (`prediction_cmap` folder). The script will train the model and evaluate its performance on the test set, reporting the final



### Acknowledgments

Portions of this repository (the code/WeCLIPPlus directory and related scripts/configs) are derived directly from the WeCLIP+ implementation. We are grateful to the authors for releasing their code and for the foundational contribution their work provides to this project.

If you use this repository or any WeCLIP+ components, please cite:

Zhang, Bingfeng, et al. “Frozen CLIP-DINO: A Strong Backbone for Weakly Supervised Semantic Segmentation.” IEEE Transactions on Pattern Analysis and Machine Intelligence (2025).

All WeCLIP+ code remains under its original license; please refer to the upstream LICENSE for terms. Any mistakes in integration or documentation here are our own.