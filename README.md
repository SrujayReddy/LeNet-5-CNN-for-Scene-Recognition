
# LeNet-5 CNN for Scene Recognition

## Overview

This repository contains a deep learning project focused on scene recognition using the MiniPlaces dataset. The project involves implementing and customizing a convolutional neural network (CNN), LeNet-5, and experimenting with various hyperparameter configurations. It showcases fundamental AI concepts and practical skills in model design, training, and evaluation using PyTorch.

## Key Features
- Implementation of **LeNet-5** architecture for scene recognition.
- Support for **custom CNN designs** to improve recognition accuracy.
- Experiments with hyperparameters like batch size, learning rate, and epochs.
- Evaluation of models using advanced profiling and checkpointing techniques.
- Integration with the **MiniPlaces dataset** for training, validation, and testing.

## Getting Started

### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch
- torchvision
- tqdm
- PIL

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/SrujayReddy/LeNet-5-CNN-for-Scene-Recognition.git
   cd LeNet-5-CNN-for-Scene-Recognition


2.  Set up the Python environment:
    
    ```bash
    conda create -n miniplaces-env pytorch torchvision torchaudio cpuonly -c pytorch
    conda activate miniplaces-env
    pip install tqdm
    
    ```
    
3.  Download the dataset and extract it:
    
    ```bash
    python dataloader.py
    
    ```
    

### Usage

#### Training the Model

Run the training script:

```bash
python train_miniplaces.py --epochs 10 --lr 0.001 --batch-size 32

```

#### Evaluating the Model

Evaluate the trained model:

```bash
python eval_miniplaces.py --load ./outputs/model_best.pth.tar

```

### Results

Validation accuracy and runtime statistics are saved in `results.txt`. Example configurations include:

-   Batch sizes: 8, 16, 32
-   Learning rates: 0.001, 0.01, 0.05
-   Epochs: 5, 10, 20

## Project Structure

-   `dataloader.py`: Handles dataset downloading, extraction, and preprocessing.
-   `train_miniplaces.py`: Main script for training models.
-   `eval_miniplaces.py`: Script for evaluating trained models.
-   `student_code.py`: Core implementation of the LeNet-5 architecture and utilities.
-   `train.txt` / `val.txt`: Dataset split files for training and validation.

## MiniPlaces Dataset

The MiniPlaces dataset consists of 120,000 images across 100 scene categories. Images are resized to 32x32 for efficient training. For more details, refer to the [MiniPlaces website](http://miniplaces.csail.mit.edu/).

## Results and Analysis

The trained model achieves competitive accuracy with the LeNet-5 baseline. Further details on experiments and performance can be found in the `results.txt` file.

## References

-   [LeNet-5 Paper](http://yann.lecun.com/exdb/lenet)
-   [MiniPlaces Dataset](http://miniplaces.csail.mit.edu/)
-   PyTorch [Documentation](https://pytorch.org/docs)

