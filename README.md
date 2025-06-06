# Deep Learning Sequence Processing with RNNs

This repository contains the implementation of encoder-decoder recurrent neural networks (RNNs) for sequence-to-sequence learning of arithmetic operations. The project focuses on building models that can learn the principles behind simple arithmetic operations (integer addition, subtraction, and multiplication).

## Project Overview

This project explores three different types of sequence-to-sequence models based on input/output data modalities:

1. **Text-to-Text**: Given a text query containing two integers and an operand (+ or -), the model outputs a sequence of integers that match the arithmetic result.

2. **Image-to-Text**: Similar to text-to-text, but the query is specified as a sequence of images containing individual digits and an operand.

3. **Text-to-Image**: The query is specified in text format, but the model's output is a sequence of images corresponding to the correct result.

## Tasks

The project is divided into two main tasks:

### Task 1: Generative Adversarial Networks (GANs)
- Implementation of GANs for image generation
- Training on the Flowers dataset
- Evaluation of generated images

### Task 2: Sequence-to-Sequence Learning
- Implementation of encoder-decoder RNN architectures
- Training models for arithmetic operations
- Evaluation of different modalities (text-to-text, image-to-text, text-to-image)

## Repository Structure

- `Task1.py`: Python script for GAN implementation
- `Task1.ipynb`: Jupyter notebook for GAN implementation with visualizations
- `Task2.py`: Python script for sequence-to-sequence RNN implementation
- `Task2.ipynb`: Jupyter notebook for sequence-to-sequence RNN implementation with visualizations
- `IDL_assignment2_report.pdf`: Detailed report of the project methodology and results

## Requirements

The project requires the following libraries:
- TensorFlow 2.x
- NumPy
- Matplotlib
- OpenCV
- TensorFlow Datasets
- TensorFlow Addons
- scikit-learn
- pandas

## Setup Instructions

1. Clone this repository
2. Install the required packages:
   ```bash
   pip install tensorflow tensorflow-datasets tensorflow-addons numpy matplotlib opencv-python scikit-learn pandas tqdm
   ```
3. Run the Jupyter notebooks or Python scripts:
   ```bash
   # For Task 1 (GANs)
   python Task1.py
   
   # For Task 2 (Sequence-to-Sequence RNNs)
   python Task2.py
   ```

## Models

### Task 1: GAN Models
- Generator: Uses transposed convolutions to generate images
- Discriminator: Uses convolutional layers to classify real vs. fake images

### Task 2: Sequence-to-Sequence Models
- Text-to-Text: LSTM/GRU-based encoder-decoder architecture
- Image-to-Text: CNN + RNN encoder with RNN decoder
- Text-to-Image: RNN encoder with CNN + RNN decoder

## Results

The models are evaluated based on:
- Task 1: Visual quality of generated images and training stability
- Task 2: Accuracy of arithmetic operations across different modalities

Detailed results and analysis can be found in the accompanying report.

## Author

Praneeth Dathu, Jelte, Burak

## License

This project is available under the MIT License.

