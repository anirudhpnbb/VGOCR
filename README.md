# VGOCR

## Overview

VGOCR is a comprehensive Optical Character Recognition (OCR) system that utilizes advanced deep learning techniques. This repository includes implementations of Variational Autoencoders (VAE) and Generative Adversarial Networks (GAN) to generate and augment data for training robust OCR models.

## Features

- **VAE and GAN Models**: Enhance data for improved OCR performance.
- **OCR Detection**: Accurate detection and recognition of text from images.
- **Synthetic Data Generation**: Generate synthetic images to boost training data.
- **MNIST Dataset**: Utilizes the MNIST dataset for training.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anirudhpnbb/VGOCR.git
   ```
2. Navigate to the project directory:
   ```bash
   cd VGOCR
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train the model**:
   ```bash
   python3 train.py --config config.json
   ```

## File Structure

- `train.py`: Script to train the OCR model.
- `VAE.py`: Script to train the VAE model.
- `gan.py`: Script to train the GAN model.
- `ocr.py`: Script to run OCR.
- `data_loader.py`: Script for loading the MNIST dataset.
- `requirements.txt`: List of dependencies.
- `config.json`: Configuration file for training.

## Folder Structure

```
VGOCR/
│
├── data/
│   ├── mnist/              # Directory containing MNIST dataset
│
├── models/
│   ├── VAE.py              # VAE model implementation
│   ├── gan.py              # GAN model implementation
│
├── scripts/
│   ├── train.py            # Training script
│   ├── ocr.py              # OCR script
│   ├── data_loader.py      # Data loader script
│
├── requirements.txt        # Project dependencies
├── config.json             # Configuration file
├── README.md               # Project README
├── LICENSE                 # Project license
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

For questions or suggestions, please contact [anirudhpnbb](https://github.com/anirudhpnbb).