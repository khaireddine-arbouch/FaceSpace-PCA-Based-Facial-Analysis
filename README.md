# FaceSpace: PCA-Based Facial Analysis

## Overview
FaceSpace is an advanced project focusing on Principal Component Analysis (PCA) for facial recognition and generation using the CelebFaces Attributes (CelebA) Dataset. This project demonstrates the power of PCA in extracting key facial features and using these features for tasks such as facial reconstruction, face similarity assessment, and generating new facial images.

## Features
- **Data Preparation**: Automates the processing of raw facial images into a standardized numerical format suitable for analysis.
- **PCA Analysis**: Implements PCA to identify and rank the principal components that capture the most significant variance in facial data.
- **Face Reconstruction**: Reconstructs faces using varying numbers of principal components to explore the balance between detail and compression.
- **Celebrity Match**: Identifies the celebrity whose facial features are closest to a provided image based on PCA features.
- **Random Face Generation**: Generates new, synthetic face images by manipulating principal components within a controlled range.

## Installation

### Prerequisites
- Python 3.8 or higher
- Pip for Python package management
- Access to the CelebA dataset

### Libraries Used
- NumPy
- Pandas
- Seaborn
- Matplotlib
- PIL (Pillow)
- scikit-learn

### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/khaireddine-arbouch/FaceSpace-PCA-Based-Facial-Analysis.git
   cd FaceSpace
   ```

2. **Install required Python packages**:
   ```bash
   pip install numpy matplotlib pillow scikit-learn seaborn scikit-image
   ```

3. **Download the CelebA dataset**:
   Ensure you have configured your Kaggle API token as described [here](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication). Then run:
   ```bash
   kaggle datasets download -d jessicali9530/celeba-dataset
   unzip celeba-dataset.zip -d data
   ```

## Usage

Run the main script after navigating to the project directory:
```bash
python pca_facial_recognition.py
```

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgments
- Dr. Fatih Kahraman, Dr. Binnur Kurt, and Bahçeşehir University's Engineering Department for their guidance and resources.
- The creators of the CelebA dataset for providing a rich dataset for academic and research purposes.
