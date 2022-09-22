# Unmixing-Tutorial
This repo is prepared for IEEE Image Analysis and Data Fusion (IADF) Summer school 3rd Oct. 2022. 

## Abstract of the Course

This course will discuss linear hyperspectral unmixing, including Geometrical Approaches, Blind Linear Unmixing, and Sparse Unmixing. The course will further discuss Autoencoders and Convolutional Networks for unmixing. The hands-on session aims to train participants to use the advanced open-source machine and deep learning-based spectral unmixing techniques and fine-tune the models.

This course contains an open source python-based repository. The repository contains two supervised unmixing, one sparse (semisupervised) unmixing, and four blind unximing. The list of the methods are as follow. Colab version of the notebooks are provided for all the methods. 

* **Supervised Unmixing**:

1. FCLSU: Fully constraint least square unmixing. VCA is used for endmember extraction.
2. UnDIP: Unmixing using Deep Image Prior. SiVM is used for endmember extraction.

* **Sparse (Semisupervised) Unmixing**:

3. SUnCNN: Sparse Unmxing using Unsupervised Convolutional Neural Network

* **Blind Unmixing**:

4. MiSiCNet: Minimum Simplex Convolutional Network for Deep Hyperspectral Unmixing
5. EDAA: Entropic Descent Archetypal Analysis for Blind Hyperspectral Unmixing
6. CNNAE: Convolutional Autoencoder for Spectral–Spatial Hyperspectral Unmixing
7. EndNet: Sparse AutoEncoder Network for Endmember Extraction and Hyperspectral Unmixing


## Usage

---

### I do not have a GPU

For users that do not have access to a GPU, we recommend to use the notebooks directly in Google Colab.
You can access Colab by clicking on the link at the top of each notebook (.ipynb files).

### I have a GPU

If you have access to a GPU, it is preferred that you run the notebook yourself, without using Colab.

In order for the notebooks to run properly, you will need to install some Python packages yourself in a dedicated `conda` environment.
Simply follow the instructions below.

#### Instructions

We recommend using `conda` to handle the Python distribution and `pip` to install the Python packages.

1. First clone the repository on your computer.

```
git clone https://github.com/BehnoodRasti/Unmixing_Tutorial_IEEE_IADF.git
cd Unmixing_Tutorial_IEEE_IADF
```

2. Create a dedicated `conda` environment, named `iadf`, based on the `conda.txt` file.

```
conda create --name iadf --file conda.txt
```

3. Activate the `conda` environment

```
conda activate iadf
```

4. Install the required Python packages

```
pip install -r requirements.txt
```
