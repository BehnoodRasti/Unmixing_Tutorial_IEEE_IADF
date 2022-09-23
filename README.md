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
You can access Colab by clicking on the link at the top of each notebook (.ipynb files) rendered on Github.

### I have a GPU

If you have access to a GPU, it is preferred that you run the notebook yourself, without using Colab.

In order for the notebooks to run properly, you will need to install some Python packages yourself in a dedicated `conda` environment.
Simply follow the instructions below and copy paste the commands in a terminal.

#### Instructions

---

##### (10 to 20 minutes depending on your Internet connection speed)

We recommend using `conda` to handle the Python distribution and `pip` to install the Python packages.

1. First clone the repository on your computer.

```
git clone https://github.com/BehnoodRasti/Unmixing_Tutorial_IEEE_IADF.git
cd Unmixing_Tutorial_IEEE_IADF
```

2. Create a dedicated `conda` environment named `iadf` for Python 3.8.

```
conda create --name iadf python=3.8
```

* If you do not have `conda` on your computer, first install it using [Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).

3. Activate the `conda` environment.

```
conda activate iadf
```

4. Install the required Python packages.

```
pip install -r requirements.txt
```

5. Install the relevant `ipython` kernel to be used in Jupyter.

```
ipython kernel install --user --name iadf
```

6. Launch `jupyter lab`. This will open a new tab in your browser.

```
jupyter lab
```

7. Open any notebook from the repository and make sure you select the right Kernel. In the menu, click on `Kernel` and select `Change Kernel...`. Finally, pick the `iadf` kernel and you are ready to go!

Note that the notebooks were primarily designed to function with Colab. As such, you may encounter code that is not useful when running on a standalone notebook. We still recommend that you run the notebook linearly to make sure everything works as intended. Address any questions to the course instructors.


#### My GPU is on a remote server - How can I render the notebooks locally?

1. **On your remote server**, launch `jupyter lab` on a dedicated port:

```
jupyter lab --no-browser --port 1234
```

2. To be able to display the notebooks on your **local browser**, you need to setup a `ssh` tunnel in a terminal on your **local** machine:

```
ssh -NL 1234:localhost:1234 <remote_username>@<remote_machine_address> -f
```

* Please change `remote_username` and `remote_machine_address` according to your specific configuration.

3. Now open your favorite browser at the following address: http://localhost:1234/lab

4. You may need to copy paste the jupyterlab-generated URL that contains a token from your remote server to your local browser, *e.g.* http://localhost:1234/lab?token=e76fd55267495be44b79e3d2313257faf68a7a47de4969b5
