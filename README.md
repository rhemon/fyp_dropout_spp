# Empirically comparing different regularization methods for student performance prediction with imbalanced data


6CCS3PRJ Final Year Individual Project of Ridhwanul Haque, k1924212, 1913343

## Declaration

I verify that I am the sole author of the programs contained in this folder, except where explicitly stated to the contrary.

## Environment Setup Instruction

<br/>

### Python and Libraries

Python 3.8.11 was used throughout the project and recommended to use the same version. The project was using the setup provided by Anaconda hence recommended to install anaconda and create a python 3 environment with following code:

```
>>> conda create -n py3 python=3.8.11
```

The environment then can be acivated by running:
```
>>> conda activate py3
```
and deactivated by:
```
>>> conda deactivate
```

Install the required libraries by running:
```
>>> pip install -r requirements.txt
```

Here PyTorch is installed for CPU only. If you plan to use GPU accoding cuda support should also be installed. A list of supported versions can be found and installed from [here](https://pytorch.org/get-started/previous-versions/).

Alternatively the code can work with CPU as well but will be significantly slow when training the BLSTM based models on KEATS and Sentiment140 dataset.

### Required Downloads

The datasets are not included in the repository. They can be downloaded from this [link](https://drive.google.com/drive/folders/1RFZyTEdY5ckX8TpBrcBRPHSVolhU81lQ?usp=sharing). Make sure the contents are downloaded into a `raw_data_sets` folder  with no changes in file names as they are hardcoded into the data loaders. The `raw_data_sets` folder must be located in the root directory. 

`KEATS Dataset` folder is required when using `OneHotKEATS` or `BoWKEATS` data loaders.
`Sentiment140` folder is required when using `Sentiment140` dataset. The folder is also needed by the `SimpleSentiment` model as it includes the GloVe embeddings that are loaded in to the models' embedding layer.
Files in `MNIST` folder is not required as `MNISTLoader` uses the dataset coming with `torchvision` which loads and saves the file in that folder.

### Optional Downloads

You can download pre trained checkpoints to evaluate or further train on them. All models discussed in the final report have their checkpoints and can be downloaded from this [Google Drive Link](https://drive.google.com/drive/folders/1p4x7gf2q0Qvfk0ecxOhAscUcps7w6owi?usp=sharing). To evaluate using the CLI, checkpoints must be downloaded into a `model_cpts` directory (placed in the root folder). This is also due to the fact that evaluation mode is hard coded for that. 

### Notebook Downloads

Notebooks are not needed in any of the execution for this project. However they can also be downloaded from [here](https://drive.google.com/drive/folders/1jYwQHZ2399eFM8tR1x49xeYwfCZ0RMVh?usp=sharing). Here, `MNIST_Gradient_Based_Dropout.ipynb` contains the code and results of models with different gradient based dropout method. The checkpoints for this is not available in previously downloaded checkpoints and only executed on Google Colab. The remaining notebooks are in pdf format, however the original ipynb files are in the `raw_data_sets` previously downloaded.

## Configuration File

To train models a json file specifying the configuration must be passed when running the CLI. An example json file is included in the repository `configs/example_config.json`. The configuration variables are also explained in the final report. Here are same key variables that can be modified to train different models on different dataset and splits:

- MODEL: Must be the model name. Valid options are the classes in `models/LinearModel`, `models/GritNet` and `models/SimpleSentiment` each of which are models with or without a specific dropout method.
- MODEL_DIR: Must specify the dir where model is located in, so valid options are: `LinearModel`, `GritNet`, `SimpleSentiment`.
- DATASET_PROCESSOR: Must be a valid data loader classes (`OneHotKEATS`, `BoWKEATS`, `MNISTLoader`, `Sentiment140`)
- OUTPUT_TYPE: `BINARY` (only valid for KEATS and Sentiment), `GRADE` (only valid for KEATS dataset), `NUMBERS` (only valid for MNIST)
- LOAD_METHOD: `ORIGINAL` (or any other random text that wont) or `OVERSAMPLE_NON_MAX_TRAIN` to oversample each minor instance to match instance with max number of example.
- CLASS_SPLIT: A list with number of examples to sample per class. If not specified original split in data is taken. Note this split is taken on the 80% of full dataset which is separated for training. 20% for test is kept as it is.

The other variables are like hyper parameters for training and evaluating models.

The `get_run_command.py` file is a helper script written to create config files for all the models trained during the project. The list `changes` is commented in the script, and be modified to generate specific files. At the end also iterates through the created config file and returns the CLI command to train all the files. Essentially all train commands are separted using `;` so that CLI continues training training one after another.

## CLI

Train command:
```
>>> python main.py -tp configs/example_config.json
```

If training to evaluate code, recommended to train on MNIST or BoWKEATS for quick execution. Training on both Sentiment140 and OneHotKEATS takes time as GritNet and SimpleSentiment model use custom BLSTM implementation which is significantly slower than PyTorch's BLSTM implementation. 

To evaluate from existing checkpoint directory (example command assuming `model_cpts` is downloaded):
```
>>> python main.py -md model_cpts/BoWKEATS/BINARY/LinearModelDropout/03-16-2022-10-39-09
```

