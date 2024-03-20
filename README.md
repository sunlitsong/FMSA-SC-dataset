# FGMSA

`feature.pkl` is a feature file containing training, validation and test set.

`models_trained` is a folder containing trained models.

`results` is a folder storing training and test results.

## 1. Get Started with Python APIs

### 1.1 run_train

Use this function to train the model and test its performance.

**Definition**:

```python3
def run_train(
        model_name: str,
        seeds=None,
        is_tune: bool = False,
        tune_times: int = 50,
        gpu_id=None,
        num_workers: int = 4,
        model_save_name: str = "",
        config_path: str="./FGMSA_code/config.json",
):
```

### 1.2 run_test

Use this function to test the performance of a trained model.

**Definition**:

```python3
def run_test(
        model_name: str,
        model_path: str,
        config_path: str,
        gpu_id=None,
        num_workers: int = 4,
):
```

## 2. Dependency Installation

Normally the pip install command will handle the python dependency packages. The required packages are listed below for
clarity:

- python >= 3.8

- torch >= 1.9.1

- transformers >= 4.4.0

- numpy >= 1.20.3

- pandas >= 1.2.5

- tqdm >= 4.62.2

- scikit-learn >= 0.24.2

- easydict >= 1.9

  



## 3. Configuration Files

The configuration files for all the best performing models are placed in "FGMSA_code/config". If you want to set up your own configuration file, you can mimic the format given and set the hyperparameters as per your requirement.

The file "config_tune.json" is used for grid searching hyperparameters. If you want to tune the parameters, please set "is_tune" to True in the "run_train" function.

## 4. Our Experiments

If you want to reproduce our experimental results, please use the "run_test" fuciton to test the corresponding \*.pth file( in the folder "models_trained"), making sure to keep the parameter configuration file consistent with the corresponding \*.pth file. We have provided an example in the "run_test.py".

## 5. Dataset 

The feature file is large (about 16G) and can be downloaded from [**Baidu Cloud**](https://pan.baidu.com/s/1psQAiTEMPIlUX-ywW-YwAA?pwd=fmsa ) to run the souce code, the feature file should be placed in the folder "MCSA".

## 6. Some Notes

1. Example file for using grid search for hyperparameter tuning: "train_tune.py"
2. All best model \*.pth files: in the folder "models_trained" 
3. Configuration files for all best models: in the folder "FGMSA_code/config" 
4. Folder where the dataset feature files should be placed: "MCSA"
5. Example file for reproducing the training of our best performing model: "run_train.py"
6. Example file for testing the results of our \*.pth model: "run_test.py"