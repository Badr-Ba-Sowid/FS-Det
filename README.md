
**Setup**

This repository contains code for a project that requires the following libraries to be installed:

PyTorch (latest version with CUDA)
NumPy
Matplotlib
tqdm
Click
PyYAML
Seaborn
scikit-learn

We recommend creating a new Conda environment and installing the latest version of PyTorch with CUDA support before running the code.

To install the required packages, please run the following command in the terminal:

```bash
pip install -r requirements.txt
```

This will install all the necessary packages in the environment.

**Configuration**

After installing the required packages, you need to setup the configuration yaml file.
Create a directory in the projects working directory and create a ```config.yaml``` file. The file structure should look like this:
```yaml
TRAINING:
  seed: 10
  train_ratio: 0.7
  epochs: 10
  learning_rate: 0.0005
  scheduler_step: 10
  scheduler_gamma: 0.1
  check_point_uri: '{$dir_name}/'
  attention: True
  device_ids: [0,1,2,3,4,5,6]

FEWSHOT:
  n_ways: 5
  k_shots: 5
  n_tasks: 10
  test_k_shots: [5, 10, 15, 20]

DATASET:
  name: '{$dataset_name}'
  dataset_source : '{$dir_name}'
  label_source: '{$dir_name}'
  num_classes: 40
  data_loader_num_workers: 1
  batch_size: 32
  experiment_result_uri: '{$dir_name}'

TESTING:
  model_state: '{$dir_name}'
  dataset_path: '{$dir_name}
```

The configuration.yaml file is a configuration file that contains various parameters and settings used by a machine learning model for training and testing. Here is an explanation of each section of the file:

***TRAINING***

This section contains parameters related to the training process.

* seed: The random seed used for reproducibility of the experiments.
* train_ratio: The ratio of training data to the total data.
* epochs: The number of epochs to train the model.
* learning_rate: The learning rate used for optimization.
* scheduler_step: The number of epochs after which the learning rate is decayed.
* scheduler_gamma: The factor by which the learning rate is multiplied.
* check_point_uri: The directory path where the checkpoints of the trained model will be saved.
* attention: Specifies whether to use attention mechanism in the model. ***(not implemented)***
*device_ids: A list of GPU device IDs to use for training.

***FEWSHOT(Optional)***

This section contains parameters related to the few-shot learning process.

* n_ways: The number of classes in each few-shot task.
* k_shots: The number of examples per class in each few-shot task.
* n_tasks: The number of few-shot tasks to generate during training.
* test_k_shots: A list of the number of examples per class to use for testing.

***DATASET***

This section contains parameters related to the dataset used for training and testing.

* name: The name of the dataset. (used when storing ckpts and plots)
* dataset_source: The file path of the dataset.
* label_source: The file path of the labels for the dataset.
* num_classes: The number of classes in the dataset.
* data_loader_num_workers: The number of worker processes to use for loading data.
* batch_size: The batch size used during training and testing.
* experiment_result_uri: The directory path where the plots of the experiment will be saved.

***TESTING***

This section contains parameters related to the testing process.

* model_state: The file path of the trained model.
* dataset_path: The file path of the dataset used for testing.

Overall, the configuration.yaml file provides a convenient way to specify all the parameters and settings required for training and testing a machine learning model. By editing this file, users can easily modify the model architecture, dataset, hyperparameters, and other settings without having to modify the code.

**Running**

Run the project by executing the Python script.
```bash 
python main.py -c {$PATH_TO_CONFIG_FILE}
```

Please refer to the project documentation for instructions on how to use the code.

If you encounter any issues or have any questions, please feel free to reach out to us.