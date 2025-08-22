# PyTorch Learning Repository

Welcome to my PyTorch learning repository! Here, I document my progress with notebooks and projects highlighting PyTorch’s capabilities and exploring its fundamentals in machine learning and deep learning.

The repository is organized into sections:

Fundamental Concepts: A series of Jupyter notebooks focused on learning and mastering the foundational elements of TensorFlow. Applied Projects: Practical projects showcasing my application of TensorFlow to solve diverse challenges and real-world problems. This repository is a work in progress. I'll update the README and content as I continue on this journey.

### Fundamental Concepts: 
#### 00_PyTorch_Notebook:
- PyTorch fundamentals content:
  * Create torch tensors and show properties like `item()` or `dim.`
  * Create random torch tensors using `torch.rand(rows, cols)`
  * Create zeroes and one's tensors.
  * Create tensor ranges, and investigate the different tensor datatypes.
  * Getting tensor attributes: `dtype`, `shape`, `device`.
  * Tensors with addition, subtraction, multiplication, division, matrix multiplication.
  * Proper matrix multiplication with proper inner shapes.
  * tensor aggregation for `sum()`, `min()`, `max()`, `mean()`.
  * Reshaping, stacking, squeezing, unsqueezing, and permute methods applications.
  * Indexing 2D, 3D, and ND torch tensors.
  * Cast torch and numpy tensors using `torch.from_numpy(ndarray)`, and `torch.Tensor.numpy()`.
  * Cast between PyTorch and NumPy.
  * PyTorch reproducibility using `torch.manual_seed(RANDOM_SEED)`
  * PyTorch agnostic duda device setup., exchanging tensors between CPU and GPUs.
- PyTorch fundamentals solutions/extra-curriculum (**Complete**):
  * Contains solutions of exercises from: https://www.learnPyTorch.io/00_PyTorch_fundamentals/#exercises
  * Contains solutions of extra-curriculum from: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/00_pytorch_fundamentals_exercises.ipynb

### 01_PyTorch_Workflow:
- PyTorch Workflow content:
  * Data preparation and loading in tensor formats.
  * Splitting data into training and testing splits and plotting them.
  * Building a Model with random values (Handmade single-value linear regression).
  * Understanding PyTorch essentials: `torch.nn`, `torch.nn.Parameter`, `torch.nn.Module`,
    `torch.optim`, overwriting `forward()`
  * Using `torch.inference_mode()` on an untrained model and see its predictions.
  * Fitting a model to training data.
  * Making predictions and evaluating it (inference).
  * Saving loading a model.
  * Putting all together.
- PyTorch Workflow solutions (**Complete**):
  * Contains solutions of exercises from: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/01_pytorch_workflow_exercises.ipynb

### 02_PyTorch_Classification_Workflow
- PyTorch Classification Content:
  * Made Classification data using `make_circles()` from `sklearn.datasets` for two circles.
  * Turn data into `torch.Tensor()` and split them into training, validation, and test data using `train_test_split()`.
  * Build a model with device agnostic code using `nn.Linear()` and `nn.Module()`; We compared too against `nn.Sequential()`.
  * TO POINT OUT, this model didn't have any non-linear activation functions, and we would see how this doesn't work with non-linear data.
  * Setup an optimizer for binary classification, we used `nn.BCEWithLogitsLoss()` and `torch.optim.SGD()`
  * Define your own accuracy function and train the model
  * Define properly how to go from raw logits (output by model) -> prediction probabilities (activation function `torch.sigmoid()`) -> prediction labels (thresholding at 0.5).
  * Set up a training and testing loop to test out this model.
  * Plot the boundary lines and get it from helper functions: https://github.com/mrdbourke/PyTorch-deep-learning/blob/main/helper_functions.py
  * Improving the model (adding more hidden units, number of layers, and number of epochs). This is not gonna work :)
  * Retrain the test on this new model, visualize why even after adding much more power, it doesn't work with non-linear data.
  * Troubleshoot our model by creating linear regression data.
  * Setup a `nn.L1Loss()` and `optim.SGD()` to train on linear regression data.
  * Realize that it works! and therefore, for the Circle model, we need to add non-linear activation functions to allow the lines to express non-linear boundary decisions
  * Create a new model, but this time using `nn.RELU()` between each linear layer.
  * Train and visualize it, it works!
  * Replicate non linear activation functions: `torch.relu()`, `torch.sigmoid()`
  * From the multi-class data set of `sklearn.datasets`, use `make_blobs` to create a multiclassification problem with 4 classes, 2 features.
  * Plot these and visualize them.
  * Create a multiclassification model using `nn.Module`. This time, the constructor is way more advanced for the parameters of in_features and out_features.
  * setup `CrossEntropyLoss()` and `optim.SGD()`.
  * Obtain the (raw output of model logits) -> Predicitons probabilities (`torch.softmax()` across columns dim = 1) -> Prediction labels (`torch.argmax()` across columns dim = 1).
  * Train this model and visualize it! Ir works. Last thing is to repeat this multiclassification model without using non-linear activation functions
  * Realize that deep learning models without activation functions would still work for linear separable data.
  * Dive into some classification metrics.
- PyTorch Classification solutions/extra-curriculum (**Completed**):
  * Contains solutions of exercises from: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/02_pytorch_classification_exercises.ipynb

### 03_PyTorch_computer_vision (Completed)
- PyTorch Computer vision model:
  * from `torchvision.datasets`, obtain the `FashionMNIST` datasets for training and testing data.
  * Look at the shapes, classes, and dimensions of the data.
  * Visualize the data for different classes (Exploratory EDA).
  * Using `DataLoader` from `torch.utils.data`; create two Dataloaders with batch size of 32.
  * Understand how these Dataloader objects work and sample one pass: Input size: `(32, 1, 28, 28)`, Label size: `(32)`.
  * Create a baseline model using `nn.Linear()` and `nn.Flatten()` layers.
  * Initiate this model on CPU.
  * Download `helper_functions.py` using `requests` library. Later import the accuracy function to use.
  * Create a code to print training time of a block of code using `default_timer()` from `timeit` module.
  * Repeat a training and testing loop, this time batched.
  * Create an `eval_model()` function to obtain results from baseline model.
  * This time, create a second model, using `nn.Linear()`, `nn.Flatten()`, and `nn.Relu()` nonlinear activation functions.
  * This time, create a `train_step()` and `test_step()` function to functionalize the batch training/testing of a single step/pass: Input size `(32, 1, 28, 28)`
  * Create a training loop using the past functions and time it, this model was run on GPU.
  * Compare both model results.
  * This time, create a third model that is resembling TinyVGGNet convolutional neural layer architecture.
  * This new model is composed of 2 convolutional blocks: `nn.Conv2D()`, `nn.Relu()`, `nn.MaxPool2d()`; and one classifier: `nn.Flatten()`, `nn.Linear()`.
  * Make coding steps with random inputs to understand `nn.Conv2D()`, `nn.MaxPool2D`, and the new model behaviour.
  * Train the model with "train_step()" and "test_step() code and compare against the other linear neural networks models.
  * Realize the CNN model has higher test and training accuracies than the other models.
  * Compare model results and training time
  * Create a `make_predictions()` funtion and retrieve some samples againts truth vs predicted.
  * Save the model using `torch.save()`, `torch.load()`, and `model.state_dict()`
- PyTorch Workflow solutions (**Completed**):
  * Contains solutions of exercises from: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/03_pytorch_computer_vision_exercises.ipynb

### 04_PyTorch_custom_dataset (Completed):
- PyTorch Custom Dataset:
  * Import the data using `requests`, `zipfile`, and `pathlib` from: https://github.com/mrdbourke/PyTorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip
  * Look at the data preparation and data exploration (visualize the image, class, height, and width).
  * Using `matplotlib.pyplot` and `numpy`, show it.
  * using `torchvision`, use `datasets` and `transforms` to transform the data and visualize it.
  * Create a function that plots othe riginal vs the transformed image.
  * Use `ImageFolder` from `datasets` to load the data since we have it in that folder structure.
  * Turn it into a `Dataloader` and sample from it.
  * Create a function called `find_classes` that finds the classes based on the directory name structure
  * Create an `ImageFolderCustom`, inheriting from `Dataset`, overwriting the `__len__` and `__getitem__` methods.
  * Create a transform and use the `ImageFolderCustom` to create train and test custom datasets.
  * Create a function to display them, as well as Dataloaders custom .
  * Investigate Data Augmentation methods from `torchvision.transforms.TrivialAugmentWide(num_magnitude_bins = 31)`
  * Create a `TinyVGG` model class and test the train_dataloader_simple to see if it works.
  * Use `torchinfo` to get an idea of the shapes of the model.
  * Create a `train_step()`, `test_step()`, and `train()` functions.
  * Setup a `nn.CrossEntropyLoss()` and `torch.optim.Adam()` for optimizing.
  * Plot the loss curves using `matplotlib` and accuracy curves.
  * Train a new `TinyVGG` model, this time using the train_dataloader_augmented.
  * Compare the model losses and accuracies using plots and pandas DataFrame.
  * Download a custom image from the website, and use it to obtain a prediction.
- PyTorch Custom Dataset Excercises (**Completed**)
  * Contains solutions from: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/04_pytorch_custom_datasets_exercises.ipynb

### 05_PyTorch_going_modular (Completed)
- PyTorch Going Modular:
  * Obtain `get_data.py` using `%%writefile src/get_data.py` in order to download the pizza, steak, and sushi dataset from: https://github.com/mrdbourke/PyTorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip
  * Obtain `data_setup.py` using `%%writefile src/data_setup.py` in order to create the dataloaders.
  * Copying from *04_PyTorch_custom_dataset*, we retreive the `TinyVGG` model class and write a function using `%%writefile src/model_builder.py`
  * Obtain `engine.py` using `%%writefile srx/engine.py` where it contains the `train_step`, `test_step` and `train` cells for this pizza, steak, and sushi image classification task.
  * Obtain `utils.py` using `%%writefile src/utils.py` in order yo save `save_model()` function using `pathlib` and `torch.save(obk = model.state_dict(), f = model_save_path)`
  * Obtain the `train.py` function, with added parser arguments for:
    - String train directory.
    - String test directory.
    - Float learning rate.
    - Integer batch size.
    - Integer number of epochs.
    - Ineger number of hidden units.
    At the end, use `utils.py` to save the model.
  * Obtain `predict.py` using `%%writefile src/predict.py` with an added parsed argument called `--image`. A string directory path for a test image.
- Pytorch Going Modular Excercises: (**Completed**)
  * Contains solutions from: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/05_pytorch_going_modular_exercise_template.ipynb

### YouTube series:
The entire knowledge of this learning journey comes from the following link: https://www.freecodecamp.org/news/learn-PyTorch-for-deep-learning-in-day/
Additionally, for **05_Going_Modular**, the last two hours of content come from: https://www.youtube.com/watch?v=LyJtbe__2i0&ab_channel=ZeroToMastery

**P.S:** I am currently following the rest using Zero to Mastery, so from this point onwards, I will upload the classes and excercises pertinent to Zero to Mastery content from Daniel Bourke :) 
