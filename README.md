# Pytorch Learning Repository

Welcome to my Pytorch learning repository! Here, I document my progress with notebooks and projects highlighting Pytorch’s capabilities and exploring its fundamentals in machine learning and deep learning.

The repository is organized into sections:

Fundamental Concepts: A series of Jupyter notebooks focused on learning and mastering the foundational elements of TensorFlow. Applied Projects: Practical projects showcasing my application of TensorFlow to solve diverse challenges and real-world problems. This repository is a work in progress. I'll update the README and content as I continue on this journey.

### Fundamental Concepts: 
#### 00_Pytorch_Notebook:
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
- Pytorch fundamentals solutions/extra-curriculum (**Complete**):
  * Contains solutions of exercises from: https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises
  * Contains solutions of extra-curriculum from: https://www.learnpytorch.io/00_pytorch_fundamentals/#extra-curriculum
### 01_Pytorch_Workflow:
- PyTorch Workflow content:
  * Data preparation and loading in tensors formats.
  * Splitting data into training and testing splits and plotting them.
  * Building Model with random values (Hand made single-value linear regression).
  * Understanding Pytorch essentials: `torch.nn`, `torch.nn.Parameter`, `torch.nn.Module`,
    `torch.optim`, overwriting `forward()`
  * Using `torch.inference_mode()` on an untrained model and see its predictions.
  * Fitting a model to training data.
  * Making predictions and evaluating it (inference).
  * Saving a loading model.
  * Putting all together.
- Pytorch Workflow solutions (**Complete**):
  * Contains solutions of exercises from: https://www.learnpytorch.io/01_pytorch_workflow/#exercises
### 02_Pytorch_Classification_Workflow
- Pytorch Classification Content:
  * Made Classification data using `make_circles()` from `sklearn.datasets` for two circles.
  * Turn data into `torch.Tensor()` and split them into training, validation, and test data using `train_test_split()`.
  * Build a model with device agnostic code using `nn.Linear()` and `nn.Module()`; We compared too against `nn.Sequential()`.
  * TO POINT OUT, this model didn't have any non-linear activation functions, and we would see how this doesn't work with non-linear data.
  * Setup an optimizer for binary classification, we used `nn.BCEWithLogitsLoss()` and `torch.optim.SGD()`
  * Define your own accuracy function and train the model
  * Define properly how to go from raw logits (output by model) -> prediction probabilities (activation function `torch.sigmoid()`) -> prediction labels (thresholding at 0.5).
  * Setup a training and testing loop to test out this model.
  * Plot the boundary lines and get it from helper functions: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py
  * Improving the model (adding more hiddent units, number of layers, and number of epochs). This is not gonna work :)
  * Retrain the test this new model, visualize why even after adding much more power, it doesnt work with non-linear data.
  * Troubleshoot our model by creating regression linear data.
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
* Train this model and visualize it! Ir works. Last thing is to repeat this multiclassification model without using non-linear activiation functions
* Realize that deep learning models without activation functions would still work for linear separable data.
* Dive into some classification metrics.
- Pytorch Classification solutions/extra-curriculum (**In progress**):
  * Contains solutions of exercises from: https://www.learnpytorch.io/02_pytorch_classification/#extra-curriculum

### Youtube series:
The entire knowledge of this learning journey comes from the following link: https://www.freecodecamp.org/news/learn-pytorch-for-deep-learning-in-day/
