# Pytorch Learning Repository

Welcome to my Pytorch learning repository! Here, I document my progress with notebooks and projects highlighting Pytorch’s capabilities and exploring its fundamentals in machine learning and deep learning.

The repository is organized into sections:

Fundamental Concepts: A series of Jupyter notebooks focused on learning and mastering the foundational elements of TensorFlow. Applied Projects: Practical projects showcasing my application of TensorFlow to solve diverse challenges and real-world problems. This repository is a work in progress. I'll update the README and content as I continue on this journey.

### Fundamental Concepts: 
#### 01_Pytorch_Notebook:
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
- Pytorch fundamentals solutions/extra-curriculum (**in progress**):
  * Contains solutions of exercises from: https://www.learnpytorch.io/00_pytorch_fundamentals/#exercises
  * Contains solutions of extra-curriculum from: https://www.learnpytorch.io/00_pytorch_fundamentals/#extra-curriculum
### 02_Pytorch_Workflow:
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
- Pytorch Workflow solutions (**in progress**):
  * Contains solutions of exercises from: https://github.com/mrdbourke/pytorch-deep-learning/blob/main/extras/exercises/01_pytorch_workflow_exercises.ipynb
### Projects

### Youtube series:
The entire knowledge of this learning journey comes from the following link: https://www.freecodecamp.org/news/learn-pytorch-for-deep-learning-in-day/
