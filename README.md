# odysseus
Using AI&amp;ML to read Ancient Greek letters

## Environment Setup

To ensure reproducibility across macOS and Windows, we use conda to manage our dependencies. We also use Weights & Biases (wandb) for experiment tracking and hyperparameter logging.

### 1. Install Conda

If you do not have Conda installed, please install Miniconda or Anaconda.

### 2. Create the Environment

Open your terminal (macOS) or Anaconda Prompt (Windows) and run:

```bash
conda env create -f environment.yml
```

### 3. Activate the Environment

```bash
conda activate odysseus
```
(In VSCode you can now select the "odysseus" conda environment as a Python interpreter)

### 4. Weights & Biases Authentication

Since we use wandb to monitor our training runs and dataset distributions, you must link your local environment to your W&B account. Run the following command and paste your API key when prompted:

```bash
wandb login
```

### 5. Register the kernel with Jupyter

```bash
python -m ipykernel install --user --name=odysseus --display-name "Python 3.11 (odysseus)"
```

### 6. Launch Jupyter

You can now launch the interactive notebook environment:

```bash
jupyter notebook
```

**IMPORTANT**: select the *Python 3.11 (odysseus)* as the kernel!


## Proposed Project Structure

- `data/`: where the dataset will reside (added to gitignore)
- `notebooks/`:
  - `00_download_kaggle_dataset.ipynb`: downloads the kaggle dataset to `../data`
  - `01_preprocessing.ipynb`: image cleaning and standardization
  - `02_feature_extraction.ipynb`: VGG-16 forward passes and vector storage
  - `03_modeling.ipynb`: Training SVM, k-NN, etc., and logging to `wandb`
  - `04_evaluation.ipynb`: Final testing and confusion matrix generation
- `environment.yml`: Shared dependencies