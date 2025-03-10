# deep-learning-challenge

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code and documentation for a deep learning project focused on predicting the success of charities funded by Alphabet Soup. The goal is to build a binary classification model using a neural network (implemented in TensorFlow/Keras) that can accurately predict whether an organization will be successful based on various features provided in a CSV dataset. The project involves data preprocessing, model building, training, evaluation, and optimization. Several iterations of model optimization are explored, culminating in a final, highly-tuned neural network model.  Although considerable effort was put into optimizing the neural network, it *did not* achieve the target accuracy of 75%. This reinforces the recommendation (made in the project report) to use a Gradient Boosted Decision Tree model (like XGBoost) for this type of task. *Note: While an XGBoost model is recommended, it is not implemented in this repository.*

## Table of Contents

*   [Repository Structure](#repository-structure)
*   [Data Source](#data-source)
*   [Getting Started](#getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
*   [Usage](#usage)
    *   [Data Preprocessing](#data-preprocessing)
    *   [Model Training](#model-training)
    *   [Model Evaluation](#model-evaluation)
    *   [Model Saving and Loading](#model-saving-and-loading)
*   [Model Details](#model-details)
  *   [Starter_Code](#starter_code)
  *   [AlphabetSoupCharity_Optimization](#alphabetsoupcharity_optimization)
    *   [Checkpointing](#checkpointing)
  *   [AlphabetSoupCharity_Optimization_Final](#alphabetsoupcharity_optimization_final)
  *  [AlphabetSoupCharity_One_Last_Hurrah](#alphabetsoupcharity_one_last_hurrah)
*   [Results and Analysis](#results-and-analysis)
*   [Recommendations](#recommendations)
*   [License](#license)
*   [Contributing](#contributing)

## Repository Structure

```
deep-learning-challenge/
├── 1_Starter_Code_ipynb_Explanation.pdf
├── 2_AlphabetSoupCharity_Optimization_ipynb_Explanation.pdf
├── 3_AlphabetSoupCharity_Optimization_Final_ipynb_Explanation.pdf
├── 4_Optimization_Progression.pdf
├── 5_Report - Deep Learning Model for Charity Success Prediction.pdf
├── README.md
└── Deep_Learning_Challenge/
    ├── Start_Code.h5
    ├── Starter_Code.ipynb
    ├── AlphabetSoupCharity_Optimization.h5
    ├── AlphabetSoupCharity_Optimization.ipynb
    ├── AlphabetSoupCharity_Optimization_Final.h5
    ├── AlphabetSoupCharity_Optimization_Final.ipynb
    ├── AlphabetSoupCharity_One_Last_Hurrah.h5
    └── AlphabetSoupCharity_One_Last_Hurrah.ipynb
```

*   **Deep_Learning_Challenge/:**  This directory contains the core project files (Jupyter Notebooks and saved models).
    *   **AlphabetSoupCharity_*.ipynb:**  Jupyter Notebooks containing the code for different model optimization attempts.  These files include detailed comments explaining each step of the process.
    *   **AlphabetSoupCharity_*.h5:** Saved Keras models (weights and architecture) from the different optimization attempts.
    *   **Start_Code.ipynb:** The initial Jupyter Notebook provided as a starting point.
    *   **Start_Code.h5:** The saved Keras model from the initial notebook.
    *   **checkpoints/:**  This directory is *created in Google Colab during model training* by the `AlphabetSoupCharity_Optimization.ipynb`,`AlphabetSoupCharity_Optimization_Final.ipynb`, and `AlphabetSoupCharity_One_Last_Hurrah.ipynb` notebooks. It stores model weight checkpoints at regular intervals, allowing you to resume training or revert to a previous state.
*  **[1-5]_*.pdf:** PDF files are located in the main project directory, providing explanations of the notebooks and a final report.  These reports are arranged in the order in which the project work was done.

## Data Source

The project uses the `charity_data.csv` dataset, which is accessed directly from a provided cloud URL within the code:

```python
pd.read_csv("https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv")
```

This dataset contains various features about organizations, including:

*   **EIN:** Identification number (dropped during preprocessing).
*   **NAME:** Organization name (dropped during preprocessing).
*   **APPLICATION_TYPE:** Type of application (binned during preprocessing).
*   **AFFILIATION:** Affiliation type.
*   **CLASSIFICATION:** Classification of the organization (binned during preprocessing).
*   **USE_CASE:** Use case for the funding.
*   **ORGANIZATION:** Type of organization.
*   **STATUS:** Status of the organization (dropped during preprocessing).
*   **INCOME_AMT:** Income bracket of the organization.
*   **SPECIAL_CONSIDERATIONS:** Special consideration flag (dropped during preprocessing).
*   **ASK_AMT:** Amount of funding requested.
*   **IS_SUCCESSFUL:**  Target variable indicating whether the organization was successful (1) or not (0).

## Getting Started

### Prerequisites

*   Python 3.7+
*   TensorFlow 2.x
*   pandas
*   scikit-learn
* Jupyter Notebook or JupyterLab

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/deep-learning-challenge.git
    cd deep-learning-challenge
    ```

    (Replace `YOUR_USERNAME` with your actual GitHub username.)

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required packages:**

    ```bash
    pip install tensorflow pandas scikit-learn jupyter
    ```

## Usage

### Data Preprocessing

The Jupyter Notebooks (`Starter_Code.ipynb`, `AlphabetSoupCharity_Optimization.ipynb`, `AlphabetSoupCharity_Optimization_Final.ipynb`, `AlphabetSoupCharity_One_Last_Hurrah.ipynb`) include comprehensive data preprocessing steps:

1.  **Loading Data:** The `charity_data.csv` dataset is loaded into a pandas DataFrame.
2.  **Dropping Columns:**  The `EIN`, `NAME`, `STATUS`, and `SPECIAL_CONSIDERATIONS` columns are dropped as they are not relevant for prediction or add noise.
3.  **Binning:** Rare categorical values in `APPLICATION_TYPE` and `CLASSIFICATION` are grouped into an "Other" category to reduce dimensionality.  Different cutoff values are explored in the optimization notebooks.
4.  **One-Hot Encoding:** Categorical features are converted into numerical representations using `pd.get_dummies()`.
5.  **Splitting Data:** The data is split into training and testing sets using `train_test_split` from scikit-learn, with `stratify=y` to ensure balanced class representation.
6.  **Scaling:** Numerical features are standardized using `StandardScaler` from scikit-learn.  The scaler is fit *only* on the training data and then used to transform both the training and testing data to prevent data leakage.

### Model Training

Each Jupyter Notebook trains a different neural network model.  The key steps are:

1.  **Model Definition:** A sequential neural network model is defined using `tf.keras.models.Sequential()`.  Different architectures (number of layers, neurons per layer, activation functions) are explored in the optimization notebooks.  Regularization (L2 and Dropout) is added in later optimization attempts.
2.  **Model Compilation:** The model is compiled using `nn.compile()`, specifying the loss function (`binary_crossentropy`), optimizer (`adam`), and metrics (`accuracy`).
3.  **Model Training:** The model is trained using `nn.fit()`, passing in the scaled training data, the number of epochs, and (in the optimization notebooks) a custom callback for saving model weights and/or early stopping.
4.  **Validation Split:** A portion of the training data is used as a validation set (using the `validation_split` argument in `nn.fit()`).

### Model Evaluation

After training, the model is evaluated on the *scaled test data*:

```python
model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test, verbose=2)
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
```

This provides the loss and accuracy on unseen data, giving a measure of the model's generalization performance.

### Model Saving and Loading

The trained Keras models are saved using `nn.save()`:

```python
nn.save("model_name.h5")  # Replace model_name with appropriate name
```

This saves the entire model (architecture, weights, and optimizer state) to an HDF5 file. You can load a saved model using:

```python
import tensorflow as tf
loaded_model = tf.keras.models.load_model("model_name.h5")
```

The optimization notebooks also include a custom callback, `SaveEveryNepochs`, which saves *only the model weights* at specified intervals (every 5 epochs by default) during training.  This creates checkpoints in the `checkpoints/` directory.  These checkpoints are saved with the `.weights.h5` extension. To load weights, you must first reconstruct the *same model architecture*, then load the weights:

```python
import tensorflow as tf

# ... (Define the model architecture exactly as it was when saved) ...
number_input_features = 39  # Example: Adjust based on your one-hot encoding
hidden_nodes_layer1 = 128
hidden_nodes_layer2 = 64
hidden_nodes_layer3 = 32
hidden_nodes_layer4 = 16

model = tf.keras.models.Sequential()

# First hidden layer
model.add(tf.keras.layers.Dense(units=hidden_nodes_layer1,
                                 input_dim=number_input_features, activation="relu",
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.3))

# Second hidden layer
model.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu",
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.3))

# Third hidden layer
model.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="relu",
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.3))

# Fourth Hidden Layer
model.add(tf.keras.layers.Dense(units=hidden_nodes_layer4, activation="relu",
                                 kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(tf.keras.layers.Dropout(0.3))

# Output layer
model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Compile the model (important for loading weights, use same optimizer/loss)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# Load the weights
model.load_weights("checkpoints/weights.XX.weights.h5") # Replace XX with epoch number.
```
**Important:** Loading just weights with `load_weights()` requires that the model architecture be *identical* to the architecture used when the weights were saved.  Loading the entire model with `load_model()` is generally preferred for simplicity and to ensure the optimizer state is restored.

## Model Details

The models evolved in this order: `Starter_Code`, `AlphabetSoupCharity_Optimization`, `AlphabetSoupCharity_Optimization_Final`, and finally `AlphabetSoupCharity_One_Last_Hurrah`.

### Starter_Code

*   **File:** `Starter_Code.ipynb`
*   **Description:** The initial model provided.
*   **Architecture:**
    *   Input Layer: Implicit (number of features after preprocessing).
    *   Hidden Layer 1: 80 neurons, 'relu' activation.
    *   Hidden Layer 2: 30 neurons, 'relu' activation.
    *   Output Layer: 1 neuron, 'sigmoid' activation.
*   **Epochs:** 100
*   **Results:** Loss: ~0.555, Accuracy: ~0.726

### AlphabetSoupCharity_Optimization

*   **File:** `AlphabetSoupCharity_Optimization.ipynb`
*   **Description:**  First optimization attempt.
*   **Changes:**
    *   Dropped `STATUS` and `SPECIAL_CONSIDERATIONS` columns.
    *   Increased binning cutoffs: `APPLICATION_TYPE` (700), `CLASSIFICATION` (1800).
    *   Added a third hidden layer (20 neurons, 'relu' activation).
    *   Increased neurons in the first two hidden layers (100 and 50, respectively).
    *   Increased epochs to 200.
    *   Added a custom `SaveEveryNepochs` callback to save model weights every 5 epochs.
*   **Architecture:**
    *   Input Layer: Implicit
    *   Hidden Layer 1: 100 neurons, 'relu' activation.
    *   Hidden Layer 2: 50 neurons, 'relu' activation.
    *   Hidden Layer 3: 20 neurons, 'relu' activation.
    *   Output Layer: 1 neuron, 'sigmoid' activation.
*   **Epochs:** 200
*   **Results:** Loss: ~0.579, Accuracy: ~0.725

#### Checkpointing

This notebook introduces checkpointing. The custom callback `SaveEveryNepochs` is defined to save the model's weights periodically during training. This allows you to:

*   **Resume Training:** If training is interrupted, you can load the latest checkpoint and continue training from that point.
*   **Experimentation:** You can easily revert to earlier versions of your model if later changes don't improve performance.
*   **Prevent Loss:** Checkpoints act as backups in case of unexpected errors or system crashes.

The checkpoints are saved in the `checkpoints/` directory.

### AlphabetSoupCharity_Optimization_Final

*   **File:** `AlphabetSoupCharity_Optimization_Final.ipynb`
*   **Description:** Second optimization attempt.
*   **Changes:**
    *   Added a fourth hidden layer (10 neurons, 'relu' activation).
    *   Increased neurons in hidden layers (100, 80, 30).
    *   Added `stratify=y` to `train_test_split`.
    *   Added a custom `SaveEveryNepochs` callback to save model weights every 5 epochs.
*   **Architecture:**
    *   Input Layer: Implicit.
    *   Hidden Layer 1: 100 neurons, 'relu' activation.
    *   Hidden Layer 2: 80 neurons, 'relu' activation.
    *   Hidden Layer 3: 30 neurons, 'relu' activation.
    *   Hidden Layer 4: 10 neurons, 'relu' activation.
    *   Output Layer: 1 neuron, 'sigmoid' activation.
*   **Epochs:** 200
* **Results:** Loss: ~0.587, Accuracy: ~0.729

### AlphabetSoupCharity_One_Last_Hurrah
*   **File:** `AlphabetSoupCharity_One_Last_Hurrah.ipynb`
*   **Description:** Final optimization attempt.
*   **Changes:**
    *  Increased Neurons and Added Regularization and Dropout
        * Increased the number of neurons in the hidden layers even further.
        * Added L2 regularization to each hidden layer to combat overfitting.
        * Added Dropout layers after each hidden layer to further reduce overfitting.
    * Added an `EarlyStopping` callback.
*   **Architecture:**
    *   Input Layer: Implicit.
    *   Hidden Layer 1: 128 neurons, 'relu' activation, L2 regularization, Dropout.
    *   Hidden Layer 2: 64 neurons, 'relu' activation, L2 regularization, Dropout.
    *   Hidden Layer 3: 32 neurons, 'relu' activation, L2 regularization, Dropout.
    *   Hidden Layer 4: 16 neurons, 'relu' activation, L2 regularization, Dropout.
    *   Output Layer: 1 neuron, 'sigmoid' activation.
*   **Epochs:** 200 (but `EarlyStopping` will likely stop training earlier)
* **Results:** Loss: ~0.565, Accuracy: ~0.728

## Results and Analysis

The project explores several iterations of a deep learning model for predicting charity success.  Even with significant optimization efforts (including adding layers, neurons, regularization, and dropout), the neural network models *did not* achieve the target accuracy of 75%. The final model, `AlphabetSoupCharity_One_Last_Hurrah`, achieved an accuracy of approximately 72.8% on the test data, which is only a marginal improvement over the initial `Starter_Code` model.

The analysis demonstrates the challenges of applying deep learning to tabular data with a relatively small dataset and a significant number of categorical features.  The high dimensionality introduced by one-hot encoding, combined with the limited data, likely contributed to the model's difficulty in achieving higher accuracy.

## Recommendations

*   **Try XGBoost (or other Gradient Boosted Decision Trees):** The project report strongly recommended using an XGBoost model.  While not implemented in *this* repository, gradient boosted decision trees (GBDTs) like XGBoost, LightGBM, and CatBoost are often highly effective on structured/tabular data and often outperform neural networks, especially with limited data and many categorical features. They also tend to require less hyperparameter tuning.
*   **Feature Engineering:**  Consider more sophisticated feature engineering.  Creating interaction terms or using domain expertise to combine features might improve model performance, regardless of the chosen model type.
*   **Collect More Data:** If possible, gathering more data could significantly benefit any model.
*   **Explore Other Models:**  Besides GBDTs, consider other machine learning models like Random Forests, Support Vector Machines (SVMs), or even a simple Logistic Regression as a baseline.
* **If sticking with Neural Networks (less recommended for this problem):**
    *  **Learning Rate Scheduling:** Implement a learning rate scheduler.
    *  **Batch Size Tuning:**  Experiment with different batch sizes.
    * **Different Optimizers:** Try optimizers other than Adam.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.
