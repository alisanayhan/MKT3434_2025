# Enhanced Machine Learning Course GUI

This application provides an interactive graphical user interface (GUI) for exploring various machine learning concepts and algorithms. It's designed for educational purposes, allowing users to load data, preprocess it, train different models, and visualize the results. This version includes significant enhancements over the base version.

## New Features in This Version

This enhanced version introduces several new capabilities:

1.  **Missing Value Handling:**
    *   Added options in the "Data Management" section to handle missing (`NaN`) values *before* splitting and scaling.
    *   Methods: Mean Imputation, Median Imputation, Forward Fill (ffill), Backward Fill (bfill), Drop Rows with NaNs.

2.  **Support Vector Regression (SVR):**
    *   Added SVR as a regression algorithm in the "Classical ML" tab.
    *   Configurable hyperparameters:
        *   `Kernel`: linear, rbf, poly
        *   `C`: Regularization parameter.
        *   `Epsilon`: Epsilon-tube within which no penalty is associated in the training loss function.
        *   `Degree`: Degree of the polynomial kernel function ('poly'). Ignored by other kernels.

3.  **Enhanced Support Vector Classification (SVC) Parameters:**
    *   SVC parameter configuration now includes:
        *   `Kernel`: linear, rbf, poly, sigmoid
        *   `C`: Regularization parameter.
        *   `Degree`: Degree for the 'poly' kernel.
        *   `Gamma`: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. ('scale' or 'auto').

4.  **Gaussian Naive Bayes (GaussianNB):**
    *   Added GaussianNB classifier to the "Classical ML" tab.
    *   Configurable hyperparameter:
        *   `Var Smoothing`: Portion of the largest variance of all features that is added to variances for calculation stability.

5.  **Deep Learning - Task-Specific Loss Functions:**
    *   Added a "Task Type" selector (Classification/Regression) in the "Deep Learning" tab.
    *   The "Loss Function" dropdown now dynamically updates based on the selected task type.
        *   **Regression Losses:** Mean Squared Error (MSE), Mean Absolute Error (MAE), Huber Loss, etc.
        *   **Classification Losses:** Categorical Cross-Entropy, Sparse Categorical Cross-Entropy, Binary Cross-Entropy, Hinge, Squared Hinge, etc.
    *   The neural network output layer and evaluation metrics are automatically adjusted based on the selected task type and loss.

6.  **Refined Data Processing Workflow:**
    *   The data processing steps are now more explicit:
        1.  Load Data (Built-in or Custom CSV).
        2.  Handle Missing Values (Applied when "Process and Split Data" is clicked).
        3.  Split Data into Train/Test sets.
        4.  Scale Features (Fit on Train, Transform Train & Test).

7.  **Updated Visualization and Metrics:**
    *   The visualization and metrics panels are updated to correctly display results and relevant metrics for the newly added models (SVR, GaussianNB) and deep learning tasks (regression/classification with different losses).

## How to Use the New Features

1.  **Loading Data:**
    *   Select a built-in dataset from the "Dataset" dropdown or click "Load Custom CSV" to load your own data.
    *   If loading custom data, you will be prompted to select the target column.

2.  **Handling Missing Values (New):**
    *   Before splitting, select a method from the **"Missing Values"** dropdown in the "Data Management" section (e.g., "Mean Imputation", "Drop Rows").
    *   This method will be applied when you click the **"Process and Split Data"** button.

3.  **Processing and Splitting:**
    *   Adjust the "Test Split" ratio if desired.
    *   Select a "Scaling" method (applied *after* splitting).
    *   Click the **"Process and Split Data"** button. This performs: Missing Value Handling -> Train/Test Split -> Scaling. The status bar will show progress.

4.  **Training Classical Models:**
    *   Navigate to the **"Classical ML"** tab.
    *   **SVR (New):** Find the "SVR" group under "Regression". Adjust Kernel, C, Epsilon, Degree as needed. Click "Train SVR".
    *   **GaussianNB (New):** Find the "GaussianNB" group under "Classification". Adjust "Var Smoothing". Click "Train GaussianNB".
    *   **SVC (Enhanced Params):** Find the "SVC" group. Adjust C, Kernel, Degree, Gamma. Click "Train SVC".
    *   For other classical models, configure parameters in their respective boxes and click their "Train..." button.

5.  **Training Deep Learning Models (New Loss Selection):**
    *   Navigate to the **"Deep Learning"** tab.
    *   Configure the network architecture using "Add Layer" and "Clear All Layers".
    *   **Select Task Type (New):** Choose "Classification" or "Regression" from the "Task Type" dropdown.
    *   **Select Loss Function (New):** Choose a loss function from the "Loss Function" dropdown. The available options depend on the selected Task Type.
    *   Configure training parameters (Batch Size, Epochs, Learning Rate).
    *   Click **"Train Neural Network"**.

6.  **Viewing Results:**
    *   After training any model, the **"Results: Visualization and Metrics"** panel will update.
    *   The left side shows a plot relevant to the task (e.g., Actual vs. Predicted for regression, PCA/scatter plot for classification/clustering, training history for NNs).
    *   The right side displays relevant performance metrics (e.g., MSE/MAE/R² for regression, Accuracy/Confusion Matrix for classification).

## Dependencies

Make sure you have the following libraries installed:

*   PyQt6
*   NumPy
*   Pandas
*   Matplotlib
*   Scikit-learn
*   TensorFlow (includes Keras)

You can install them using pip:

```bash
pip install PyQt6 numpy pandas matplotlib scikit-learn tensorflow
