# Enhanced Machine Learning Course GUI

This application provides an interactive graphical user interface (GUI) for exploring various machine learning concepts and algorithms. It's designed for educational purposes, allowing users to load data, preprocess it, train different models, visualize the results, and evaluate model performance using techniques like cross-validation.

## Features

This GUI includes a range of functionalities covering the typical machine learning workflow:

1.  **Data Management:**
    *   Load built-in datasets (Iris, Breast Cancer, Digits, California Housing).
    *   Load custom CSV datasets with user-selectable target columns.
    *   **Missing Value Handling:** Options to handle `NaN` values before splitting (Mean/Median Imputation, Forward/Backward Fill, Drop Rows).
    *   **Feature Scaling:** Apply Standard, Min-Max, or Robust scaling after splitting.
    *   **Train/Test Split:** Configure the test set size ratio.

2.  **Supervised Learning Tab:**
    *   **Regression Algorithms:**
        *   Linear Regression
        *   Support Vector Regression (SVR) with configurable kernel, C, epsilon, degree.
        *   Decision Tree Regressor with configurable criterion, depth, split size.
        *   Random Forest Regressor with configurable estimators, criterion, depth, split size.
        *   K-Neighbors Regressor with configurable neighbors, weights, metric.
    *   **Classification Algorithms:**
        *   Logistic Regression with configurable C, max\_iter, multi\_class, solver.
        *   Gaussian Naive Bayes (GaussianNB) with configurable variance smoothing.
        *   Support Vector Classification (SVC) with configurable C, kernel, degree, gamma, probability.
        *   Decision Tree Classifier with configurable criterion, depth, split size.
        *   Random Forest Classifier with configurable estimators, criterion, depth, split size.
        *   K-Neighbors Classifier with configurable neighbors, weights, metric.

3.  **Unsupervised & Dimensionality Reduction Tab:**
    *   **Clustering:**
        *   **K-Means:** Configurable clusters (k), init method, n\_init, max\_iter.
        *   **Elbow Method:** Button to plot inertia vs. k to help choose the optimal number of clusters.
        *   **Silhouette Score:** Automatically calculated and displayed for K-Means results.
    *   **Dimensionality Reduction:**
        *   **PCA (Principal Component Analysis):** User-selectable number of components, optional whitening. Explained variance ratio plot and metrics.
        *   **LDA (Linear Discriminant Analysis):** Supervised reduction. User-selectable components, solver choice. Embedding plot (colored by true labels) and metrics (explained variance, Silhouette score on embedding).
        *   **t-SNE (t-distributed Stochastic Neighbor Embedding):** For visualization. Configurable components, perplexity, learning rate, iterations, initialization. Embedding plot.
        *   **UMAP (Uniform Manifold Approximation and Projection):** For visualization (requires `umap-learn`). Configurable components, neighbors, minimum distance. Embedding plot.

4.  **Deep Learning Tab:**
    *   Build custom neural network architectures layer by layer (Dense, Dropout, Conv2D, MaxPooling2D, Flatten, LSTM, GRU).
    *   **Task-Specific Configuration:** Select "Classification" or "Regression" task type.
    *   **Dynamic Loss Functions:** Loss function dropdown updates based on the task type (e.g., MSE/MAE for regression, Cross-Entropy/Hinge for classification).
    *   Configure training parameters (Batch Size, Epochs, Learning Rate).
    *   Training progress visualization (loss/metric curves).

5.  **Model Evaluation:**
    *   **Train/Test Evaluation:** Metrics (Accuracy, Confusion Matrix, Precision/Recall/F1 for classification; MSE, MAE, RMSE, R² for regression) and visualizations are displayed after training/applying a model on the test set.
    *   **k-Fold Cross-Validation:** Dedicated section to run k-fold CV (user-selectable `k`) on the *last trained supervised model*. Reports mean and standard deviation for relevant metrics (Accuracy, F1, MSE, R², etc.) across folds and visualizes results.

6.  **Visualization:**
    *   Dynamic plots based on the task:
        *   Actual vs. Predicted (Regression)
        *   Scatter plots of data/embeddings (Classification, Clustering, Dim Reduction - PCA/LDA/t-SNE/UMAP), potentially colored by labels/clusters. 2D and basic 3D plots supported.
        *   Explained Variance plot (PCA, LDA).
        *   Elbow Method plot (K-Means).
        *   Training History (Deep Learning).
        *   Cross-Validation Score Distribution plot.

7.  **Examples:**
    *   Button to show the calculation of eigenvectors for 1D projection using a sample covariance matrix.

## How to Use

1.  **Load Data:**
    *   Select a built-in dataset or load a custom CSV via the buttons in "Data Management".
    *   If loading custom data, select the target column when prompted.

2.  **Preprocess Data:**
    *   Choose a "Missing Values" handling strategy (optional).
    *   Select a "Scaling" method (optional).
    *   Adjust the "Test Split Ratio".
    *   Click **"Process and Split Data"**. This performs: Missing Value Handling -> Train/Test Split -> Scaling. Check the status bar for progress.

3.  **Train/Apply Models:**
    *   Navigate to the relevant tab ("Supervised Learning", "Unsupervised & Dim Reduction", "Deep Learning").
    *   Select the desired algorithm/model type.
    *   Configure the hyperparameters in the model's specific group box.
    *   Click the **"Train [Model Name]"** button (for supervised models) or **"Apply [Model Name]"** button (for unsupervised/dim reduction models).
    *   *For K-Means Elbow Plot:* Set the "Max k for Elbow" and click "Plot Elbow Method" *before* or *after* applying K-Means for a specific `k`.
    *   *For LDA:* Ensure data has been processed *with* a target variable (`y`). LDA requires `y` for fitting.

4.  **Run Cross-Validation (Supervised Models Only):**
    *   **First, train a model** from the "Supervised Learning" tab (e.g., click "Train Logistic Regression"). This sets the model type and parameters to be used for CV.
    *   Go to the **"Model Evaluation: Cross-Validation"** section (below the tabs).
    *   Set the desired "Number of Folds (k)".
    *   Click **"Run k-Fold Cross-Validation"**. Results (metrics and plot) will appear in the Results panel.

5.  **Train Deep Learning Models:**
    *   Go to the "Deep Learning" tab.
    *   Add layers to define the architecture.
    *   Select the "Task Type" and "Loss Function".
    *   Set Epochs, Batch Size, Learning Rate.
    *   Click **"Train Neural Network"**. Training history will be plotted.

6.  **View Results:**
    *   After training/applying any model or running CV, the **"Results: Visualization and Metrics"** panel updates.
    *   The plot area shows visualizations relevant to the last action (e.g., predictions, embeddings, elbow plot, CV scores, NN history).
    *   The text area displays relevant performance metrics.

7.  **Eigenvector Example:**
    *   Click the "Show Eigenvector Example (1D Projection)" button in the "Examples & Tools" section to see the calculation in a pop-up window.

## Dependencies

Make sure you have the following libraries installed:

*   PyQt6
*   NumPy
*   Pandas
*   Matplotlib
*   Scikit-learn (`sklearn`)
*   TensorFlow (includes Keras)
*   umap-learn (Optional, for UMAP functionality)

You can typically install them using pip:

```bash
# Install core dependencies
pip install PyQt6 numpy pandas matplotlib scikit-learn tensorflow

# Install optional UMAP dependency
pip install umap-learn
