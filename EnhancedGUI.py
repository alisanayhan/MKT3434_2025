import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QTabWidget, QPushButton, QLabel,
                           QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                           QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                           QDialog, QLineEdit)
from PyQt6.QtCore import Qt, pyqtSlot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn import datasets, preprocessing, model_selection
from sklearn.impute import SimpleImputer # For Mean/Median Imputation
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR # Added SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, confusion_matrix, r2_score # Added MAE, R2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses # Added losses

# Helper function to safely convert string to tuple of ints
def parse_int_tuple(text, default=(3, 3)):
    try:
        parts = [int(p.strip()) for p in text.split(',')]
        if len(parts) >= 2:
            return tuple(parts[:2]) # Ensure it's a tuple of 2 ints for kernel_size
        else:
            return default
    except ValueError:
        return default


class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI (Enhanced)")
        self.setGeometry(100, 100, 1500, 850) # Increased size slightly

        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Initialize data containers
        self.original_data = None # Store original loaded data before processing
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        self.current_model_name = None # To know which model was trained

        # Neural network configuration
        self.layer_config = []
        self.nn_task_type = 'Classification' # Default DL task type

        # Create components
        self.create_data_section()
        self.create_tabs()
        self.create_visualization()
        self.create_status_bar()

    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        data_layout = QGridLayout() # Use GridLayout for better alignment

        # Row 0: Dataset selection and Load Button
        data_layout.addWidget(QLabel("Dataset:"), 0, 0)
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "Boston Housing Dataset (Regression)", # Explicitly mention type
            # "MNIST Dataset" # MNIST requires more specific handling (flattening, CNN shapes) - Keep commented for now
        ])
        # Connect after initial setup to avoid premature loading
        self.dataset_combo.currentIndexChanged.connect(self._dataset_selection_changed)
        data_layout.addWidget(self.dataset_combo, 0, 1)

        self.load_btn = QPushButton("Load Custom CSV")
        self.load_btn.clicked.connect(self.load_custom_data)
        data_layout.addWidget(self.load_btn, 0, 2)

        # Row 1: Preprocessing options
        data_layout.addWidget(QLabel("Missing Values:"), 1, 0)
        self.missing_values_combo = QComboBox()
        self.missing_values_combo.addItems([
            "No Handling",
            "Mean Imputation",
            "Median Imputation",
            # "Interpolate", # Interpolation needs careful axis handling
            "Forward Fill",
            "Backward Fill",
            "Drop Rows" # Added option to drop rows with NaNs
        ])
        data_layout.addWidget(self.missing_values_combo, 1, 1)

        data_layout.addWidget(QLabel("Scaling:"), 1, 2)
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ])
        data_layout.addWidget(self.scaling_combo, 1, 3)

        data_layout.addWidget(QLabel("Test Split:"), 1, 4)
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)
        data_layout.addWidget(self.split_spin, 1, 5)

        # Row 2: Process Data Button
        self.process_data_btn = QPushButton("Process and Split Data")
        self.process_data_btn.clicked.connect(self.process_and_split_data)
        data_layout.addWidget(self.process_data_btn, 2, 0, 1, 6) # Span across columns

        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)

    @pyqtSlot()
    def _dataset_selection_changed(self):
        """Handle built-in dataset selection."""
        if self.dataset_combo.currentText() != "Load Custom Dataset":
            self.load_builtin_dataset()

    def load_builtin_dataset(self):
        """Load selected built-in dataset"""
        try:
            dataset_name = self.dataset_combo.currentText()
            self.status_bar.showMessage(f"Loading {dataset_name}...")
            QApplication.processEvents() # Update UI

            if dataset_name == "Iris Dataset":
                data = datasets.load_iris(as_frame=True)
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer(as_frame=True)
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits(as_frame=True)
            elif dataset_name == "Boston Housing Dataset (Regression)":
                 # Boston dataset removed from newer sklearn. Using California housing as replacement.
                try:
                    data = datasets.fetch_california_housing(as_frame=True)
                    # California housing has 'MedHouseVal' as target
                    data.target = data.frame['MedHouseVal']
                    data.data = data.frame.drop('MedHouseVal', axis=1)
                    dataset_name = "California Housing Dataset (Regression)" # Update name
                except ImportError:
                     self.show_error("California Housing dataset requires scikit-learn >= 0.22. "
                                     "Please update scikit-learn or choose another dataset.")
                     self.status_bar.showMessage("Dataset loading failed.")
                     return
                except Exception as fetch_error: # Catch other potential fetch errors
                    self.show_error(f"Failed to load California Housing: {fetch_error}")
                    self.status_bar.showMessage("Dataset loading failed.")
                    return
            # elif dataset_name == "MNIST Dataset": # Requires specific reshaping/handling
            #     (X_train_mnist, y_train_mnist), (X_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()
            #     # Flatten MNIST images for classical models / initial processing
            #     X_mnist = np.vstack((X_train_mnist, X_test_mnist)).reshape(-1, 28*28)
            #     y_mnist = np.hstack((y_train_mnist, y_test_mnist))
            #     self.original_data = pd.DataFrame(X_mnist)
            #     self.original_data['target'] = y_mnist
            #     self.status_bar.showMessage(f"Loaded {dataset_name}. Requires specific NN layers (Flatten, Conv2D).")
            #     self.X = self.original_data.drop('target', axis=1)
            #     self.y = self.original_data['target']
            #     # For MNIST, usually use the predefined split, but allow reprocessing here
            #     self.process_and_split_data() # Trigger processing
            #     return # Skip standard processing below
            else:
                self.show_error(f"Dataset '{dataset_name}' not implemented yet.")
                self.status_bar.showMessage("Dataset loading failed.")
                return

            # Store original data (features + target)
            self.original_data = data.frame # Use the frame directly
            self.status_bar.showMessage(f"Loaded {dataset_name}. Click 'Process and Split Data'.")
            # Clear previous results
            self.clear_results()

        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")
            self.status_bar.showMessage("Dataset loading failed.")
            self.original_data = None
            self.clear_data()

    def load_custom_data(self):
        """Load custom dataset from CSV file"""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Custom CSV Dataset",
                "",
                "CSV files (*.csv)"
            )

            if file_name:
                self.status_bar.showMessage(f"Loading custom data from {file_name}...")
                QApplication.processEvents()
                # Load data
                data = pd.read_csv(file_name)

                # Ask user to select target column
                target_col = self.select_target_column(data.columns)

                if target_col:
                    self.original_data = data
                    self.target_column_name = target_col # Store the target column name
                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}. Click 'Process and Split Data'.")
                    # Clear previous results
                    self.clear_results()
                else:
                    self.status_bar.showMessage("Custom data loading cancelled (no target selected).")
                    self.original_data = None

        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")
            self.status_bar.showMessage("Custom data loading failed.")
            self.original_data = None
            self.clear_data()

    def select_target_column(self, columns):
        """Dialog to select target column from dataset"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)

        label = QLabel("Please select the target variable (y) column:")
        layout.addWidget(label)

        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)

        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("Select")
        btn_ok.clicked.connect(dialog.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(dialog.reject)
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None

    def handle_missing_values(self, data):
        """Apply selected missing value handling method"""
        method = self.missing_values_combo.currentText()
        self.status_bar.showMessage(f"Applying {method}...")
        QApplication.processEvents()
        original_shape = data.shape

        try:
            numeric_cols = data.select_dtypes(include=np.number).columns
            non_numeric_cols = data.select_dtypes(exclude=np.number).columns

            if method == "Mean Imputation":
                if len(numeric_cols) > 0:
                    imputer = SimpleImputer(strategy='mean')
                    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
                else:
                     self.status_bar.showMessage("Mean Imputation skipped: No numeric columns found.", 5000)
            elif method == "Median Imputation":
                 if len(numeric_cols) > 0:
                    imputer = SimpleImputer(strategy='median')
                    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
                 else:
                     self.status_bar.showMessage("Median Imputation skipped: No numeric columns found.", 5000)
            # elif method == "Interpolate": # Needs more complex handling (method, axis)
            #     if len(numeric_cols) > 0:
            #         data[numeric_cols] = data[numeric_cols].interpolate(method='linear', axis=0)
            #     else:
            #          self.status_bar.showMessage("Interpolation skipped: No numeric columns found.", 5000)
            elif method == "Forward Fill":
                data.ffill(inplace=True)
            elif method == "Backward Fill":
                data.bfill(inplace=True)
            elif method == "Drop Rows":
                data.dropna(inplace=True)

            # Check if any NaNs remain after handling (except 'No Handling')
            if method != "No Handling" and data.isnull().sum().sum() > 0:
                self.show_error(f"Warning: NaNs may still remain after {method}. Check data or try dropping rows/columns.")

            self.status_bar.showMessage(f"Applied {method}. Shape change: {original_shape} -> {data.shape}", 5000)
            return data

        except Exception as e:
            self.show_error(f"Error during missing value handling ({method}): {str(e)}")
            self.status_bar.showMessage(f"Missing value handling failed for {method}.")
            return None # Indicate failure


    def process_and_split_data(self):
        """Handle missing values, split data, and apply scaling."""
        if self.original_data is None:
            self.show_error("No data loaded. Please load a dataset first.")
            return

        try:
            self.status_bar.showMessage("Processing data...")
            QApplication.processEvents()

            data_processed = self.original_data.copy()

            # 1. Handle Missing Values
            data_processed = self.handle_missing_values(data_processed)
            if data_processed is None: # Handle failure in imputation
                self.clear_data()
                return

            # 2. Identify Features (X) and Target (y)
            if hasattr(self, 'target_column_name'): # Custom data
                 if self.target_column_name not in data_processed.columns:
                      self.show_error(f"Target column '{self.target_column_name}' not found after preprocessing. It might have been dropped.")
                      self.clear_data()
                      return
                 self.X = data_processed.drop(self.target_column_name, axis=1)
                 self.y = data_processed[self.target_column_name]
            else: # Built-in dataset (assuming 'target' column if frame was used, or separate data/target)
                if isinstance(self.original_data, pd.DataFrame) and 'target' in self.original_data.columns:
                     if 'target' not in data_processed.columns:
                        # This case might happen if target was in original_data but dropped during imputation (unlikely but possible)
                         self.show_error("Target column 'target' not found after preprocessing.")
                         self.clear_data()
                         return
                     self.X = data_processed.drop('target', axis=1)
                     self.y = data_processed['target']
                elif hasattr(self.original_data, 'data') and hasattr(self.original_data, 'target'):
                     # Fallback for datasets not loaded as frames initially, or if frame structure is lost
                     # Need to re-align X and y if rows were dropped during imputation
                     self.X = pd.DataFrame(self.original_data.data, index=data_processed.index) # Use index from processed data
                     self.y = pd.Series(self.original_data.target, index=data_processed.index) # Use index from processed data
                     self.X = self.X.loc[data_processed.index] # Ensure alignment if rows were dropped
                     self.y = self.y.loc[data_processed.index]
                else:
                     self.show_error("Cannot automatically determine features (X) and target (y).")
                     self.clear_data()
                     return


            # --- Handle potential non-numeric features ---
            numeric_cols = self.X.select_dtypes(include=np.number).columns
            non_numeric_cols = self.X.select_dtypes(exclude=np.number).columns

            if len(non_numeric_cols) > 0:
                 # Option 1: Try One-Hot Encoding (Simple approach)
                try:
                    self.status_bar.showMessage(f"Non-numeric columns found: {list(non_numeric_cols)}. Attempting One-Hot Encoding...")
                    QApplication.processEvents()
                    self.X = pd.get_dummies(self.X, columns=non_numeric_cols, drop_first=True) # drop_first to avoid multicollinearity
                    self.status_bar.showMessage("One-Hot Encoding applied.", 3000)
                except Exception as encode_err:
                     self.show_error(f"Failed to automatically One-Hot Encode non-numeric features: {encode_err}. "
                                     "Please preprocess manually or remove these columns: {list(non_numeric_cols)}")
                     self.clear_data()
                     return
                # Option 2: Show error and stop (Safer for complex cases)
                # self.show_error(f"Non-numeric feature columns detected: {list(non_numeric_cols)}. "
                #                 "This GUI currently requires numeric features. Please preprocess your data (e.g., One-Hot Encode).")
                # self.clear_data()
                # return

            # 3. Split Data
            test_size = self.split_spin.value()
            self.status_bar.showMessage(f"Splitting data (Test size: {test_size})...")
            QApplication.processEvents()
            try:
                 # Check if stratification is needed (for classification tasks)
                 is_classification = self.y.nunique() < 20 and self.y.dtype != 'float' # Heuristic for classification
                 stratify_param = self.y if is_classification else None

                 self.X_train, self.X_test, self.y_train, self.y_test = \
                    model_selection.train_test_split(self.X, self.y,
                                                  test_size=test_size,
                                                  random_state=42,
                                                  stratify=stratify_param) # Add stratification
            except ValueError as split_err:
                 if "stratify" in str(split_err):
                     self.status_bar.showMessage("Stratification failed (e.g., too few samples per class). Splitting without stratification.", 5000)
                     self.X_train, self.X_test, self.y_train, self.y_test = \
                        model_selection.train_test_split(self.X, self.y,
                                                      test_size=test_size,
                                                      random_state=42)
                 else:
                     raise split_err # Re-raise other splitting errors


            # 4. Apply Scaling (Fit on Train, Transform on Train & Test)
            self.apply_scaling() # This method now operates on self.X_train/self.X_test

            self.status_bar.showMessage(f"Data processed successfully. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
            self.clear_results() # Clear previous model results

        except Exception as e:
            self.show_error(f"Error processing data: {str(e)}")
            self.status_bar.showMessage("Data processing failed.")
            self.clear_data() # Clear all data state on error

    def apply_scaling(self):
        """Apply selected scaling method to the train/test data"""
        if self.X_train is None or self.X_test is None:
            # This shouldn't happen if called from process_and_split_data, but good safeguard
            self.status_bar.showMessage("Cannot apply scaling: Data not split yet.", 5000)
            return

        scaling_method = self.scaling_combo.currentText()
        self.status_bar.showMessage(f"Applying {scaling_method}...")
        QApplication.processEvents()

        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                else: # Should not happen
                    return

                # Fit scaler ONLY on training data
                self.X_train = scaler.fit_transform(self.X_train)
                # Transform BOTH training and test data
                self.X_test = scaler.transform(self.X_test)

                self.status_bar.showMessage(f"Applied {scaling_method} successfully.", 3000)

            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")
                self.status_bar.showMessage(f"Scaling failed for {scaling_method}.")
                # Optionally revert to unscaled data or stop? For now, just report error.


    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()

        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Clustering & Dim Reduction", self.create_cluster_dim_reduction_tab), # Renamed tab
            # ("Reinforcement Learning", self.create_rl_tab) # Keep RL simpler for now
        ]

        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)

        self.layout.addWidget(self.tab_widget)

    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""
        widget = QWidget()
        layout = QGridLayout(widget) # Use Grid for 2 columns

        # --- Regression section ---
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()

        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {"fit_intercept": ("checkbox", True)}, # Add default value
             # "normalize" is deprecated, scaling is handled separately
        )
        regression_layout.addWidget(lr_group)

        # SVR (Support Vector Regression) - NEW
        svr_group = self.create_algorithm_group(
             "SVR", # Short name for button
             {"kernel": (["linear", "rbf", "poly"], "rbf"), # Default kernel
              "C": ("double", 1.0, 0.01, 1000.0, 0.1),  # Type, Default, Min, Max, Step
              "epsilon": ("double", 0.1, 0.0, 10.0, 0.1),
              "degree": ("int", 3, 2, 10, 1)} # Relevant for 'poly' kernel
        )
        regression_layout.addWidget(svr_group)


        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0, Qt.AlignmentFlag.AlignTop) # Align group to top

        # --- Classification section ---
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()

        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": ("double", 1.0, 0.01, 1000.0, 0.1),
             "max_iter": ("int", 100, 50, 10000, 50),
             "multi_class": (["ovr", "multinomial"], "ovr"),
             "solver": (["liblinear", "lbfgs", "saga"], "lbfgs")} # Added solver
        )
        classification_layout.addWidget(logistic_group)

        # Naive Bayes (GaussianNB) - NEW
        nb_group = self.create_algorithm_group(
            "GaussianNB",
            {"var_smoothing": ("double", 1e-9, 1e-12, 1e-3, 1e-9)} # Log scale might be better here?
        )
        classification_layout.addWidget(nb_group)

        # SVC (Support Vector Classification) - Enhanced
        svm_group = self.create_algorithm_group(
            "SVC", # Short name for button
            {"C": ("double", 1.0, 0.01, 1000.0, 0.1),
             "kernel": (["linear", "rbf", "poly", "sigmoid"], "rbf"),
             "degree": ("int", 3, 2, 10, 1), # Relevant for 'poly' kernel
             "gamma": (["scale", "auto"], "scale")} # Gamma for RBF, Poly, Sigmoid
        )
        classification_layout.addWidget(svm_group)

        # Decision Trees
        dt_group = self.create_algorithm_group(
            "Decision Tree",
            {"criterion": (["gini", "entropy"], "gini"),
             "max_depth": ("int_optional", 5, 1, 100, 1), # Allow None (no limit)
             "min_samples_split": ("int", 2, 2, 100, 1)
             }
        )
        classification_layout.addWidget(dt_group)

        # Random Forest
        rf_group = self.create_algorithm_group(
            "Random Forest",
            {"n_estimators": ("int", 100, 10, 1000, 10),
             "criterion": (["gini", "entropy"], "gini"),
             "max_depth": ("int_optional", 10, 1, 100, 1),
             "min_samples_split": ("int", 2, 2, 100, 1)}
        )
        classification_layout.addWidget(rf_group)

        # KNN
        knn_group = self.create_algorithm_group(
            "K-Nearest Neighbors",
            {"n_neighbors": ("int", 5, 1, 100, 1),
             "weights": (["uniform", "distance"], "uniform"),
             "metric": (["euclidean", "manhattan", "minkowski"], "minkowski")}
        )
        classification_layout.addWidget(knn_group)

        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1, Qt.AlignmentFlag.AlignTop) # Align group to top

        return widget

    def create_cluster_dim_reduction_tab(self): # Renamed function
        """Create the clustering and dimensionality reduction tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # --- Clustering section ---
        clustering_group = QGroupBox("Clustering")
        clustering_layout = QVBoxLayout()

        # K-Means
        kmeans_params = self.create_algorithm_group(
            "K-Means", # Use algo name for button
            {"n_clusters": ("int", 3, 2, 50, 1), # Default 3 clusters
             "init": (["k-means++", "random"], "k-means++"),
             "n_init": ("int", 10, 1, 50, 1), # Number of initializations
             "max_iter": ("int", 300, 50, 1000, 50)}
        )
        clustering_layout.addWidget(kmeans_params)
        clustering_group.setLayout(clustering_layout)
        layout.addWidget(clustering_group, 0, 0, Qt.AlignmentFlag.AlignTop)

        # --- Dimensionality Reduction section ---
        dim_reduction_group = QGroupBox("Dimensionality Reduction")
        dim_reduction_layout = QVBoxLayout()

        # PCA
        pca_params = self.create_algorithm_group(
            "PCA", # Use algo name for button
            {"n_components": ("int_optional", 2, 1, 50, 1), # Default 2 components for visualization
             "whiten": ("checkbox", False)}
        )
        # Note: PCA is often used for preprocessing/visualization, not typically "trained" like others.
        # We might want a separate "Apply PCA" button or integrate it differently later.
        # For now, the train button will fit PCA and show explained variance.
        dim_reduction_layout.addWidget(pca_params)
        dim_reduction_group.setLayout(dim_reduction_layout)
        layout.addWidget(dim_reduction_group, 0, 1, Qt.AlignmentFlag.AlignTop)

        return widget

    # def create_rl_tab(self):
    #     """Create the reinforcement learning tab (Simplified)"""
    #     widget = QWidget()
    #     layout = QGridLayout(widget)
    #     label = QLabel("Reinforcement Learning section (Not fully implemented)")
    #     layout.addWidget(label)
    #     # Add placeholders if needed later
    #     return widget

    def create_visualization(self):
        """Create the visualization and metrics section"""
        viz_metrics_group = QGroupBox("Results: Visualization and Metrics")
        viz_metrics_layout = QHBoxLayout()

        # Matplotlib Figure
        self.figure = Figure(figsize=(7, 6)) # Adjusted size slightly
        self.canvas = FigureCanvas(self.figure)
        viz_metrics_layout.addWidget(self.canvas, 2) # Give more stretch factor

        # Metrics Display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMinimumWidth(300) # Ensure metrics area is readable
        viz_metrics_layout.addWidget(self.metrics_text, 1) # Less stretch factor

        viz_metrics_group.setLayout(viz_metrics_layout)
        self.layout.addWidget(viz_metrics_group)

    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200) # Limit progress bar width
        self.progress_bar.setTextVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("Ready")


    def create_algorithm_group(self, name, params_config):
        """
        Helper method to create algorithm parameter groups.
        params_config format: { param_name: (type_info, default_value, [min_val, max_val, step]) }
        type_info can be "int", "double", "checkbox", ["item1", "item2"], "int_optional"
        """
        group = QGroupBox(name)
        layout = QVBoxLayout()
        param_widgets = {}

        for param_name, config in params_config.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name.replace('_', ' ').title()}:")) # Nicer label

            widget = None
            param_type = config[0]
            default_value = config[1] if len(config) > 1 else None

            if param_type == "int":
                widget = QSpinBox()
                if len(config) > 4: widget.setRange(config[2], config[3]); widget.setSingleStep(config[4])
                else: widget.setRange(-99999, 99999)
                if default_value is not None: widget.setValue(default_value)
            elif param_type == "int_optional": # Allows setting to 'None' via checkbox
                 widget = QWidget() # Container widget
                 h_layout = QHBoxLayout(widget)
                 h_layout.setContentsMargins(0,0,0,0)
                 num_widget = QSpinBox()
                 if len(config) > 4: num_widget.setRange(config[2], config[3]); num_widget.setSingleStep(config[4])
                 else: num_widget.setRange(-99999, 99999)
                 if default_value is not None: num_widget.setValue(default_value)

                 cb_widget = QCheckBox("None")
                 cb_widget.setChecked(default_value is None)
                 num_widget.setEnabled(not cb_widget.isChecked())
                 cb_widget.toggled.connect(num_widget.setDisabled)

                 h_layout.addWidget(num_widget)
                 h_layout.addWidget(cb_widget)
                 # Store both widgets, access spinbox via [0]
                 param_widgets[param_name] = (num_widget, cb_widget)

            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setDecimals(4) # More precision for some params
                if len(config) > 4: widget.setRange(config[2], config[3]); widget.setSingleStep(config[4])
                else: widget.setRange(-99999.0, 99999.0)
                if default_value is not None: widget.setValue(default_value)
            elif param_type == "checkbox":
                widget = QCheckBox()
                if default_value is not None: widget.setChecked(default_value)
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)
                if default_value is not None and default_value in param_type:
                    widget.setCurrentText(default_value)

            if widget is not None: # Add non-compound widgets to layout
                 param_layout.addWidget(widget)
                 if param_type != "int_optional": # Store simple widgets directly
                     param_widgets[param_name] = widget

            layout.addLayout(param_layout)


        # Add Train button
        train_btn = QPushButton(f"Train {name}")
        # Use lambda with default argument to capture current state
        train_btn.clicked.connect(lambda checked=False, n=name, p=param_widgets: self.train_model(n, p))
        layout.addWidget(train_btn)

        group.setLayout(layout)
        return group

    def train_model(self, model_name, param_widgets):
        """Train the selected classical ML model"""
        if self.X_train is None or self.y_train is None:
            self.show_error("Data not processed or split yet. Please process data first.")
            return

        try:
            self.status_bar.showMessage(f"Training {model_name}...")
            self.progress_bar.setRange(0, 0) # Indeterminate progress
            QApplication.processEvents()

            params = {}
            for name, widget_or_tuple in param_widgets.items():
                if isinstance(widget_or_tuple, tuple) and isinstance(widget_or_tuple[1], QCheckBox): # Handle int_optional
                    num_widget, cb_widget = widget_or_tuple
                    if cb_widget.isChecked():
                        params[name] = None
                    else:
                        params[name] = num_widget.value()
                elif isinstance(widget_or_tuple, QSpinBox):
                    params[name] = widget_or_tuple.value()
                elif isinstance(widget_or_tuple, QDoubleSpinBox):
                    params[name] = widget_or_tuple.value()
                elif isinstance(widget_or_tuple, QCheckBox):
                    params[name] = widget_or_tuple.isChecked()
                elif isinstance(widget_or_tuple, QComboBox):
                    params[name] = widget_or_tuple.currentText()

            # --- Instantiate the correct model ---
            model = None
            task_type = 'unknown' # Determine if regression or classification

            if model_name == "Linear Regression":
                model = LinearRegression(**params)
                task_type = 'regression'
            elif model_name == "Logistic Regression":
                # Handle solver compatibility with multi_class and penalty (L1 needs saga/liblinear)
                if params.get('solver') == 'liblinear' and params.get('multi_class') == 'multinomial':
                     params['solver'] = 'lbfgs' # liblinear doesn't support multinomial
                     self.status_bar.showMessage("Warning: Switched solver to lbfgs for multinomial logistic regression.", 4000)
                model = LogisticRegression(**params, random_state=42)
                task_type = 'classification'
            elif model_name == "GaussianNB":
                 model = GaussianNB(**params)
                 task_type = 'classification'
            elif model_name == "SVC":
                 # gamma='auto' might need explicit calculation based on features if 'scale' isn't desired
                 model = SVC(**params, probability=True, random_state=42) # probability=True for potential ROC plots later
                 task_type = 'classification'
            elif model_name == "SVR":
                 model = SVR(**params)
                 task_type = 'regression'
            elif model_name == "Decision Tree":
                 model = DecisionTreeClassifier(**params, random_state=42)
                 task_type = 'classification' # Assuming classification DT
            elif model_name == "Random Forest":
                 model = RandomForestClassifier(**params, random_state=42, n_jobs=-1) # Use all cores
                 task_type = 'classification' # Assuming classification RF
            elif model_name == "K-Nearest Neighbors":
                 model = KNeighborsClassifier(**params, n_jobs=-1) # Use all cores
                 task_type = 'classification'
            elif model_name == "K-Means":
                model = KMeans(**params, random_state=42)
                task_type = 'clustering'
            elif model_name == "PCA":
                 model = PCA(**params)
                 task_type = 'dim_reduction'
            else:
                self.show_error(f"Model '{model_name}' training logic not implemented.")
                self.progress_bar.setRange(0, 100)
                self.status_bar.showMessage("Training failed.")
                return

            # --- Train the model ---
            if task_type == 'clustering' or task_type == 'dim_reduction':
                 # Clustering and PCA fit on the data (usually all data or just X_train)
                 # For simplicity, let's fit on X_train for consistency here
                 model.fit(self.X_train)
                 if task_type == 'clustering':
                      y_pred = model.predict(self.X_test) # Predict cluster labels for test set
                 else: # PCA
                      model.transform(self.X_train) # Apply transform
                      model.transform(self.X_test)
                      y_pred = None # No direct prediction for PCA evaluation here
            else: # Supervised models
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

            self.current_model = model
            self.current_model_name = model_name

            # --- Update Results ---
            self.update_metrics(y_pred, task_type)
            self.update_visualization(y_pred, task_type)

            self.progress_bar.setRange(0, 100) # Reset progress bar
            self.progress_bar.setValue(100)
            self.status_bar.showMessage(f"{model_name} training complete.")

        except Exception as e:
            self.show_error(f"Error training {model_name}: {str(e)}")
            self.progress_bar.setRange(0, 100)
            self.status_bar.showMessage(f"{model_name} training failed.")
            self.current_model = None
            self.current_model_name = None


    def create_deep_learning_tab(self):
        """Create the deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)

        # --- Architecture Configuration ---
        arch_group = QGroupBox("Neural Network Architecture")
        arch_layout = QVBoxLayout()

        self.layer_list_widget = QTextEdit() # Use QTextEdit to display layers
        self.layer_list_widget.setReadOnly(True)
        self.layer_list_widget.setFixedHeight(100)
        self._update_layer_display() # Initial empty display
        arch_layout.addWidget(QLabel("Current Layers:"))
        arch_layout.addWidget(self.layer_list_widget)


        layer_btn_layout = QHBoxLayout()
        add_layer_btn = QPushButton("Add Layer")
        add_layer_btn.clicked.connect(self.add_layer_dialog)
        clear_layers_btn = QPushButton("Clear All Layers")
        clear_layers_btn.clicked.connect(self._clear_layers)
        layer_btn_layout.addWidget(add_layer_btn)
        layer_btn_layout.addWidget(clear_layers_btn)
        arch_layout.addLayout(layer_btn_layout)

        arch_group.setLayout(arch_layout)
        layout.addWidget(arch_group, 0, 0)

        # --- Training Parameters ---
        train_params_group = QGroupBox("Training Configuration")
        train_params_layout = QVBoxLayout()

        # Task Type (Regression/Classification) - NEW
        task_layout = QHBoxLayout()
        task_layout.addWidget(QLabel("Task Type:"))
        self.nn_task_combo = QComboBox()
        self.nn_task_combo.addItems(["Classification", "Regression"])
        self.nn_task_combo.currentTextChanged.connect(self._update_loss_options) # Update loss when task changes
        task_layout.addWidget(self.nn_task_combo)
        train_params_layout.addLayout(task_layout)

        # Loss Function - NEW
        loss_layout = QHBoxLayout()
        loss_layout.addWidget(QLabel("Loss Function:"))
        self.nn_loss_combo = QComboBox()
        self._update_loss_options(self.nn_task_combo.currentText()) # Initial population
        loss_layout.addWidget(self.nn_loss_combo)
        train_params_layout.addLayout(loss_layout)

        # Other training parameters (Epochs, Batch Size, LR)
        train_params_layout.addWidget(self.create_training_params_group())

        # Train button
        train_nn_btn = QPushButton("Train Neural Network")
        train_nn_btn.clicked.connect(self.train_neural_network)
        train_params_layout.addWidget(train_nn_btn)

        train_params_group.setLayout(train_params_layout)
        layout.addWidget(train_params_group, 0, 1)

        # Placeholders for CNN/RNN specific controls if needed later
        # cnn_group = self.create_cnn_controls()
        # layout.addWidget(cnn_group, 1, 0)
        # rnn_group = self.create_rnn_controls()
        # layout.addWidget(rnn_group, 1, 1)

        return widget

    def _update_layer_display(self):
        """Updates the text edit showing the layer configuration."""
        if not self.layer_config:
            self.layer_list_widget.setText("No layers added yet.")
        else:
            display_text = ""
            for i, layer in enumerate(self.layer_config):
                params_str = ", ".join(f"{k}={v}" for k, v in layer["params"].items())
                display_text += f"{i+1}: {layer['type']}({params_str})\n"
            self.layer_list_widget.setText(display_text)

    def _clear_layers(self):
        """Clears the neural network layer configuration."""
        self.layer_config = []
        self._update_layer_display()
        self.status_bar.showMessage("Neural network layers cleared.")

    @pyqtSlot(str)
    def _update_loss_options(self, task_type):
        """Update the loss function ComboBox based on the selected task type."""
        self.nn_loss_combo.clear()
        self.nn_task_type = task_type # Store current task type
        if task_type == "Classification":
            self.nn_loss_combo.addItems([
                'categorical_crossentropy', # For multi-class one-hot encoded
                'sparse_categorical_crossentropy', # For multi-class integer labels
                'binary_crossentropy', # For binary classification
                'hinge',
                'squared_hinge'
            ])
        elif task_type == "Regression":
            self.nn_loss_combo.addItems([
                'mean_squared_error',
                'mean_absolute_error',
                'huber_loss',
                'mean_squared_logarithmic_error' # Good for positive targets with large range
            ])
        else: # Default/fallback
             self.nn_loss_combo.addItem("Select Task Type First")


    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        dialog_layout = QVBoxLayout(dialog) # Use layout directly on dialog

        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        # Common layers
        type_combo.addItems(["Dense", "Dropout", "Flatten", # General
                             "Conv2D", "MaxPooling2D", # CNN
                             "LSTM", "GRU" # RNN (require specific input shape)
                             ])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        dialog_layout.addLayout(type_layout)

        # Parameters input group
        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout() # Use QVBoxLayout for vertical stacking of params
        params_group.setLayout(params_layout)
        dialog_layout.addWidget(params_group)

        # Dictionary to hold dynamically created input widgets
        layer_param_inputs = {}

        def update_params():
            # Clear previous parameter widgets
            while params_layout.count():
                child = params_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
                elif child.layout(): # Clear sub-layouts if any
                    while child.layout().count():
                        sub_child = child.layout().takeAt(0)
                        if sub_child.widget():
                            sub_child.widget().deleteLater()
            layer_param_inputs.clear()

            layer_type = type_combo.currentText()

            # --- Add widgets based on layer type ---
            if layer_type == "Dense":
                units_layout = QHBoxLayout()
                units_layout.addWidget(QLabel("Units:"))
                units_input = QSpinBox()
                units_input.setRange(1, 10000); units_input.setValue(32)
                units_layout.addWidget(units_input)
                params_layout.addLayout(units_layout)
                layer_param_inputs["units"] = units_input

                act_layout = QHBoxLayout()
                act_layout.addWidget(QLabel("Activation:"))
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax", "linear", "None"]) # Added linear and None
                activation_combo.setCurrentText("relu")
                act_layout.addWidget(activation_combo)
                params_layout.addLayout(act_layout)
                layer_param_inputs["activation"] = activation_combo

            elif layer_type == "Dropout":
                rate_layout = QHBoxLayout()
                rate_layout.addWidget(QLabel("Dropout Rate:"))
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 0.9); rate_input.setValue(0.5); rate_input.setSingleStep(0.1)
                rate_layout.addWidget(rate_input)
                params_layout.addLayout(rate_layout)
                layer_param_inputs["rate"] = rate_input

            elif layer_type == "Flatten":
                params_layout.addWidget(QLabel("No parameters needed for Flatten."))

            elif layer_type == "Conv2D":
                filters_layout = QHBoxLayout()
                filters_layout.addWidget(QLabel("Filters:"))
                filters_input = QSpinBox()
                filters_input.setRange(1, 1024); filters_input.setValue(32)
                filters_layout.addWidget(filters_input)
                params_layout.addLayout(filters_layout)
                layer_param_inputs["filters"] = filters_input

                kernel_layout = QHBoxLayout()
                kernel_layout.addWidget(QLabel("Kernel Size (e.g., 3,3):"))
                kernel_input = QLineEdit("3,3")
                kernel_layout.addWidget(kernel_input)
                params_layout.addLayout(kernel_layout)
                layer_param_inputs["kernel_size"] = kernel_input # Will parse later

                act_layout = QHBoxLayout()
                act_layout.addWidget(QLabel("Activation:"))
                activation_combo_cnn = QComboBox()
                activation_combo_cnn.addItems(["relu", "tanh", "linear", "None"])
                activation_combo_cnn.setCurrentText("relu")
                act_layout.addWidget(activation_combo_cnn)
                params_layout.addLayout(act_layout)
                layer_param_inputs["activation"] = activation_combo_cnn

                padding_layout = QHBoxLayout()
                padding_layout.addWidget(QLabel("Padding:"))
                padding_combo = QComboBox()
                padding_combo.addItems(["valid", "same"])
                padding_layout.addWidget(padding_combo)
                params_layout.addLayout(padding_layout)
                layer_param_inputs["padding"] = padding_combo


            elif layer_type == "MaxPooling2D":
                 pool_layout = QHBoxLayout()
                 pool_layout.addWidget(QLabel("Pool Size (e.g., 2,2):"))
                 pool_input = QLineEdit("2,2")
                 pool_layout.addWidget(pool_input)
                 params_layout.addLayout(pool_layout)
                 layer_param_inputs["pool_size"] = pool_input # Will parse later

            elif layer_type == "LSTM" or layer_type == "GRU":
                 units_layout = QHBoxLayout()
                 units_layout.addWidget(QLabel("Units:"))
                 units_input_rnn = QSpinBox()
                 units_input_rnn.setRange(1, 10000); units_input_rnn.setValue(32)
                 units_layout.addWidget(units_input_rnn)
                 params_layout.addLayout(units_layout)
                 layer_param_inputs["units"] = units_input_rnn

                 # Add more RNN params like return_sequences if needed
                 params_layout.addWidget(QLabel("Note: RNN layers require specific 3D input shape (batch, timesteps, features)."))


        type_combo.currentIndexChanged.connect(update_params)
        update_params()  # Initial call to populate params for default selection

        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        dialog_layout.addLayout(btn_layout)

        def on_add_layer():
            layer_type = type_combo.currentText()
            layer_params = {}
            try:
                for param_name, widget in layer_param_inputs.items():
                    value = None
                    if isinstance(widget, QSpinBox):
                        value = widget.value()
                    elif isinstance(widget, QDoubleSpinBox):
                        value = widget.value()
                    elif isinstance(widget, QComboBox):
                         value = widget.currentText()
                         if value == "None": value = None # Handle activation='None'
                    elif isinstance(widget, QLineEdit):
                        # Handle tuple inputs carefully
                        if param_name in ["kernel_size", "pool_size"]:
                             value = parse_int_tuple(widget.text())
                        else: # Default string input
                             value = widget.text()
                    # Store collected value
                    if value is not None or param_name == 'activation': # Keep activation even if None
                        layer_params[param_name] = value

                self.layer_config.append({
                    "type": layer_type,
                    "params": layer_params
                })
                self._update_layer_display() # Update the list in the main window
                dialog.accept()
            except Exception as e:
                 self.show_error(f"Error parsing layer parameters: {e}")


        add_btn.clicked.connect(on_add_layer)
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec()


    def create_training_params_group(self):
        """Create group widget for NN training parameters (reusable)"""
        # This returns a QWidget containing the layout, not a QGroupBox
        # to be added directly to another layout.
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0,0,0,0) # No extra margins

        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 2048); self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)

        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000); self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)

        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(1e-6, 1.0); self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001); self.lr_spin.setDecimals(6)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)

        # Optimizer (Optional - Add if more needed)
        # opt_layout = QHBoxLayout()
        # opt_layout.addWidget(QLabel("Optimizer:"))
        # self.optimizer_combo = QComboBox()
        # self.optimizer_combo.addItems(["Adam", "RMSprop", "SGD"])
        # opt_layout.addWidget(self.optimizer_combo)
        # layout.addLayout(opt_layout)

        return widget

    def create_cnn_controls(self):
        """Create placeholder controls for CNN (if needed later)"""
        group = QGroupBox("CNN Specific Config (Placeholder)")
        layout = QVBoxLayout()
        label = QLabel("CNN Controls (To be implemented if needed)")
        layout.addWidget(label)
        group.setLayout(layout)
        return group

    def create_rnn_controls(self):
        """Create placeholder controls for RNN (if needed later)"""
        group = QGroupBox("RNN Specific Config (Placeholder)")
        layout = QVBoxLayout()
        label = QLabel("RNN Controls (To be implemented if needed)")
        layout.addWidget(label)
        group.setLayout(layout)
        return group


    def create_neural_network(self, input_shape, num_classes=None):
        """Create neural network based on current layer configuration"""
        model = models.Sequential(name="ML_Course_NN")
        first_layer = True

        for i, layer_conf in enumerate(self.layer_config):
            layer_type = layer_conf["type"]
            params = layer_conf["params"].copy() # Copy params to avoid modifying original config
            layer = None

            try:
                if layer_type == "Dense":
                    if first_layer: params['input_shape'] = input_shape
                    layer = layers.Dense(**params)
                elif layer_type == "Conv2D":
                     if first_layer: params['input_shape'] = input_shape # Expecting (H, W, C)
                     layer = layers.Conv2D(**params)
                elif layer_type == "MaxPooling2D":
                    layer = layers.MaxPooling2D(**params)
                elif layer_type == "Flatten":
                    # Flatten needs input_shape if it's the first layer (unusual but possible)
                    if first_layer: params['input_shape'] = input_shape
                    layer = layers.Flatten(**params)
                elif layer_type == "Dropout":
                    layer = layers.Dropout(**params)
                elif layer_type == "LSTM":
                     if first_layer: params['input_shape'] = input_shape # Expecting (timesteps, features)
                     # Decide if return_sequences=True is needed based on next layer
                     if i + 1 < len(self.layer_config) and self.layer_config[i+1]['type'] in ["LSTM", "GRU"]:
                         params['return_sequences'] = True
                     layer = layers.LSTM(**params)
                elif layer_type == "GRU":
                     if first_layer: params['input_shape'] = input_shape # Expecting (timesteps, features)
                     if i + 1 < len(self.layer_config) and self.layer_config[i+1]['type'] in ["LSTM", "GRU"]:
                         params['return_sequences'] = True
                     layer = layers.GRU(**params)
                else:
                     print(f"Warning: Layer type '{layer_type}' not recognized by builder.") # Use print for now
                     continue

                if layer:
                     model.add(layer)
                     first_layer = False # Input shape only needed for the very first layer added

            except Exception as build_err:
                 self.show_error(f"Error adding layer {i+1} ({layer_type}): {build_err}\nParams: {params}")
                 return None # Indicate model build failure


        # --- Add final output layer based on task type ---
        if not first_layer: # Only add output if other layers exist
            if self.nn_task_type == 'Classification':
                if num_classes is None:
                     self.show_error("Cannot determine number of classes for classification output layer.")
                     return None
                activation = 'softmax' if num_classes > 2 else 'sigmoid' # Softmax for multi-class, sigmoid for binary
                # Handle case where loss is hinge - needs linear activation
                selected_loss = self.nn_loss_combo.currentText()
                if 'hinge' in selected_loss:
                    activation = 'linear'
                output_units = num_classes if num_classes > 2 else 1 # 1 unit for binary classification
                model.add(layers.Dense(output_units, activation=activation, name='output'))
            elif self.nn_task_type == 'Regression':
                model.add(layers.Dense(1, activation='linear', name='output')) # Typically 1 output node for regression

        return model

    def train_neural_network(self):
        """Train the neural network with current configuration"""
        if self.X_train is None or self.y_train is None:
            self.show_error("Data not processed or split yet. Please process data first.")
            return
        if not self.layer_config:
            self.show_error("Please add at least one layer to the neural network.")
            return

        try:
            self.status_bar.showMessage("Configuring Neural Network...")
            self.progress_bar.setRange(0, 100) # Use 0-100 for epochs
            self.progress_bar.setValue(0)
            QApplication.processEvents()

            # --- Prepare Data ---
            X_train_nn = self.X_train
            X_test_nn = self.X_test
            y_train_nn = self.y_train
            y_test_nn = self.y_test

            # Determine input shape (handle 1D, 2D, 3D data)
            if len(X_train_nn.shape) == 1: # Should not happen with tabular data, but safeguard
                input_shape = (1,)
            else:
                input_shape = X_train_nn.shape[1:] # (features,) or (H, W, C) etc.

             # Check if data needs reshaping for CNN/RNN (basic check)
            first_layer_type = self.layer_config[0]['type'] if self.layer_config else None
            if first_layer_type in ["Conv2D", "MaxPooling2D"]:
                 # Try to infer image dimensions (assuming square images if 2D input)
                 if len(input_shape) == 1: # Flattened image?
                     side = int(np.sqrt(input_shape[0]))
                     if side * side == input_shape[0]:
                         # Reshape to (H, W, C=1) for grayscale
                         X_train_nn = X_train_nn.reshape((-1, side, side, 1))
                         X_test_nn = X_test_nn.reshape((-1, side, side, 1))
                         input_shape = (side, side, 1)
                         self.status_bar.showMessage("Reshaped flat input for Conv2D layer.", 3000)
                     else:
                         self.show_error(f"Cannot automatically reshape 1D input of size {input_shape[0]} for Conv2D. Requires manual reshaping.")
                         return
                 elif len(input_shape) == 2: # Already (H, W)? Assume 1 channel.
                     X_train_nn = X_train_nn.reshape((-1, input_shape[0], input_shape[1], 1))
                     X_test_nn = X_test_nn.reshape((-1, input_shape[0], input_shape[1], 1))
                     input_shape = input_shape + (1,) # Add channel dim
                     self.status_bar.showMessage("Added channel dimension for Conv2D layer.", 3000)
                 elif len(input_shape) != 3: # Expecting (H, W, C)
                      self.show_error(f"Conv2D layers expect 3D input (Height, Width, Channels), got shape {input_shape}")
                      return

            elif first_layer_type in ["LSTM", "GRU"]:
                 # RNNs expect (batch, timesteps, features)
                 # Basic assumption: treat sequence of features as 1 timestep
                 if len(input_shape) == 1: # (features,)
                      X_train_nn = X_train_nn.reshape((-1, 1, input_shape[0]))
                      X_test_nn = X_test_nn.reshape((-1, 1, input_shape[0]))
                      input_shape = (1, input_shape[0]) # (timesteps, features)
                      self.status_bar.showMessage("Reshaped input for RNN layer (assuming 1 timestep).", 3000)
                 elif len(input_shape) != 2: # Expecting (timesteps, features)
                      self.show_error(f"RNN layers expect 2D input (timesteps, features) per sample, got shape {input_shape}")
                      return


            # --- Target Variable Preparation ---
            num_classes = None
            if self.nn_task_type == 'Classification':
                 # Ensure y is integer type for sparse crossentropy or one-hot encoding
                 try:
                     y_train_nn = y_train_nn.astype(int)
                     y_test_nn = y_test_nn.astype(int)
                 except ValueError as e:
                     self.show_error(f"Could not convert target variable to integer for classification: {e}. Ensure target is categorical.")
                     return

                 num_classes = len(np.unique(y_train_nn)) # Use unique labels in train set
                 selected_loss = self.nn_loss_combo.currentText()

                 # One-hot encode ONLY if using categorical_crossentropy
                 if selected_loss == 'categorical_crossentropy':
                     if num_classes < 2:
                          self.show_error("Categorical crossentropy requires at least 2 classes.")
                          return
                     y_train_nn = tf.keras.utils.to_categorical(y_train_nn, num_classes=num_classes)
                     y_test_nn = tf.keras.utils.to_categorical(y_test_nn, num_classes=num_classes)
                     self.status_bar.showMessage("One-hot encoded target variable.", 3000)
                 # For binary/sparse/hinge, keep integer labels
                 elif num_classes == 2 and selected_loss != 'binary_crossentropy':
                      # For hinge losses with binary, often expect {-1, 1} but Keras handles {0, 1}
                      pass
            else: # Regression - ensure y is float
                 try:
                     y_train_nn = y_train_nn.astype(float)
                     y_test_nn = y_test_nn.astype(float)
                 except ValueError as e:
                     self.show_error(f"Could not convert target variable to float for regression: {e}. Ensure target is numeric.")
                     return


            # --- Create and Compile Model ---
            model = self.create_neural_network(input_shape, num_classes)
            if model is None: return # Error during model creation

            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            loss_function = self.nn_loss_combo.currentText()
            optimizer = optimizers.Adam(learning_rate=learning_rate)

            # Determine metrics based on task
            metrics = []
            if self.nn_task_type == 'Classification':
                 # Use appropriate accuracy metric based on loss/encoding
                 if loss_function == 'categorical_crossentropy':
                     metrics.append(tf.keras.metrics.CategoricalAccuracy(name='accuracy'))
                 elif loss_function == 'sparse_categorical_crossentropy':
                     metrics.append(tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'))
                 elif loss_function == 'binary_crossentropy':
                      metrics.append(tf.keras.metrics.BinaryAccuracy(name='accuracy'))
                 else: # Hinge etc. - still use accuracy, but interpretation might differ
                      metrics.append(tf.keras.metrics.Accuracy(name='accuracy')) # Generic accuracy for others
            elif self.nn_task_type == 'Regression':
                 metrics.append(tf.keras.metrics.MeanSquaredError(name='mse'))
                 metrics.append(tf.keras.metrics.MeanAbsoluteError(name='mae'))

            model.compile(optimizer=optimizer,
                          loss=loss_function,
                          metrics=metrics)

            model.summary(print_fn=lambda x: self.metrics_text.append(x)) # Display summary in metrics box

            self.status_bar.showMessage(f"Starting training ({epochs} epochs)...")
            QApplication.processEvents()

            # --- Train Model ---
            history = model.fit(X_train_nn, y_train_nn,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(X_test_nn, y_test_nn),
                                callbacks=[self.create_progress_callback()],
                                verbose=0) # Use callback for progress, less console output

            self.current_model = model # Store the trained Keras model
            self.current_model_name = "Neural Network"

            # --- Update Visualization & Metrics ---
            self.plot_training_history(history) # Show loss/metric curves
            # Optionally evaluate and show final metrics on test set
            eval_results = model.evaluate(X_test_nn, y_test_nn, verbose=0)
            metrics_str = f"Final Test Set Performance ({self.nn_task_type}):\n"
            for name, value in zip(model.metrics_names, eval_results):
                 metrics_str += f" - {name}: {value:.4f}\n"
            self.metrics_text.append("\n" + metrics_str)


            self.status_bar.showMessage("Neural Network Training Complete.")

        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")
            self.status_bar.showMessage("Neural Network training failed.")
            self.progress_bar.setRange(0,100) # Reset progress bar
            self.progress_bar.setValue(0)
            self.current_model = None
            self.current_model_name = None


    def create_progress_callback(self):
        """Create callback for updating progress bar during Keras training"""
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar, status_bar):
                super().__init__()
                self.progress_bar = progress_bar
                self.status_bar = status_bar
                self.epochs = 0

            def on_train_begin(self, logs=None):
                self.epochs = self.params['epochs']
                self.progress_bar.setRange(0, self.epochs)
                self.progress_bar.setValue(0)

            def on_epoch_end(self, epoch, logs=None):
                self.progress_bar.setValue(epoch + 1)
                # Display current loss/metric in status bar
                log_str = f"Epoch {epoch+1}/{self.epochs}"
                for k, v in logs.items():
                    if 'loss' in k or 'accuracy' in k or 'mae' in k or 'mse' in k: # Show common metrics
                        log_str += f" - {k}: {v:.4f}"
                self.status_bar.showMessage(log_str)
                QApplication.processEvents() # Keep UI responsive

            def on_train_end(self, logs=None):
                 self.progress_bar.setRange(0,100) # Reset after training
                 self.progress_bar.setValue(100)


        return ProgressCallback(self.progress_bar, self.status_bar)


    def update_visualization(self, y_pred, task_type):
        """Update the visualization panel based on the task type and predictions"""
        if self.X_test is None: return
        self.figure.clear()

        try:
            if task_type == 'regression':
                ax = self.figure.add_subplot(111)
                ax.scatter(self.y_test, y_pred, alpha=0.6, label="Predictions")
                # Add a line y=x for reference
                lims = [
                    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
                ]
                ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Ideal Fit")
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.set_title(f"{self.current_model_name}: Actual vs. Predicted")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)

            elif task_type == 'classification':
                 # Use PCA for visualization if more than 2 features
                 if self.X_test.shape[1] > 2:
                     pca = PCA(n_components=2, random_state=42)
                     X_test_2d = pca.fit_transform(self.X_test)
                     ax = self.figure.add_subplot(111)
                     scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
                     ax.set_xlabel("Principal Component 1")
                     ax.set_ylabel("Principal Component 2")
                     ax.set_title(f"{self.current_model_name}: Predictions (PCA Reduced)")
                     # Add colorbar legend
                     try: # Handle cases where colorbar might fail (e.g., single class prediction)
                          legend1 = self.figure.colorbar(scatter, ax=ax)
                          legend1.set_label('Predicted Class')
                     except Exception as cb_err:
                          print(f"Could not create colorbar: {cb_err}") # Log error
                 elif self.X_test.shape[1] == 2: # Direct 2D plot
                     ax = self.figure.add_subplot(111)
                     scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
                     # Try getting feature names if X_test is a DataFrame
                     try:
                          ax.set_xlabel(self.X_test.columns[0])
                          ax.set_ylabel(self.X_test.columns[1])
                     except AttributeError:
                          ax.set_xlabel("Feature 1")
                          ax.set_ylabel("Feature 2")
                     ax.set_title(f"{self.current_model_name}: Predictions")
                     try:
                          legend1 = self.figure.colorbar(scatter, ax=ax)
                          legend1.set_label('Predicted Class')
                     except Exception as cb_err:
                           print(f"Could not create colorbar: {cb_err}")
                 else: # 1 Feature classification
                      ax = self.figure.add_subplot(111)
                      # Jitter y-values slightly for better visibility
                      y_jitter = np.random.rand(len(y_pred)) * 0.1
                      ax.scatter(self.X_test[:, 0], y_pred + y_jitter, c=y_pred, cmap='viridis', alpha=0.6)
                      try: ax.set_xlabel(self.X_test.columns[0])
                      except AttributeError: ax.set_xlabel("Feature")
                      ax.set_ylabel("Predicted Class (with jitter)")
                      ax.set_title(f"{self.current_model_name}: Predictions")
                      ax.set_yticks(np.unique(y_pred)) # Set ticks to actual predicted classes

            elif task_type == 'clustering':
                 # Similar visualization to classification, using PCA if needed
                 if self.X_test.shape[1] > 2:
                     pca = PCA(n_components=2, random_state=42)
                     X_test_2d = pca.fit_transform(self.X_test)
                     ax = self.figure.add_subplot(111)
                     scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
                     ax.set_xlabel("Principal Component 1")
                     ax.set_ylabel("Principal Component 2")
                     ax.set_title(f"{self.current_model_name}: Clusters (PCA Reduced)")
                     try:
                          legend1 = self.figure.colorbar(scatter, ax=ax)
                          legend1.set_label('Predicted Cluster')
                     except Exception as cb_err:
                          print(f"Could not create colorbar: {cb_err}")
                 elif self.X_test.shape[1] == 2:
                     ax = self.figure.add_subplot(111)
                     scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
                     try: ax.set_xlabel(self.X_test.columns[0]); ax.set_ylabel(self.X_test.columns[1])
                     except AttributeError: ax.set_xlabel("Feature 1"); ax.set_ylabel("Feature 2")
                     ax.set_title(f"{self.current_model_name}: Clusters")
                     try:
                          legend1 = self.figure.colorbar(scatter, ax=ax)
                          legend1.set_label('Predicted Cluster')
                     except Exception as cb_err:
                          print(f"Could not create colorbar: {cb_err}")
                 else:
                     ax = self.figure.add_subplot(111)
                     ax.scatter(self.X_test[:, 0], np.zeros_like(self.X_test[:, 0]), c=y_pred, cmap='viridis', alpha=0.7)
                     try: ax.set_xlabel(self.X_test.columns[0])
                     except AttributeError: ax.set_xlabel("Feature")
                     ax.set_yticks([])
                     ax.set_title(f"{self.current_model_name}: Clusters (1D)")

            elif task_type == 'dim_reduction': # PCA specific visualization
                 if isinstance(self.current_model, PCA):
                     explained_variance_ratio = self.current_model.explained_variance_ratio_
                     ax = self.figure.add_subplot(111)
                     ax.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.8)
                     ax.plot(range(1, len(explained_variance_ratio) + 1), np.cumsum(explained_variance_ratio), 'r-o', label='Cumulative Variance')
                     ax.set_xlabel("Principal Component")
                     ax.set_ylabel("Explained Variance Ratio")
                     ax.set_title(f"PCA Explained Variance (Total: {np.sum(explained_variance_ratio):.3f})")
                     ax.set_xticks(range(1, len(explained_variance_ratio) + 1))
                     ax.legend()
                     ax.grid(True, axis='y', linestyle='--', alpha=0.5)
                 else:
                      self.figure.text(0.5, 0.5, 'Visualization not available for this reduction type.', ha='center', va='center')

            else:
                 self.figure.text(0.5, 0.5, 'Visualization not implemented for this task type.', ha='center', va='center')

            self.figure.tight_layout() # Adjust layout to prevent overlap
            self.canvas.draw()

        except Exception as e:
             self.show_error(f"Error updating visualization: {e}")
             self.figure.clear()
             self.figure.text(0.5, 0.5, f"Error creating plot:\n{e}", color='red', ha='center', va='center', wrap=True)
             self.canvas.draw()


    def update_metrics(self, y_pred, task_type):
        """Update metrics display based on task type and predictions"""
        metrics_title = f"--- {self.current_model_name} Performance Metrics ---"
        metrics_text = metrics_title + "\n\n"

        try:
            if task_type == 'regression':
                 if y_pred is not None and self.y_test is not None:
                     mse = mean_squared_error(self.y_test, y_pred)
                     mae = mean_absolute_error(self.y_test, y_pred)
                     rmse = np.sqrt(mse)
                     # R2 score might fail if model hasn't implemented score or for constant predictions
                     r2 = float('nan')
                     try:
                         if hasattr(self.current_model, 'score'):
                              r2 = self.current_model.score(self.X_test, self.y_test)
                         else: # Calculate manually if score method not present
                              r2 = r2_score(self.y_test, y_pred)
                     except Exception as r2_err:
                          print(f"Could not calculate R2 score: {r2_err}")


                     metrics_text += f"Mean Squared Error (MSE): {mse:.4f}\n"
                     metrics_text += f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
                     metrics_text += f"Mean Absolute Error (MAE): {mae:.4f}\n"
                     metrics_text += f"R² Score: {r2:.4f}\n"
                 else:
                      metrics_text += "Predictions or test data not available for metric calculation.\n"

            elif task_type == 'classification':
                 if y_pred is not None and self.y_test is not None:
                     accuracy = accuracy_score(self.y_test, y_pred)
                     metrics_text += f"Accuracy: {accuracy:.4f}\n\n"
                     try:
                         conf_matrix = confusion_matrix(self.y_test, y_pred)
                         metrics_text += "Confusion Matrix:\n"
                         # Pretty print matrix if possible
                         try:
                             labels = np.unique(np.concatenate((self.y_test, y_pred)))
                             cm_df = pd.DataFrame(conf_matrix, index=[f"True:{l}" for l in labels], columns=[f"Pred:{l}" for l in labels])
                             metrics_text += cm_df.to_string() + "\n"
                         except: # Fallback to numpy array string
                              metrics_text += str(conf_matrix) + "\n"

                     except ValueError as cm_error: # Handle case where CM cannot be calculated
                          metrics_text += f"Could not generate Confusion Matrix: {cm_error}\n"

                     # Add note for GaussianNB var_smoothing
                     if isinstance(self.current_model, GaussianNB):
                          metrics_text += f"\n(GaussianNB var_smoothing: {self.current_model.var_smoothing})\n"

                 else:
                      metrics_text += "Predictions or test data not available for metric calculation.\n"

            elif task_type == 'clustering':
                 metrics_text += "Clustering Metrics (if ground truth available):\n"
                 # Placeholder for metrics like Silhouette Score, ARI, etc.
                 # These often require y_test (true labels) which might not always be relevant/available for clustering.
                 if hasattr(self.current_model, 'inertia_'):
                     metrics_text += f" - Inertia (within-cluster SSE): {self.current_model.inertia_:.4f}\n"
                 metrics_text += "(Add Silhouette Score, ARI, etc. if needed)\n"

            elif task_type == 'dim_reduction':
                  if isinstance(self.current_model, PCA):
                     metrics_text += "Principal Component Analysis Results:\n"
                     metrics_text += f" - Number of Components: {self.current_model.n_components_}\n"
                     metrics_text += f" - Explained Variance Ratio per Component:\n {np.round(self.current_model.explained_variance_ratio_, 4)}\n"
                     metrics_text += f" - Total Explained Variance: {np.sum(self.current_model.explained_variance_ratio_):.4f}\n"
                  else:
                      metrics_text += "Metrics not implemented for this reduction type.\n"

            else:
                 metrics_text += "Metrics calculation not implemented for this task type.\n"

        except Exception as e:
             metrics_text += f"\nError calculating metrics: {str(e)}\n"

        self.metrics_text.setText(metrics_text)


    def plot_training_history(self, history):
        """Plot neural network training history (loss and metrics)"""
        self.figure.clear()
        history_dict = history.history

        # --- Determine primary metric (accuracy or MAE/MSE) ---
        primary_metric = None
        val_primary_metric = None
        if 'accuracy' in history_dict:
            primary_metric = 'accuracy'
            val_primary_metric = 'val_accuracy'
        elif 'binary_accuracy' in history_dict:
            primary_metric = 'binary_accuracy'
            val_primary_metric = 'val_binary_accuracy'
        elif 'categorical_accuracy' in history_dict:
            primary_metric = 'categorical_accuracy'
            val_primary_metric = 'val_categorical_accuracy'
        elif 'sparse_categorical_accuracy' in history_dict:
            primary_metric = 'sparse_categorical_accuracy'
            val_primary_metric = 'val_sparse_categorical_accuracy'
        elif 'mae' in history_dict:
            primary_metric = 'mae'
            val_primary_metric = 'val_mae'
        elif 'mse' in history_dict: # Fallback if MAE not present
             primary_metric = 'mse'
             val_primary_metric = 'val_mse'

        epochs_range = range(1, len(history_dict['loss']) + 1)

        # Plot Loss
        ax1 = self.figure.add_subplot(211 if primary_metric else 111) # Use 2 rows if metric exists
        ax1.plot(epochs_range, history_dict['loss'], label='Training Loss')
        if 'val_loss' in history_dict:
            ax1.plot(epochs_range, history_dict['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Plot Primary Metric (if found)
        if primary_metric and val_primary_metric and val_primary_metric in history_dict:
            ax2 = self.figure.add_subplot(212)
            ax2.plot(epochs_range, history_dict[primary_metric], label=f'Training {primary_metric.title()}')
            ax2.plot(epochs_range, history_dict[val_primary_metric], label=f'Validation {primary_metric.title()}')
            ax2.set_title(f'Training and Validation {primary_metric.title()}')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel(primary_metric.title())
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.5)
        elif primary_metric: # Only training metric available?
            ax2 = self.figure.add_subplot(212)
            ax2.plot(epochs_range, history_dict[primary_metric], label=f'Training {primary_metric.title()}')
            ax2.set_title(f'Training {primary_metric.title()}')
            ax2.set_xlabel('Epoch'); ax2.set_ylabel(primary_metric.title())
            ax2.legend(); ax2.grid(True, linestyle='--', alpha=0.5)


        self.figure.tight_layout()
        self.canvas.draw()

    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)

    def clear_data(self):
        """Clear all loaded and processed data."""
        self.original_data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_column_name = None # Clear custom target name
        self.status_bar.showMessage("Data cleared.")
        self.clear_results()

    def clear_results(self):
         """Clear model, metrics, and visualization."""
         self.current_model = None
         self.current_model_name = None
         self.metrics_text.clear()
         self.figure.clear()
         self.canvas.draw()
         self.progress_bar.setValue(0)


def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    # Set High DPI scaling for better look on modern displays
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
