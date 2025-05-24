""" 
    1906A703 Alişan AYHAN

    Dependencies: 

    *   PyQt6
    *   NumPy
    *   Pandas
    *   Matplotlib
    *   Scikit-learn (`sklearn`)
    *   TensorFlow (includes Keras)
    *   umap-learn

    pip install PyQt6 numpy pandas matplotlib scikit-learn tensorflow umap-learn

"""


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
# Sklearn imports
from sklearn import datasets, preprocessing, model_selection
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor # Added Regressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # Added Regressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor # Added Regressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # Added LDA
from sklearn.manifold import TSNE # Added t-SNE
from sklearn.metrics import (accuracy_score, mean_squared_error, mean_absolute_error,
                           confusion_matrix, r2_score, silhouette_score, # Added Silhouette
                           get_scorer) # Helper for CV scoring
from sklearn.model_selection import cross_val_score, KFold # Added CV tools
from sklearn.base import clone # For cloning models in CV

# UMAP import (requires pip install umap-learn)
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not found. UMAP functionality will be disabled. "
          "Install with: pip install umap-learn")

# Tensorflow import
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

# --- Helper function (unchanged) ---
def parse_int_tuple(text, default=(3, 3)):
    try:
        parts = [int(p.strip()) for p in text.split(',')]
        if len(parts) >= 2:
            return tuple(parts[:2])
        else:
            return default
    except ValueError:
        return default
    




class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI (Enhanced + Dim Reduction)")
        self.setGeometry(100, 100, 1600, 900) # Increased size

        # --- Data Containers (mostly unchanged) ---
        self.original_data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        self.current_model_name = None
        self.last_trained_params = {} # Store params used for training/CV
        self.last_task_type = 'unknown' # Store task type of last trained model

        # --- NN Config (unchanged) ---
        self.layer_config = []
        self.nn_task_type = 'Classification'

        # --- Main Layout ---
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # --- Create Components ---
        self.create_data_section()
        self.create_status_bar()
        self.create_tabs()
        # Add Cross-Validation section separately
        self.create_cross_validation_section()
        self.create_visualization()
        # Add Eigenvector Example Button
        self.create_misc_section()

        self.X_train_for_grads = None
        self.y_train_for_grads = None
        self.gradient_tracker_callback = None # Also ensure this is initialized

    def create_data_section(self):
        """Create the data loading and preprocessing section (Minor Edits)"""
        data_group = QGroupBox("Data Management")
        data_layout = QGridLayout()

        # Row 0: Dataset selection and Load Button
        data_layout.addWidget(QLabel("Dataset:"), 0, 0)
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset (Classification)", # Add task type hint
            "Breast Cancer Dataset (Classification)",
            "Digits Dataset (Classification)",
            "California Housing Dataset (Regression)", # Renamed during implementation
            # "Boston Housing Dataset (Regression)", # Removed/Replaced
        ])
        self.dataset_combo.currentIndexChanged.connect(self._dataset_selection_changed)
        data_layout.addWidget(self.dataset_combo, 0, 1)

        self.load_btn = QPushButton("Load Custom CSV")
        self.load_btn.clicked.connect(self.load_custom_data)
        data_layout.addWidget(self.load_btn, 0, 2)

        # Row 1: Preprocessing options
        data_layout.addWidget(QLabel("Missing Values:"), 1, 0)
        self.missing_values_combo = QComboBox()
        self.missing_values_combo.addItems([
            "No Handling", "Mean Imputation", "Median Imputation",
            "Forward Fill", "Backward Fill", "Drop Rows"
        ])
        data_layout.addWidget(self.missing_values_combo, 1, 1)

        data_layout.addWidget(QLabel("Scaling:"), 1, 2)
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling", "Standard Scaling", "Min-Max Scaling", "Robust Scaling"
        ])
        data_layout.addWidget(self.scaling_combo, 1, 3)

        # Row 2: Test Split
        data_layout.addWidget(QLabel("Test Split Ratio:"), 2, 0)
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.05) # Finer steps
        data_layout.addWidget(self.split_spin, 2, 1)

        # Row 3: Process Data Button
        self.process_data_btn = QPushButton("Process and Split Data")
        self.process_data_btn.clicked.connect(self.process_and_split_data)
        data_layout.addWidget(self.process_data_btn, 3, 0, 1, 4) # Span columns

        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)

    class GradientTrackerCallback(tf.keras.callbacks.Callback):
        def __init__(self, gui_instance, update_interval=5): # update_interval: store gradients every N epochs
            super().__init__()
            self.gui = gui_instance
            self.update_interval = update_interval
            self.all_gradients_history = {} # {layer_name: {epoch: [grads]}}
            self.current_epoch_gradients = {} # {layer_name: [grads_for_current_epoch]}

        def on_epoch_begin(self, epoch, logs=None):
            self.current_epoch_gradients = {} # Reset for new epoch

        def on_train_batch_end(self, batch, logs=None):
            if not hasattr(self.model, 'trainable_weights') or not self.model.trainable_weights:
                return
            if not hasattr(self.model.optimizer, 'get_gradients'):
                return # Optimizer might not expose gradients directly (e.g. custom)

            # This part is a bit tricky as direct access to gradients per batch can be complex
            # A more common way is to use tf.GradientTape within a custom training loop.
            # For callbacks, it's harder. A simplified approach:
            # We might only be able to get gradients at the end of an epoch or via a more involved setup.
            # Let's try to get them if the optimizer supports it, otherwise this feature will be limited.
            # This is a very simplified version. For robust gradient tracking, TensorBoard or custom train_step is better.
            
            # For now, let's store weights at end of epoch as a proxy, or try to access gradients if simple.
            # The prompt asks for "weight gradients".
            pass # Actual per-batch gradient capture is complex here. We'll do per-epoch below.


        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % self.update_interval == 0:
                self.gui.status_bar.showMessage(f"Epoch {epoch+1}: Capturing gradients...", 1000)
                QApplication.processEvents()
                
                # This requires a bit of a hack or specific model structure to easily get gradients.
                # For simplicity, let's store gradients of a few layers if possible.
                # This is illustrative and might need a custom training loop for full power.
                # We'll try a simplified approach using a GradientTape on a dummy forward pass.
                
                if hasattr(self.model, 'trainable_weights') and self.model.trainable_weights:
                    # This is not the gradients *during* the optimizer update, but gradients *at this point in time*.
                    # It's an approximation.
                    # We need input data to compute gradients. Use a sample from X_train if available.
                    # This is computationally expensive to do every epoch.
                    
                    # Let's keep it simpler: store the *weights* and their *changes* or just enable a button
                    # for the user to trigger a gradient calculation on a sample.
                    # The prompt is "plot histograms of weight gradients *during* training".
                    # Let's store some representative gradients.
                    
                    # --- Alternative: Store weights and plot their distribution (not gradients) ---
                    # layer_weights = {}
                    # for layer in self.model.layers:
                    #     if layer.trainable_weights:
                    #         layer_weights[layer.name] = [w.numpy().flatten() for w in layer.get_weights() if w.ndim > 1] # Kernels mainly
                    # self.all_gradients_history[epoch] = layer_weights # Storing weights, not gradients here

                    # --- Attempting to store actual gradients (can be slow) ---
                    # We need some input data. This is tricky within a callback without access to current batch.
                    # For now, this callback will primarily serve as a placeholder for a more robust solution
                    # or will be used to collect gradients if model.optimizer.get_gradients is available and simple.

                    # Store the current epoch's gradients (if any were collected by a more complex mechanism)
                    # For now, this `self.current_epoch_gradients` would be populated by a more direct method.
                    # We will fake it by storing weights if actual gradient capture is too complex for callback.
                    
                    # Let's make this callback store the weights for now, and we'll rename "gradients" to "weights" in plotting
                    # if true gradient capture is too hard.
                    # For the spirit of the request, let's assume `self.current_epoch_gradients` can be populated.
                    # If it remains empty, the plot function will handle it.
                    #
                    # A more feasible approach within a callback for illustrative purposes:
                    # If we have `self.gui.X_train_nn_sample` (a small sample of training data)
                    # we could compute gradients with respect to that sample.
                    if hasattr(self.gui, 'X_train_for_grads') and self.gui.X_train_for_grads is not None and \
                    hasattr(self.gui, 'y_train_for_grads') and self.gui.y_train_for_grads is not None:
                        
                        with tf.GradientTape() as tape:
                            y_pred = self.model(self.gui.X_train_for_grads, training=False) # Use a small sample
                            # Calculate loss based on model's compiled loss
                            loss_value = self.model.compiled_loss(self.gui.y_train_for_grads, y_pred, regularization_losses=self.model.losses)

                        watched_weights = [w for w in self.model.trainable_weights if "kernel" in w.name or "bias" in w.name] # Focus on kernels/biases
                        if watched_weights:
                            grads = tape.gradient(loss_value, watched_weights)
                            for i, w_name in enumerate(watched_weights):
                                layer_name = w_name.name.split('/')[0] # Get layer name part
                                if grads[i] is not None:
                                    if layer_name not in self.current_epoch_gradients:
                                        self.current_epoch_gradients[layer_name] = []
                                    self.current_epoch_gradients[layer_name].append(grads[i].numpy().flatten())
                    
                    if self.current_epoch_gradients:
                        self.all_gradients_history[epoch] = self.current_epoch_gradients.copy()
                        self.gui.status_bar.showMessage(f"Epoch {epoch+1}: Gradients captured for some layers.", 1000)
                    else:
                        self.gui.status_bar.showMessage(f"Epoch {epoch+1}: Could not capture gradients (setup needed).", 1000)


    def plot_gradient_histograms_from_callback(self): # Now a method of MLCourseGUI
        if not self.gradient_tracker_callback or not self.gradient_tracker_callback.all_gradients_history:
            self.show_error("No gradient history captured. Train a model with the GradientTrackerCallback first.")
            return

        history_data = self.gradient_tracker_callback.all_gradients_history
        
        # Plot gradients from the last recorded epoch
        if not history_data:
            self.metrics_text.append("\nNo gradient data available.")
            return # Added return here for clarity
            
        last_epoch_with_data = max(history_data.keys())
        gradients_at_epoch = history_data[last_epoch_with_data]

        if not gradients_at_epoch:
            self.metrics_text.append(f"\nNo gradient data for epoch {last_epoch_with_data}.")
            return # Added return here

        self.figure.clear()
        num_layers_with_grads = len(gradients_at_epoch)
        if num_layers_with_grads == 0:
             self.figure.text(0.5, 0.5, 'No gradient data captured for plotting.', ha='center', va='center')
             self.canvas.draw()
             return

        # Determine grid size for subplots
        cols = int(np.ceil(np.sqrt(num_layers_with_grads)))
        rows = int(np.ceil(num_layers_with_grads / cols))
        
        fig_title = f"Weight Gradient Histograms (Epoch {last_epoch_with_data + 1})"
        self.figure.suptitle(fig_title, fontsize=14)

        plot_idx = 1
        for layer_name, grad_list_for_layer in gradients_at_epoch.items():
            if not grad_list_for_layer: continue
            
            ax = self.figure.add_subplot(rows, cols, plot_idx)
            all_layer_grads = np.concatenate([g.flatten() for g in grad_list_for_layer if g is not None])
            
            if all_layer_grads.size > 0:
                ax.hist(all_layer_grads, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                mean_grad = np.mean(all_layer_grads)
                std_grad = np.std(all_layer_grads)
                ax.set_title(f"{layer_name}\nMean:{mean_grad:.2e}, Std:{std_grad:.2e}", fontsize=8)
                ax.tick_params(axis='x', labelsize=7)
                ax.tick_params(axis='y', labelsize=7)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f"{layer_name}", fontsize=8)

            plot_idx += 1
        
        try:
            self.figure.tight_layout(rect=[0, 0, 1, 0.96]) 
        except Exception:
            pass 
        self.canvas.draw()
        self.metrics_text.append(f"\nPlotted gradient histograms for epoch {last_epoch_with_data + 1}.")
        self.status_bar.showMessage("Gradient histograms plotted.") # This line uses self.status_bar

    @pyqtSlot()
    def _dataset_selection_changed(self):
        """Handle built-in dataset selection."""
        if self.dataset_combo.currentText() != "Load Custom Dataset":
            self.load_builtin_dataset()

    def load_builtin_dataset(self):
        """Load selected built-in dataset (California Housing check)"""
        try:
            dataset_name = self.dataset_combo.currentText().split(" (")[0] # Get name before parenthesis
            self.status_bar.showMessage(f"Loading {dataset_name}...")
            QApplication.processEvents()

            if dataset_name == "Iris Dataset":
                data = datasets.load_iris(as_frame=True)
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer(as_frame=True)
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits(as_frame=True)
            elif dataset_name == "California Housing Dataset":
                try:
                    data = datasets.fetch_california_housing(as_frame=True)
                    # Ensure 'target' column exists for consistency
                    if 'MedHouseVal' in data.frame.columns:
                         data.frame['target'] = data.frame['MedHouseVal']
                         data.frame = data.frame.drop('MedHouseVal', axis=1)
                    else:
                        # Fallback if structure is different, assuming last column is target
                        target_col_name = data.feature_names[-1] # Example, might need adjustment
                        data.frame['target'] = data.frame[target_col_name]
                        data.frame = data.frame.drop(target_col_name, axis=1)

                    # Update self.original_data definition for consistency
                    # We need both features and target in the frame for processing
                    # The 'data' and 'target' attributes might get separated later
                    # Let's keep the combined frame for now.
                except ImportError:
                    self.show_error("California Housing dataset requires scikit-learn >= 0.22.")
                    self.status_bar.showMessage("Dataset loading failed.")
                    return
                except Exception as fetch_error:
                    self.show_error(f"Failed to load California Housing: {fetch_error}")
                    self.status_bar.showMessage("Dataset loading failed.")
                    return
            else:
                self.show_error(f"Dataset '{dataset_name}' loading not fully implemented.")
                self.status_bar.showMessage("Dataset loading failed.")
                return

            # Ensure data.frame exists and contains target
            if not hasattr(data, 'frame'):
                 # Create frame if not loaded as such (e.g., older sklearn versions)
                 all_data = np.c_[data.data, data.target]
                 columns = getattr(data, 'feature_names', [f'feature_{i}' for i in range(data.data.shape[1])]) + ['target']
                 data.frame = pd.DataFrame(all_data, columns=columns)

            self.original_data = data.frame
            self.status_bar.showMessage(f"Loaded {self.dataset_combo.currentText()}. Click 'Process and Split Data'.")
            self.clear_results()
            # Clear target name from previous custom loads
            if hasattr(self, 'target_column_name'):
                del self.target_column_name

        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")
            self.status_bar.showMessage("Dataset loading failed.")
            self.original_data = None
            self.clear_data()

    def load_custom_data(self):
        
        try:
            file_name, _ = QFileDialog.getOpenFileName(self, "Load Custom CSV Dataset", "", "CSV files (*.csv)")
            if not file_name:
                self.status_bar.showMessage("Custom data loading cancelled.")
                return

            self.status_bar.showMessage(f"Loading custom data from {file_name}...")
            QApplication.processEvents()
            data = pd.read_csv(file_name)

            target_col = self.select_target_column(data.columns)
            if not target_col:
                self.status_bar.showMessage("Custom data loading cancelled (no target selected).")
                self.original_data = None
                return

            self.original_data = data
            self.target_column_name = target_col # Store the target column name
            self.status_bar.showMessage(f"Loaded custom dataset: {file_name}. Target: '{target_col}'. Click 'Process and Split Data'.")
            self.clear_results()

        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")
            self.status_bar.showMessage("Custom data loading failed.")
            self.original_data = None
            self.clear_data()

    def select_target_column(self, columns):
        """Dialog to select target column from dataset (Unchanged)"""
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
        
        method = self.missing_values_combo.currentText()
        self.status_bar.showMessage(f"Applying {method}...")
        QApplication.processEvents()
        original_shape = data.shape
        try:
            numeric_cols = data.select_dtypes(include=np.number).columns
            if method == "Mean Imputation":
                if len(numeric_cols) > 0:
                    imputer = SimpleImputer(strategy='mean')
                    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
            elif method == "Median Imputation":
                 if len(numeric_cols) > 0:
                    imputer = SimpleImputer(strategy='median')
                    data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
            elif method == "Forward Fill":
                data.ffill(inplace=True)
            elif method == "Backward Fill":
                data.bfill(inplace=True)
            elif method == "Drop Rows":
                data.dropna(inplace=True)

            if method != "No Handling" and data.isnull().sum().sum() > 0:
                 # Check numeric columns specifically, as non-numeric might still have NaNs after mean/median
                 if data[numeric_cols].isnull().sum().sum() > 0:
                     self.show_error(f"Warning: Numeric NaNs may still remain after {method}. Check data or try dropping.")

            self.status_bar.showMessage(f"Applied {method}. Shape change: {original_shape} -> {data.shape}", 5000)
            return data
        except Exception as e:
            self.show_error(f"Error during missing value handling ({method}): {str(e)}")
            self.status_bar.showMessage(f"Missing value handling failed for {method}.")
            return None

    def process_and_split_data(self):
        """Handle preprocessing and split data (Train/Test only for now)"""
        if self.original_data is None:
            self.show_error("No data loaded. Please load a dataset first.")
            return

        try:
            self.status_bar.showMessage("Processing data...")
            QApplication.processEvents()

            data_processed = self.original_data.copy()

            # 1. Handle Missing Values
            data_processed = self.handle_missing_values(data_processed)
            if data_processed is None:
                self.clear_data()
                return
            if data_processed.empty:
                 self.show_error("Data is empty after handling missing values. Check data or imputation method.")
                 self.clear_data()
                 return

            # 2. Identify Features (X) and Target (y)
            target_col = None
            if hasattr(self, 'target_column_name'): # Custom data
                 target_col = self.target_column_name
            elif 'target' in data_processed.columns: # Built-in data convention
                 target_col = 'target'

            if not target_col or target_col not in data_processed.columns:
                 self.show_error(f"Target column ('{target_col or 'target'}') not found after preprocessing.")
                 self.clear_data()
                 return

            self.y = data_processed[target_col]
            self.X = data_processed.drop(target_col, axis=1)

            if self.X.empty:
                 self.show_error("No feature columns remaining after selecting target.")
                 self.clear_data()
                 return

            # --- Handle non-numeric features ---
            numeric_cols = self.X.select_dtypes(include=np.number).columns
            non_numeric_cols = self.X.select_dtypes(exclude=np.number).columns

            if len(non_numeric_cols) > 0:
                self.status_bar.showMessage(f"Non-numeric columns found: {list(non_numeric_cols)}. Attempting One-Hot Encoding...")
                QApplication.processEvents()
                try:
                    self.X = pd.get_dummies(self.X, columns=non_numeric_cols, drop_first=True)
                    self.status_bar.showMessage("One-Hot Encoding applied.", 3000)
                    # Ensure all columns are numeric after encoding
                    non_numeric_after = self.X.select_dtypes(exclude=np.number).columns
                    if len(non_numeric_after) > 0:
                        raise ValueError(f"Non-numeric columns still present after OHE: {list(non_numeric_after)}")
                except Exception as encode_err:
                    self.show_error(f"Automatic One-Hot Encoding failed: {encode_err}. "
                                    "Please preprocess data manually or ensure only numeric features remain.")
                    self.clear_data()
                    return

            # --- Convert X to NumPy array for consistency ---
            # Store column names *before* converting to numpy
            self.feature_names = list(self.X.columns)
            self.X = self.X.values
            self.y = self.y.values # Also convert y

            # 3. Split Data (Train/Test)
            test_size = self.split_spin.value()
            self.status_bar.showMessage(f"Splitting data (Test size: {test_size})...")
            QApplication.processEvents()

            if len(self.X) == 0 or len(self.y) == 0:
                 self.show_error("Cannot split data - X or y is empty after preprocessing.")
                 self.clear_data()
                 return
            if len(self.X) != len(self.y):
                 self.show_error(f"Data inconsistency: X length ({len(self.X)}) != y length ({len(self.y)}).")
                 self.clear_data()
                 return

            try:
                 # Check for classification task (heuristic)
                 # Use np.unique on the numpy array self.y
                 is_classification = len(np.unique(self.y)) < 20 and not np.issubdtype(self.y.dtype, np.floating)

                 # Check for minimum samples per class for stratification
                 can_stratify = False
                 if is_classification:
                    unique_classes, counts = np.unique(self.y, return_counts=True)
                    # Need at least 2 samples per class for stratification with test_split > 0
                    if all(counts >= 2):
                        can_stratify = True
                    else:
                        self.status_bar.showMessage("Warning: Not enough samples in some classes for stratification. Splitting without it.", 5000)

                 stratify_param = self.y if is_classification and can_stratify else None

                 self.X_train, self.X_test, self.y_train, self.y_test = \
                    model_selection.train_test_split(self.X, self.y,
                                                     test_size=test_size,
                                                     random_state=42,
                                                     stratify=stratify_param)
            except Exception as split_err:
                 self.show_error(f"Error during data splitting: {split_err}. Trying without stratification.")
                 try:
                     self.X_train, self.X_test, self.y_train, self.y_test = \
                        model_selection.train_test_split(self.X, self.y,
                                                         test_size=test_size,
                                                         random_state=42)
                 except Exception as split_err_again:
                     self.show_error(f"Splitting failed completely: {split_err_again}")
                     self.clear_data()
                     return


            # 4. Apply Scaling (Fit on Train, Transform on Train & Test)
            self.apply_scaling()

            self.status_bar.showMessage(f"Data processed successfully. Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
            self.clear_results() # Clear previous model results

        except Exception as e:
            self.show_error(f"Error processing data: {str(e)}")
            import traceback
            traceback.print_exc() # Print traceback for debugging
            self.status_bar.showMessage("Data processing failed.")
            self.clear_data()

    def apply_scaling(self):
        """Apply selected scaling method to the train/test data (Operates on NumPy arrays)"""
        if self.X_train is None or self.X_test is None:
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
                else: return

                # Fit only on training data
                self.X_train = scaler.fit_transform(self.X_train)
                # Transform both train and test data
                self.X_test = scaler.transform(self.X_test)

                self.status_bar.showMessage(f"Applied {scaling_method} successfully.", 3000)
            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")
                self.status_bar.showMessage(f"Scaling failed for {scaling_method}.")
                # Revert? For now, just report. Consider reloading data or disabling scaling.


    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()
        tabs = [
            ("Supervised Learning", self.create_supervised_ml_tab), # Renamed
            ("Unsupervised & Dim Reduction", self.create_unsupervised_dim_reduction_tab), # Renamed
            ("Deep Learning", self.create_deep_learning_tab),
        ]
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        self.layout.addWidget(self.tab_widget)

    def create_supervised_ml_tab(self):
        """Create the supervised machine learning algorithms tab (Regression + Classification)"""
        widget = QWidget()
        layout = QGridLayout(widget) # Use Grid for 2 columns

        # --- Regression section ---
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()
        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression", {"fit_intercept": ("checkbox", True)}, is_supervised=True)
        regression_layout.addWidget(lr_group)
        # SVR
        svr_group = self.create_algorithm_group(
            "SVR", {"kernel": (["linear", "rbf", "poly"], "rbf"),
                    "C": ("double", 1.0, 0.01, 1000.0, 0.1),
                    "epsilon": ("double", 0.1, 0.0, 10.0, 0.1),
                    "degree": ("int", 3, 2, 10, 1)}, is_supervised=True)
        regression_layout.addWidget(svr_group)
        # Decision Tree Regressor - NEW
        dtr_group = self.create_algorithm_group(
            "Decision Tree Regressor", {
                 "criterion": (["squared_error", "friedman_mse", "absolute_error"], "squared_error"),
                 "max_depth": ("int_optional", 5, 1, 100, 1),
                 "min_samples_split": ("int", 2, 2, 100, 1)}, is_supervised=True)
        regression_layout.addWidget(dtr_group)
        # Random Forest Regressor - NEW
        rfr_group = self.create_algorithm_group(
            "Random Forest Regressor", {
                "n_estimators": ("int", 100, 10, 1000, 10),
                "criterion": (["squared_error", "absolute_error"], "squared_error"), # Removed poisson
                "max_depth": ("int_optional", 10, 1, 100, 1),
                "min_samples_split": ("int", 2, 2, 100, 1)}, is_supervised=True)
        regression_layout.addWidget(rfr_group)
         # KNN Regressor - NEW
        knnr_group = self.create_algorithm_group(
            "KNeighbors Regressor", {
                 "n_neighbors": ("int", 5, 1, 100, 1),
                 "weights": (["uniform", "distance"], "uniform"),
                 "metric": (["euclidean", "manhattan", "minkowski"], "minkowski")}, is_supervised=True)
        regression_layout.addWidget(knnr_group)

        regression_group.setLayout(regression_layout)
        layout.addWidget(regression_group, 0, 0, Qt.AlignmentFlag.AlignTop)

        # --- Classification section ---
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()
        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression", {"C": ("double", 1.0, 0.01, 1000.0, 0.1),
                                    "max_iter": ("int", 100, 50, 10000, 50),
                                    "multi_class": (["ovr", "multinomial"], "ovr"),
                                    "solver": (["liblinear", "lbfgs", "saga"], "lbfgs")}, is_supervised=True)
        classification_layout.addWidget(logistic_group)
        # GaussianNB
        nb_group = self.create_algorithm_group(
            "GaussianNB", {"var_smoothing": ("double", 1e-9, 1e-12, 1e-3, 1e-9)}, is_supervised=True) # Log scale might be better here?
        classification_layout.addWidget(nb_group)
        # SVC
        svm_group = self.create_algorithm_group(
            "SVC", {"C": ("double", 1.0, 0.01, 1000.0, 0.1),
                    "kernel": (["linear", "rbf", "poly", "sigmoid"], "rbf"),
                    "degree": ("int", 3, 2, 10, 1),
                    "gamma": (["scale", "auto"], "scale"),
                    "probability": ("checkbox", True)}, is_supervised=True) # Keep probability=True
        classification_layout.addWidget(svm_group)
        # Decision Tree Classifier
        dt_group = self.create_algorithm_group(
            "Decision Tree Classifier", {"criterion": (["gini", "entropy"], "gini"),
                                         "max_depth": ("int_optional", 5, 1, 100, 1),
                                         "min_samples_split": ("int", 2, 2, 100, 1)}, is_supervised=True)
        classification_layout.addWidget(dt_group)
        # Random Forest Classifier
        rf_group = self.create_algorithm_group(
            "Random Forest Classifier", {"n_estimators": ("int", 100, 10, 1000, 10),
                                         "criterion": (["gini", "entropy"], "gini"),
                                         "max_depth": ("int_optional", 10, 1, 100, 1),
                                         "min_samples_split": ("int", 2, 2, 100, 1)}, is_supervised=True)
        classification_layout.addWidget(rf_group)
        # KNN Classifier
        knn_group = self.create_algorithm_group(
            "KNeighbors Classifier", {"n_neighbors": ("int", 5, 1, 100, 1),
                                      "weights": (["uniform", "distance"], "uniform"),
                                      "metric": (["euclidean", "manhattan", "minkowski"], "minkowski")}, is_supervised=True)
        classification_layout.addWidget(knn_group)

        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1, Qt.AlignmentFlag.AlignTop)

        return widget

    def create_unsupervised_dim_reduction_tab(self):
        """Create the Unsupervised Learning and Dimensionality Reduction tab"""
        widget = QWidget()
        widget.setMinimumWidth(780)
        layout = QGridLayout(widget) # Use Grid for multiple columns if needed

        # --- Clustering section ---
        clustering_group = QGroupBox("Clustering")
        clustering_layout = QVBoxLayout()


        # K-Means
        kmeans_widget = QWidget()
        kmeans_inner_layout = QVBoxLayout(kmeans_widget)
        kmeans_params_group = self.create_algorithm_group(
            "K-Means", {"n_clusters": ("int", 3, 2, 50, 1),
                        "init": (["k-means++", "random"], "k-means++"),
                        "n_init": ("int", 10, 1, 50, 1), # Use 'auto' or int
                        "max_iter": ("int", 300, 50, 1000, 50)}, is_supervised=False) # K-Means itself is unsupervised
        kmeans_inner_layout.addWidget(kmeans_params_group)

        # Elbow Method Button
        elbow_layout = QHBoxLayout()
        elbow_layout.addWidget(QLabel("Max k for Elbow:"))
        self.kmeans_elbow_k_spin = QSpinBox()
        self.kmeans_elbow_k_spin.setRange(5, 50); self.kmeans_elbow_k_spin.setValue(10)
        elbow_btn = QPushButton("Plot Elbow Method")
        elbow_btn.clicked.connect(self.run_kmeans_elbow)
        elbow_layout.addWidget(self.kmeans_elbow_k_spin)
        elbow_layout.addWidget(elbow_btn)
        kmeans_inner_layout.addLayout(elbow_layout)

        clustering_layout.addWidget(kmeans_widget)
        clustering_group.setLayout(clustering_layout)
        layout.addWidget(clustering_group, 0, 0, Qt.AlignmentFlag.AlignTop)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)


        # --- Dimensionality Reduction section ---
        dim_reduction_group = QGroupBox("Dimensionality Reduction")
        dim_reduction_layout = QVBoxLayout()

        # PCA
        pca_group = self.create_algorithm_group(
            "PCA", {"n_components": ("int_optional", 2, 1, 50, 1),
                    "whiten": ("checkbox", False)}, is_supervised=False)
        dim_reduction_layout.addWidget(pca_group)

        # LDA - Supervised
        lda_group = self.create_algorithm_group(
            "LDA", {"n_components": ("int_optional", None, 1, 50, 1), # Can be None (auto)
                    "solver": (["svd", "lsqr", "eigen"], "svd")},
             is_supervised=True) # LDA needs y
        dim_reduction_layout.addWidget(lda_group)

        # t-SNE
        tsne_group = self.create_algorithm_group(
            "t-SNE", {"n_components": ("int", 2, 1, 3, 1), # Usually 2 or 3
                      "perplexity": ("double", 30.0, 2.0, 100.0, 1.0),
                      "learning_rate": (["auto"] + [str(lr) for lr in [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]], "auto"),
                      "n_iter": ("int", 1000, 250, 5000, 250),
                      "init": (["random", "pca"], "pca")}, is_supervised=False)
        dim_reduction_layout.addWidget(tsne_group)

        # UMAP (Conditional on import)
        if HAS_UMAP:
            umap_group = self.create_algorithm_group(
                "UMAP", {"n_components": ("int", 2, 1, 5, 1), # Usually 2 or 3
                         "n_neighbors": ("int", 15, 2, 100, 1),
                         "min_dist": ("double", 0.1, 0.0, 0.99, 0.1)}, is_supervised=False)
            dim_reduction_layout.addWidget(umap_group)
        else:
            dim_reduction_layout.addWidget(QLabel("UMAP disabled (umap-learn not installed)"))


        dim_reduction_group.setLayout(dim_reduction_layout)
        layout.addWidget(dim_reduction_group, 0, 1, Qt.AlignmentFlag.AlignTop)

        return widget

    def create_deep_learning_tab(self):
        """Create the deep learning tab with enhanced features."""
        widget = QWidget()
        main_layout = QVBoxLayout(widget) # Main vertical layout for the tab

        # --- Architecture Configuration Section ---
        arch_group = QGroupBox("Neural Network Architecture")
        arch_layout = QGridLayout() # Use grid for better layout

        arch_layout.addWidget(QLabel("Current Layers:"), 0, 0, 1, 2)
        self.layer_list_widget = QTextEdit()
        self.layer_list_widget.setReadOnly(True)
        self.layer_list_widget.setFixedHeight(150) # Increased height
        self._update_layer_display()
        arch_layout.addWidget(self.layer_list_widget, 1, 0, 1, 2)

        add_layer_btn = QPushButton("Add Layer")
        add_layer_btn.clicked.connect(self.add_layer_dialog_detailed) # Changed to new dialog
        remove_last_layer_btn = QPushButton("Remove Last Layer")
        remove_last_layer_btn.clicked.connect(self._remove_last_layer)
        clear_layers_btn = QPushButton("Clear All Layers")
        clear_layers_btn.clicked.connect(self._clear_layers)
        
        arch_layout.addWidget(add_layer_btn, 2, 0)
        arch_layout.addWidget(remove_last_layer_btn, 2, 1)
        arch_layout.addWidget(clear_layers_btn, 3, 0, 1, 2)

        save_arch_btn = QPushButton("Save Architecture")
        save_arch_btn.clicked.connect(self.save_nn_architecture)
        load_arch_btn = QPushButton("Load Architecture")
        load_arch_btn.clicked.connect(self.load_nn_architecture)
        arch_layout.addWidget(save_arch_btn, 4, 0)
        arch_layout.addWidget(load_arch_btn, 4, 1)

        arch_group.setLayout(arch_layout)
        main_layout.addWidget(arch_group)

        # --- Training Configuration Section ---
        train_config_group = QGroupBox("Training Configuration")
        train_config_main_layout = QVBoxLayout(train_config_group)

        # Task Type and Loss
        task_loss_layout = QHBoxLayout()
        task_loss_layout.addWidget(QLabel("Task Type:"))
        self.nn_task_combo = QComboBox()
        self.nn_task_combo.addItems(["Classification", "Regression"])
        self.nn_task_combo.currentTextChanged.connect(self._update_loss_options)
        task_loss_layout.addWidget(self.nn_task_combo)
        task_loss_layout.addWidget(QLabel("Loss Function:"))
        self.nn_loss_combo = QComboBox()
        self._update_loss_options(self.nn_task_combo.currentText()) # Initial population
        task_loss_layout.addWidget(self.nn_loss_combo)
        train_config_main_layout.addLayout(task_loss_layout)

        # Optimizer
        opt_layout = QHBoxLayout()
        opt_layout.addWidget(QLabel("Optimizer:"))
        self.nn_optimizer_combo = QComboBox()
        self.nn_optimizer_combo.addItems(["Adam", "SGD", "RMSprop"])
        self.nn_optimizer_combo.currentTextChanged.connect(self._update_optimizer_params_display)
        opt_layout.addWidget(self.nn_optimizer_combo)
        train_config_main_layout.addLayout(opt_layout)

        # Optimizer-specific parameters (placeholder for dynamic widgets)
        self.optimizer_params_widget = QWidget() # Container for dynamic optimizer params
        self.optimizer_params_layout = QGridLayout(self.optimizer_params_widget)
        train_config_main_layout.addWidget(self.optimizer_params_widget)
        self._update_optimizer_params_display("Adam") # Initialize for Adam

        # General Training Params (Epochs, Batch Size, LR)
        train_params_grid = QGridLayout()
        train_params_grid.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 10000); self.epochs_spin.setValue(50)
        train_params_grid.addWidget(self.epochs_spin, 0, 1)

        train_params_grid.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(1, 4096); self.batch_size_spin.setValue(32)
        train_params_grid.addWidget(self.batch_size_spin, 1, 1)
        
        train_params_grid.addWidget(QLabel("Initial Learning Rate:"), 0, 2)
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setRange(1e-7, 1.0); self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.0001); self.lr_spin.setDecimals(7)
        train_params_grid.addWidget(self.lr_spin, 0, 3)
        train_config_main_layout.addLayout(train_params_grid)

        # Callbacks: Early Stopping, LR Scheduler
        callbacks_group = QGroupBox("Callbacks")
        callbacks_layout = QGridLayout(callbacks_group)

        self.early_stopping_cb = QCheckBox("Enable Early Stopping")
        callbacks_layout.addWidget(self.early_stopping_cb, 0, 0)
        callbacks_layout.addWidget(QLabel("Patience:"), 0, 1)
        self.early_stopping_patience_spin = QSpinBox(); self.early_stopping_patience_spin.setRange(1, 100); self.early_stopping_patience_spin.setValue(10)
        callbacks_layout.addWidget(self.early_stopping_patience_spin, 0, 2)

        callbacks_layout.addWidget(QLabel("LR Scheduler:"), 1, 0)
        self.lr_scheduler_combo = QComboBox()
        self.lr_scheduler_combo.addItems(["None", "Step Decay", "Exponential Decay"])
        callbacks_layout.addWidget(self.lr_scheduler_combo, 1, 1, 1, 2)
        # Placeholder for LR scheduler params if needed
        train_config_main_layout.addWidget(callbacks_group)
        
        # --- Image Augmentation Section (Optional) ---
        self.image_augmentation_group = QGroupBox("Image Augmentation (for 2D Image Data)")
        aug_layout = QGridLayout(self.image_augmentation_group)
        self.aug_rotation_cb = QCheckBox("Random Rotation (e.g., 20 deg)")
        self.aug_rotation_spin = QSpinBox(); self.aug_rotation_spin.setRange(0, 90); self.aug_rotation_spin.setValue(20)
        self.aug_flip_cb = QCheckBox("Random Horizontal Flip")
        self.aug_zoom_cb = QCheckBox("Random Zoom (e.g., 0.2)")
        self.aug_zoom_spin = QDoubleSpinBox(); self.aug_zoom_spin.setRange(0.0, 0.5); self.aug_zoom_spin.setValue(0.2)

        aug_layout.addWidget(self.aug_rotation_cb, 0, 0); aug_layout.addWidget(self.aug_rotation_spin, 0, 1)
        aug_layout.addWidget(self.aug_flip_cb, 1, 0)
        aug_layout.addWidget(self.aug_zoom_cb, 2, 0); aug_layout.addWidget(self.aug_zoom_spin, 2, 1)
        self.image_augmentation_group.setCheckable(True) # Allow enabling/disabling the whole group
        self.image_augmentation_group.setChecked(False)
        train_config_main_layout.addWidget(self.image_augmentation_group)

        train_nn_btn = QPushButton("Train Neural Network")
        train_nn_btn.clicked.connect(self.train_neural_network_enhanced) # Changed to new training func
        train_config_main_layout.addWidget(train_nn_btn)
        main_layout.addWidget(train_config_group)

        # --- Pre-trained Models Section ---
        pretrained_group = QGroupBox("Pre-trained Models (Transfer Learning for Images)")
        pretrained_layout = QGridLayout(pretrained_group)

        pretrained_layout.addWidget(QLabel("Base Model:"), 0, 0)
        self.pretrained_model_combo = QComboBox()
        self.pretrained_model_combo.addItems(["None", "VGG16", "ResNet50", "MobileNetV2"])
        pretrained_layout.addWidget(self.pretrained_model_combo, 0, 1)

        self.finetune_cb = QCheckBox("Fine-tune (unfreeze some top layers of base model)")
        pretrained_layout.addWidget(self.finetune_cb, 1, 0, 1, 2)
        pretrained_layout.addWidget(QLabel("Layers to unfreeze from top:"), 2, 0)
        self.finetune_layers_spin = QSpinBox(); self.finetune_layers_spin.setRange(0, 50); self.finetune_layers_spin.setValue(5)
        pretrained_layout.addWidget(self.finetune_layers_spin, 2, 1)
        
        pretrained_group.setCheckable(True) # Allow enabling/disabling the whole group
        pretrained_group.setChecked(False) # Default to not using pre-trained
        pretrained_group.toggled.connect(self._toggle_pretrained_options)
        self._toggle_pretrained_options(False) # Initialize disabled state

        main_layout.addWidget(pretrained_group)
        
        # Gradient Histogram Button
        self.plot_gradients_btn = QPushButton("Plot Last Gradients (after training epoch)")
        self.plot_gradients_btn.clicked.connect(self.plot_gradient_histograms_from_callback)
        self.plot_gradients_btn.setEnabled(False) # Enabled after training with callback
        main_layout.addWidget(self.plot_gradients_btn)
        
        self.gradient_tracker_callback = None # To store gradient data

        return widget

    def _toggle_pretrained_options(self, checked):
        """Enable/disable fine-tuning options based on pre-trained group state."""
        self.finetune_cb.setEnabled(checked)
        self.finetune_layers_spin.setEnabled(checked and self.finetune_cb.isChecked())
        if checked:
            self.status_bar.showMessage("Pre-trained model mode activated. Layer config will be ignored if a base model is selected.", 5000)
            # Disable manual layer adding if pre-trained model is active?
            # For now, just a warning. User should use one or the other.
        else:
            self.status_bar.showMessage("Manual architecture mode activated.", 3000)

    def _update_optimizer_params_display(self, optimizer_name):
        """Dynamically update UI for optimizer-specific parameters."""
        # Clear previous params
        for i in reversed(range(self.optimizer_params_layout.count())):
            widget_to_remove = self.optimizer_params_layout.itemAt(i).widget()
            self.optimizer_params_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

        if optimizer_name == "Adam":
            # Adam has beta_1, beta_2, epsilon, amsgrad
            self.adam_beta1_spin = QDoubleSpinBox(); self.adam_beta1_spin.setRange(0.0, 0.999); self.adam_beta1_spin.setValue(0.9); self.adam_beta1_spin.setDecimals(3)
            self.adam_beta2_spin = QDoubleSpinBox(); self.adam_beta2_spin.setRange(0.0, 0.9999); self.adam_beta2_spin.setValue(0.999); self.adam_beta2_spin.setDecimals(4)
            self.adam_epsilon_spin = QDoubleSpinBox(); self.adam_epsilon_spin.setRange(1e-8, 1e-3); self.adam_epsilon_spin.setValue(1e-7); self.adam_epsilon_spin.setDecimals(8)
            self.adam_amsgrad_cb = QCheckBox("AMSGrad")

            self.optimizer_params_layout.addWidget(QLabel("Beta 1:"), 0, 0); self.optimizer_params_layout.addWidget(self.adam_beta1_spin, 0, 1)
            self.optimizer_params_layout.addWidget(QLabel("Beta 2:"), 0, 2); self.optimizer_params_layout.addWidget(self.adam_beta2_spin, 0, 3)
            self.optimizer_params_layout.addWidget(QLabel("Epsilon:"), 1, 0); self.optimizer_params_layout.addWidget(self.adam_epsilon_spin, 1, 1)
            self.optimizer_params_layout.addWidget(self.adam_amsgrad_cb, 1, 2)

        elif optimizer_name == "SGD":
            # SGD has momentum, nesterov
            self.sgd_momentum_spin = QDoubleSpinBox(); self.sgd_momentum_spin.setRange(0.0, 0.99); self.sgd_momentum_spin.setValue(0.0); self.sgd_momentum_spin.setDecimals(2)
            self.sgd_nesterov_cb = QCheckBox("Nesterov Momentum")
            self.optimizer_params_layout.addWidget(QLabel("Momentum:"), 0, 0); self.optimizer_params_layout.addWidget(self.sgd_momentum_spin, 0, 1)
            self.optimizer_params_layout.addWidget(self.sgd_nesterov_cb, 0, 2)

        elif optimizer_name == "RMSprop":
            # RMSprop has rho, momentum, epsilon
            self.rmsprop_rho_spin = QDoubleSpinBox(); self.rmsprop_rho_spin.setRange(0.0, 0.99); self.rmsprop_rho_spin.setValue(0.9); self.rmsprop_rho_spin.setDecimals(2)
            self.rmsprop_momentum_spin = QDoubleSpinBox(); self.rmsprop_momentum_spin.setRange(0.0, 0.99); self.rmsprop_momentum_spin.setValue(0.0); self.rmsprop_momentum_spin.setDecimals(2)
            self.rmsprop_epsilon_spin = QDoubleSpinBox(); self.rmsprop_epsilon_spin.setRange(1e-8, 1e-3); self.rmsprop_epsilon_spin.setValue(1e-7); self.rmsprop_epsilon_spin.setDecimals(8)

            self.optimizer_params_layout.addWidget(QLabel("Rho:"), 0, 0); self.optimizer_params_layout.addWidget(self.rmsprop_rho_spin, 0, 1)
            self.optimizer_params_layout.addWidget(QLabel("Momentum:"), 0, 2); self.optimizer_params_layout.addWidget(self.rmsprop_momentum_spin, 0, 3)
            self.optimizer_params_layout.addWidget(QLabel("Epsilon:"), 1, 0); self.optimizer_params_layout.addWidget(self.rmsprop_epsilon_spin, 1, 1)

    def create_cross_validation_section(self):
        """Creates a dedicated section for cross-validation controls below the tabs"""
        cv_group = QGroupBox("Model Evaluation: Cross-Validation")
        cv_layout = QGridLayout(cv_group)

        cv_layout.addWidget(QLabel("Note: Uses the last trained model's type and parameters from the 'Supervised Learning' tab."), 0, 0, 1, 4)
        cv_layout.addWidget(QLabel("Number of Folds (k):"), 1, 0)

        self.cv_k_spin = QSpinBox()
        self.cv_k_spin.setRange(2, 20) # Sensible range for k
        self.cv_k_spin.setValue(5)
        cv_layout.addWidget(self.cv_k_spin, 1, 1)

        self.run_cv_btn = QPushButton("Run k-Fold Cross-Validation")
        self.run_cv_btn.clicked.connect(self.run_cross_validation)
        cv_layout.addWidget(self.run_cv_btn, 1, 2, 1, 2) # Span 2 columns

        self.layout.addWidget(cv_group) # Add this section to the main layout

    def create_visualization(self):
        """Create the visualization and metrics section (Adjust size)"""
        viz_metrics_group = QGroupBox("Results: Visualization and Metrics")
        viz_metrics_layout = QHBoxLayout()

        # Matplotlib Figure
        self.figure = Figure(figsize=(8, 7)) # Slightly larger figure
        self.canvas = FigureCanvas(self.figure)
        viz_metrics_layout.addWidget(self.canvas, 3) # More stretch factor

        # Metrics Display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMinimumWidth(350) # Wider metrics area
        viz_metrics_layout.addWidget(self.metrics_text, 1) # Less stretch factor

        viz_metrics_group.setLayout(viz_metrics_layout)
        self.layout.addWidget(viz_metrics_group)

    def create_status_bar(self):
        """Create the status bar (Unchanged)"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setTextVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_bar.showMessage("Ready")

    def create_misc_section(self):
        """ Create section for miscellaneous actions like the eigenvector example """
        misc_group = QGroupBox("Examples & Tools")
        misc_layout = QHBoxLayout(misc_group)

        eigen_btn = QPushButton("Show Eigenvector Example (1D Projection)")
        eigen_btn.clicked.connect(self.show_eigenvector_example)
        misc_layout.addWidget(eigen_btn)

        misc_layout.addStretch() # Push button to the left

        self.layout.addWidget(misc_group)


    def create_algorithm_group(self, name, params_config, is_supervised):
        """
        Helper method to create algorithm parameter groups.
        Adds a Train button specific to this algorithm.
        `is_supervised` flag indicates if the model requires y.
        """
        group = QGroupBox(name)
        layout = QVBoxLayout()
        param_widgets = {}

        for param_name, config in params_config.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name.replace('_', ' ').title()}:"))

            widget = None
            param_type = config[0]
            default_value = config[1] if len(config) > 1 else None

            # --- Widget Creation (Handles int, int_optional, double, checkbox, list) ---
            # (This part is the same as the original function - keep it)
            if param_type == "int":
                widget = QSpinBox()
                if len(config) > 4: widget.setRange(config[2], config[3]); widget.setSingleStep(config[4])
                else: widget.setRange(-99999, 99999)
                if default_value is not None: widget.setValue(default_value)
            elif param_type == "int_optional":
                 widget = QWidget()
                 h_layout = QHBoxLayout(widget); h_layout.setContentsMargins(0,0,0,0)
                 num_widget = QSpinBox()
                 if len(config) > 4: num_widget.setRange(config[2], config[3]); num_widget.setSingleStep(config[4])
                 else: num_widget.setRange(-99999, 99999)
                 if default_value is not None: num_widget.setValue(default_value)
                 cb_widget = QCheckBox("None")
                 cb_widget.setChecked(default_value is None)
                 num_widget.setEnabled(not cb_widget.isChecked())
                 cb_widget.toggled.connect(num_widget.setDisabled)
                 h_layout.addWidget(num_widget); h_layout.addWidget(cb_widget)
                 param_widgets[param_name] = (num_widget, cb_widget) # Store tuple
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setDecimals(9) # Increased precision for smoothing, LR etc.
                if len(config) > 4: widget.setRange(config[2], config[3]); widget.setSingleStep(config[4])
                else: widget.setRange(-99999.0, 99999.0)
                if default_value is not None: widget.setValue(default_value)
            elif param_type == "checkbox":
                widget = QCheckBox()
                if default_value is not None: widget.setChecked(default_value)
            elif isinstance(param_type, list): # ComboBox
                widget = QComboBox()
                widget.addItems(param_type)
                if default_value is not None and default_value in param_type:
                    widget.setCurrentText(default_value)
            # --- End Widget Creation ---

            if widget is not None:
                 param_layout.addWidget(widget)
                 if param_type != "int_optional":
                     param_widgets[param_name] = widget # Store simple widget

            layout.addLayout(param_layout)

        # Add Train/Apply button
        # Use "Apply" for unsupervised/transformers, "Train" for supervised
        button_text = "Train" if is_supervised else "Apply"
        train_btn = QPushButton(f"{button_text} {name}")
        train_btn.clicked.connect(lambda checked=False, n=name, p=param_widgets, sup=is_supervised: self.run_single_model(n, p, sup))
        layout.addWidget(train_btn)

        group.setLayout(layout)
        return group

    def _get_params_from_widgets(self, param_widgets):
        """Extracts parameters from the widget dictionary."""
        params = {}
        for name, widget_or_tuple in param_widgets.items():
            if isinstance(widget_or_tuple, tuple) and isinstance(widget_or_tuple[1], QCheckBox): # int_optional
                num_widget, cb_widget = widget_or_tuple
                params[name] = None if cb_widget.isChecked() else num_widget.value()
            elif isinstance(widget_or_tuple, QSpinBox): params[name] = widget_or_tuple.value()
            elif isinstance(widget_or_tuple, QDoubleSpinBox): params[name] = widget_or_tuple.value()
            elif isinstance(widget_or_tuple, QCheckBox): params[name] = widget_or_tuple.isChecked()
            elif isinstance(widget_or_tuple, QComboBox): params[name] = widget_or_tuple.currentText()
            elif isinstance(widget_or_tuple, QLineEdit): params[name] = widget_or_tuple.text() # Added for potential future use
        return params

    def run_single_model(self, model_name, param_widgets, is_supervised):
        """
        Train or apply a single selected model (handles supervised, unsupervised, transformers).
        Replaces the old `train_model`.
        """
        if self.X_train is None:
            self.show_error("Data not processed or split yet. Please process data first.")
            return
        if is_supervised and self.y_train is None:
             self.show_error("Supervised model requires target variable (y_train), which is missing.")
             return
        # Special check for LDA needing y
        if model_name == "LDA" and self.y_train is None:
             self.show_error("LDA is supervised and requires the target variable (y). Please process data including target.")
             return

        try:
            action = "Training" if is_supervised else "Applying"
            self.status_bar.showMessage(f"{action} {model_name}...")
            self.progress_bar.setRange(0, 0) # Indeterminate progress
            QApplication.processEvents()

            params = self._get_params_from_widgets(param_widgets)
            self.last_trained_params = params.copy() # Store params for potential CV use
            self.current_model_name = model_name # Store name

            # --- Instantiate the correct model ---
            model = None
            task_type = 'unknown'

            # == Supervised Models ==
            if model_name == "Linear Regression":
                model = LinearRegression(**params); task_type = 'regression'
            elif model_name == "SVR":
                model = SVR(**params); task_type = 'regression'
            elif model_name == "Decision Tree Regressor":
                model = DecisionTreeRegressor(**params, random_state=42); task_type = 'regression'
            elif model_name == "Random Forest Regressor":
                model = RandomForestRegressor(**params, random_state=42, n_jobs=-1); task_type = 'regression'
            elif model_name == "KNeighbors Regressor":
                model = KNeighborsRegressor(**params, n_jobs=-1); task_type = 'regression'
            elif model_name == "Logistic Regression":
                if params.get('solver') == 'liblinear' and params.get('multi_class') == 'multinomial':
                    params['solver'] = 'lbfgs'; self.status_bar.showMessage("Warning: Switched solver to lbfgs for multinomial.", 4000)
                # Ensure probability is boolean if present (SVC uses it)
                if 'probability' in params: del params['probability']
                model = LogisticRegression(**params, random_state=42); task_type = 'classification'
            elif model_name == "GaussianNB":
                model = GaussianNB(**params); task_type = 'classification'
            elif model_name == "SVC":
                # Keras passes probability param, ensure it's handled
                prob_param = params.get('probability', False) # Default to False if not in widgets
                clean_params = {k: v for k, v in params.items() if k != 'probability'}
                model = SVC(**clean_params, probability=prob_param, random_state=42)
                task_type = 'classification'
            elif model_name == "Decision Tree Classifier":
                model = DecisionTreeClassifier(**params, random_state=42); task_type = 'classification'
            elif model_name == "Random Forest Classifier":
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1); task_type = 'classification'
            elif model_name == "KNeighbors Classifier":
                model = KNeighborsClassifier(**params, n_jobs=-1); task_type = 'classification'

            # == Unsupervised / Dimensionality Reduction Models ==
            elif model_name == "K-Means":
                 # Ensure n_init is handled correctly (might be str 'auto' or int)
                 if 'n_init' in params:
                     try: params['n_init'] = int(params['n_init'])
                     except ValueError: params['n_init'] = 'auto' # Or a default int like 10

                 model = KMeans(**params, random_state=42)
                 task_type = 'clustering'
            elif model_name == "PCA":
                 model = PCA(**params)
                 task_type = 'dim_reduction'
            elif model_name == "LDA": # Supervised dim reduction
                 # n_components for LDA cannot be larger than min(n_features, n_classes - 1)
                 n_features = self.X_train.shape[1]
                 n_classes = len(np.unique(self.y_train))
                 max_lda_components = min(n_features, n_classes - 1)
                 if params.get('n_components') is not None and params['n_components'] > max_lda_components:
                     self.status_bar.showMessage(f"Warning: LDA n_components reduced to {max_lda_components} (max allowed).", 5000)
                     params['n_components'] = max_lda_components
                 elif params.get('n_components') == 0: # Allow 0 to mean None
                     params['n_components'] = None

                 model = LinearDiscriminantAnalysis(**params)
                 task_type = 'dim_reduction' # Treat as dim reduction for plotting/metrics here
            elif model_name == "t-SNE":
                 # Handle potential string 'auto' for learning rate
                 if 'learning_rate' in params and params['learning_rate'] == 'auto':
                     pass # Keep it as 'auto'
                 elif 'learning_rate' in params:
                     try: params['learning_rate'] = float(params['learning_rate'])
                     except ValueError: params['learning_rate'] = 'auto' # Fallback

                 model = TSNE(**params, random_state=42)
                 task_type = 'dim_reduction' # Embedding
            elif model_name == "UMAP" and HAS_UMAP:
                 model = umap.UMAP(**params, random_state=42)
                 task_type = 'dim_reduction' # Embedding
            else:
                self.show_error(f"Model '{model_name}' logic not implemented.")
                self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
                self.status_bar.showMessage(f"{action} failed.")
                return

            self.current_model = model # Store the instantiated model
            self.last_task_type = task_type # Store task type for CV

            # --- Fit the model ---
            y_pred = None
            X_transformed = None # For storing transformed data from dim reduction

            self.status_bar.showMessage(f"Fitting {model_name} on training data...")
            QApplication.processEvents()

            if task_type in ['regression', 'classification']:
                model.fit(self.X_train, self.y_train)
                if hasattr(model, "predict"):
                     y_pred = model.predict(self.X_test)
            elif task_type == 'clustering':
                 # Fit on training data, predict on test data
                 model.fit(self.X_train) # Fit K-Means
                 y_pred = model.predict(self.X_test) # Assign test points to clusters
            elif task_type == 'dim_reduction':
                 # Fit on training data, transform both train and test
                 if model_name == "LDA":
                     # LDA needs y for fitting
                     model.fit(self.X_train, self.y_train)
                 else:
                     # PCA, t-SNE, UMAP fit only on X
                     model.fit(self.X_train) # Fit the transformer

                 # Transform the test data for visualization/evaluation
                 if hasattr(model, "transform"):
                      X_transformed = model.transform(self.X_test)
                 elif hasattr(model, "fit_transform") and model_name in ["t-SNE", "UMAP"]:
                      # t-SNE/UMAP often applied directly via fit_transform.
                      # For consistency, we fit above and transform test here.
                      # Note: Applying fit_transform to test data *separately* isn't standard
                      # as it doesn't use the learned training embedding structure.
                      # It's better to fit on train and transform test.
                      # If transform isn't available (older UMAP versions?), we might need fit_transform on test.
                      try:
                         # Re-fitting on test data is generally discouraged for t-SNE/UMAP eval
                         # We visualize the embedding OF THE TEST SET based on the training fit
                         # If transform exists, use it. If not, warn.
                          X_transformed = model.transform(self.X_test)
                          self.status_bar.showMessage(f"Applied {model_name} transform to test data.", 3000)
                      except AttributeError:
                           # Fallback: Apply fit_transform on test data (less ideal)
                           # self.status_bar.showMessage(f"Warning: {model_name} has no 'transform' method. Applying 'fit_transform' directly to test data for visualization.", 5000)
                           # X_transformed = model.fit_transform(self.X_test)

                           # Better approach: just show the training embedding if transform fails
                           self.status_bar.showMessage(f"Warning: Could not apply {model_name} 'transform' to test data. Visualizing training data embedding.", 5000)
                           if hasattr(model, 'embedding_'):
                                X_transformed = model.embedding_ # Use the embedding learned during fit(X_train)
                           else:
                               X_transformed = model.fit_transform(self.X_train) # Re-run on train if embedding not stored


            # --- Update Results ---
            self.update_metrics(y_pred, X_transformed, task_type) # Pass transformed data too
            self.update_visualization(y_pred, X_transformed, task_type) # Pass transformed data

            self.progress_bar.setRange(0, 100); self.progress_bar.setValue(100)
            self.status_bar.showMessage(f"{model_name} {action} complete.")

        except Exception as e:
            self.show_error(f"Error during {action} of {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
            self.status_bar.showMessage(f"{model_name} {action} failed.")
            self.current_model = None
            self.current_model_name = None
            self.last_task_type = 'unknown'

    def run_kmeans_elbow(self):
        """Runs K-Means for a range of k and plots the inertia (elbow method)."""
        if self.X_train is None:
            self.show_error("Data not processed or split yet. Cannot run Elbow method.")
            return

        try:
            max_k = self.kmeans_elbow_k_spin.value()
            k_range = range(1, max_k + 1) # Check k from 1 up to max_k
            inertias = []

            self.status_bar.showMessage(f"Running K-Means Elbow method (k=1 to {max_k})...")
            self.progress_bar.setRange(0, max_k)
            QApplication.processEvents()

            # Get base params, excluding n_clusters
            kmeans_group = self.findChild(QGroupBox, "K-Means")
            param_widgets = {}
            # Need to find the widgets within the K-Means group correctly
            # Assuming the structure created by create_algorithm_group
            for child in kmeans_group.findChildren(QWidget): # Find all widgets
                obj_name = child.objectName() # We might need to set object names in create_algorithm_group
                # Or iterate layouts and find widgets by type/label association
                # Simpler: Re-fetch widgets using the structure knowledge
                # This is fragile if layout changes, but avoids complex searches
                layouts = kmeans_group.findChildren(QHBoxLayout)
                for layout in layouts:
                    label = layout.itemAt(0).widget()
                    widget = layout.itemAt(1).widget()
                    if isinstance(label, QLabel) and label.text().startswith("N Clusters"): continue # Skip n_clusters
                    if isinstance(label, QLabel):
                         param_name = label.text().replace(':', '').replace(' ', '_').lower()
                         # Handle complex widgets like int_optional if needed here
                         param_widgets[param_name] = widget

            base_params = self._get_params_from_widgets(param_widgets)
            # Ensure n_init is handled
            if 'n_init' in base_params:
                try: base_params['n_init'] = int(base_params['n_init'])
                except ValueError: base_params['n_init'] = 'auto'

            for i, k in enumerate(k_range):
                self.status_bar.showMessage(f"Calculating K-Means for k={k}...")
                QApplication.processEvents()
                kmeans = KMeans(n_clusters=k, **base_params, random_state=42)
                kmeans.fit(self.X_train) # Fit on training data
                inertias.append(kmeans.inertia_)
                self.progress_bar.setValue(i + 1)

            # Plotting the results
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(k_range, inertias, 'bo-')
            ax.set_xlabel('Number of clusters (k)')
            ax.set_ylabel('Inertia (Within-cluster SSE)')
            ax.set_title('Elbow Method for Optimal k')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.set_xticks(k_range)
            self.canvas.draw()
            self.metrics_text.setText(f"Elbow Method Results (Inertia):\n{pd.Series(inertias, index=k_range).to_string()}")
            self.status_bar.showMessage("Elbow method plot generated.")
            self.progress_bar.setRange(0, 100); self.progress_bar.setValue(100)

        except Exception as e:
            self.show_error(f"Error running Elbow method: {str(e)}")
            self.status_bar.showMessage("Elbow method failed.")
            self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)


    def run_cross_validation(self):
        """Runs k-fold cross-validation on the last trained supervised model."""
        if self.current_model is None or self.last_task_type not in ['classification', 'regression']:
            self.show_error("Please train a supervised model (Classification or Regression) first using its 'Train' button.")
            return
        if self.X_train is None or self.y_train is None:
             self.show_error("Training data (X_train, y_train) is not available. Please process data.")
             return

        try:
            k = self.cv_k_spin.value()
            self.status_bar.showMessage(f"Running {k}-Fold Cross-Validation for {self.current_model_name}...")
            self.progress_bar.setRange(0, k) # Progress per fold
            QApplication.processEvents()

            # Clone the last model with its parameters to ensure a fresh model for CV
            # Need to reinstantiate with stored params, as internal state might be complex
            # Re-create the model instance using the stored parameters
            cloned_model = None
            model_name = self.current_model_name
            params = self.last_trained_params
            task_type = self.last_task_type

            # --- Re-instantiate model (similar logic to run_single_model) ---
            # This is redundant but ensures CV uses the exact params shown in UI
            # == Supervised Models ==
            if model_name == "Linear Regression": cloned_model = LinearRegression(**params)
            elif model_name == "SVR": cloned_model = SVR(**params)
            elif model_name == "Decision Tree Regressor": cloned_model = DecisionTreeRegressor(**params, random_state=42)
            elif model_name == "Random Forest Regressor": cloned_model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
            elif model_name == "KNeighbors Regressor": cloned_model = KNeighborsRegressor(**params, n_jobs=-1)
            elif model_name == "Logistic Regression":
                if params.get('solver') == 'liblinear' and params.get('multi_class') == 'multinomial': params['solver'] = 'lbfgs'
                if 'probability' in params: del params['probability']
                cloned_model = LogisticRegression(**params, random_state=42)
            elif model_name == "GaussianNB": cloned_model = GaussianNB(**params)
            elif model_name == "SVC":
                prob_param = params.get('probability', False)
                clean_params = {k: v for k, v in params.items() if k != 'probability'}
                cloned_model = SVC(**clean_params, probability=prob_param, random_state=42)
            elif model_name == "Decision Tree Classifier": cloned_model = DecisionTreeClassifier(**params, random_state=42)
            elif model_name == "Random Forest Classifier": cloned_model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
            elif model_name == "KNeighbors Classifier": cloned_model = KNeighborsClassifier(**params, n_jobs=-1)
            # --- End Re-instantiation ---

            if cloned_model is None:
                 raise ValueError(f"Could not re-instantiate model '{model_name}' for Cross-Validation.")


            # Define k-fold strategy
            # Check for classification stratification
            is_classification = task_type == 'classification'
            stratify = False
            if is_classification:
                unique_classes, counts = np.unique(self.y_train, return_counts=True)
                if all(counts >= k): # Need at least k samples per class for k-fold stratified CV
                    stratify = True
                else:
                    self.status_bar.showMessage(f"Warning: Cannot stratify {k}-fold CV (not enough samples per class).", 5000)

            # Use KFold with shuffle for potentially better robustness, ensure stratification if possible
            cv_strategy = KFold(n_splits=k, shuffle=True, random_state=42)
            # Note: cross_val_score handles stratification internally if cv has integer y and model is classifier
            # Let's rely on cross_val_score's default behavior for classification tasks.

            # Determine scoring metric
            scoring = None
            metrics_list = [] # Metrics to calculate
            if task_type == 'classification':
                scoring = 'accuracy'
                metrics_list = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'] # Common classification metrics
            elif task_type == 'regression':
                scoring = 'neg_mean_squared_error' # Sklearn uses negative MSE for maximization
                metrics_list = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']

            # --- Run Cross-Validation ---
            # cross_validate allows multiple metrics
            cv_results = model_selection.cross_validate(cloned_model, self.X_train, self.y_train,
                                                         cv=cv_strategy, scoring=metrics_list,
                                                         n_jobs=-1, # Use all available cores
                                                         error_score='raise') # Raise error if a fold fails

            # --- Display Results ---
            self.figure.clear() # Clear previous plot
            self.metrics_text.clear()
            metrics_title = f"--- {k}-Fold Cross-Validation Results ({model_name}) ---"
            metrics_text = metrics_title + "\n\n"
            metrics_text += f"Parameters used: {params}\n\n"

            mean_scores = {}
            std_scores = {}
            for metric in metrics_list:
                 score_key = f'test_{metric}' # Key used by cross_validate
                 if score_key in cv_results:
                     scores = cv_results[score_key]
                     # Handle negative scores (like neg_mse)
                     if metric.startswith('neg_'):
                         scores = -scores # Convert back to positive MSE/MAE
                         metric_name = metric[4:] # Remove 'neg_' prefix
                     else:
                         metric_name = metric

                     mean_score = np.mean(scores)
                     std_score = np.std(scores)
                     mean_scores[metric_name] = mean_score
                     std_scores[metric_name] = std_score

                     metrics_text += f"{metric_name.replace('_', ' ').title()}:\n"
                     metrics_text += f"  Scores per fold: {np.round(scores, 4)}\n"
                     metrics_text += f"  Mean: {mean_score:.4f}\n"
                     metrics_text += f"  Standard Deviation: {std_score:.4f}\n\n"

            self.metrics_text.setText(metrics_text)

            # --- Visualize CV Results (e.g., box plot of scores) ---
            if mean_scores:
                 ax = self.figure.add_subplot(111)
                 metric_names_plot = list(mean_scores.keys())
                 means_plot = [mean_scores[m] for m in metric_names_plot]
                 stds_plot = [std_scores[m] for m in metric_names_plot]

                 # Simple bar chart of means with error bars
                 ax.bar(metric_names_plot, means_plot, yerr=stds_plot, capsize=5, alpha=0.7)
                 ax.set_ylabel('Score')
                 ax.set_title(f'{k}-Fold CV Mean Scores (+/- Std Dev)')
                 ax.tick_params(axis='x', rotation=30, labelsize=9) # Rotate labels if long
                 self.figure.tight_layout()
                 self.canvas.draw()

            self.progress_bar.setRange(0, 100); self.progress_bar.setValue(100)
            self.status_bar.showMessage(f"Cross-Validation complete for {model_name}.")

        except Exception as e:
            self.show_error(f"Error during Cross-Validation: {str(e)}")
            import traceback
            traceback.print_exc()
            self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
            self.status_bar.showMessage("Cross-Validation failed.")


    def show_eigenvector_example(self):
        """Calculates and displays the eigenvector example."""
        try:
            # Given covariance matrix Sigma
            Sigma = np.array([[5, 2],
                              [2, 3]])

            # Compute eigenvalues and eigenvectors
            eigenvalues, eigenvectors = np.linalg.eig(Sigma)

            # Sort eigenvectors by eigenvalues in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # The principal eigenvector (corresponding to the largest eigenvalue)
            # defines the direction of maximum variance for 1D projection.
            principal_eigenvector = eigenvectors[:, 0]

            result_text = "--- Eigenvector Example for 1D Projection ---\n\n"
            result_text += f"Given Covariance Matrix:\n{Sigma}\n\n"
            result_text += f"Eigenvalues:\n{np.round(eigenvalues, 4)}\n\n"
            result_text += f"Eigenvectors (columns):\n{np.round(eigenvectors, 4)}\n\n"
            result_text += ("The eigenvector corresponding to the largest eigenvalue "
                            f"({eigenvalues[0]:.4f}) defines the direction for 1D projection:\n")
            result_text += f"Principal Eigenvector (Projection Direction): {np.round(principal_eigenvector, 4)}\n\n"
            result_text += ("To project a 2D data point 'x' (as a row vector) into 1D, "
                            "calculate: p = x @ v\n"
                            "where 'v' is the principal eigenvector (column vector).")

            # Display in a message box or the metrics area
            # Using message box for this specific example
            QMessageBox.information(self, "Eigenvector Example", result_text)
            self.status_bar.showMessage("Eigenvector example calculation displayed.")

        except Exception as e:
            self.show_error(f"Error calculating eigenvector example: {str(e)}")
            self.status_bar.showMessage("Eigenvector example failed.")


    def update_visualization(self, y_pred, X_transformed, task_type):
        """Update the visualization panel (handles new types: embeddings, elbow)"""
        # Check if test data exists (X_test is numpy array now)
        test_data_available = self.X_test is not None and len(self.X_test) > 0
        # Use X_transformed if available (priority for dim reduction results), else use X_test
        data_to_plot = X_transformed if X_transformed is not None else self.X_test if test_data_available else None
        plot_title_suffix = ""
        xlabel = "Feature 1"
        ylabel = "Feature 2"
        zlabel = "Feature 3" # For 3D plots

        self.figure.clear()
        if data_to_plot is None and task_type not in ['dim_reduction']: # Dim reduction might plot variance without data
            self.figure.text(0.5, 0.5, 'No data available for plotting.', ha='center', va='center')
            self.canvas.draw()
            return

        try:
            n_components = data_to_plot.shape[1] if data_to_plot is not None else 0

            if task_type == 'regression':
                if y_pred is not None and self.y_test is not None:
                    ax = self.figure.add_subplot(111)
                    ax.scatter(self.y_test, y_pred, alpha=0.6, label="Predictions")
                    lims = [np.min([self.y_test.min(), y_pred.min()]), np.max([self.y_test.max(), y_pred.max()])]
                    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label="Ideal Fit")
                    ax.set_xlabel("Actual Values"); ax.set_ylabel("Predicted Values")
                    ax.set_title(f"{self.current_model_name}: Actual vs. Predicted")
                    ax.legend(); ax.grid(True, linestyle='--', alpha=0.5)
                else:
                     self.figure.text(0.5, 0.5, 'Regression plot needs Actual & Predicted values.', ha='center', va='center')

            elif task_type == 'classification':
                 if y_pred is not None and data_to_plot is not None:
                     # Default to 2D plot
                     plot_dim = 2 if n_components >= 2 else 1
                     if n_components >= 3: plot_dim = 3 # Option for 3D if available

                     if plot_dim == 3:
                         ax = self.figure.add_subplot(111, projection='3d')
                         scatter = ax.scatter(data_to_plot[:, 0], data_to_plot[:, 1], data_to_plot[:, 2], c=y_pred, cmap='viridis', alpha=0.7)
                         ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
                         legend = self.figure.colorbar(scatter, ax=ax, shrink=0.7)
                         legend.set_label('Predicted Class')
                     elif plot_dim == 2:
                         ax = self.figure.add_subplot(111)
                         scatter = ax.scatter(data_to_plot[:, 0], data_to_plot[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
                         ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
                         legend = self.figure.colorbar(scatter, ax=ax)
                         legend.set_label('Predicted Class')
                     else: # 1D plot
                         ax = self.figure.add_subplot(111)
                         y_jitter = np.random.rand(len(y_pred)) * 0.1
                         ax.scatter(data_to_plot[:, 0], y_pred + y_jitter, c=y_pred, cmap='viridis', alpha=0.6)
                         ax.set_xlabel(xlabel); ax.set_ylabel("Predicted Class (jittered)")
                         ax.set_yticks(np.unique(y_pred))

                     ax.set_title(f"{self.current_model_name}: Predictions{plot_title_suffix}")
                 else:
                      self.figure.text(0.5, 0.5, 'Classification plot needs Predictions & Features.', ha='center', va='center')

            elif task_type == 'clustering':
                 # Similar to classification plot, but using cluster labels
                 if y_pred is not None and data_to_plot is not None:
                     plot_dim = 2 if n_components >= 2 else 1
                     if n_components >= 3: plot_dim = 3

                     if plot_dim == 3:
                          ax = self.figure.add_subplot(111, projection='3d')
                          scatter = ax.scatter(data_to_plot[:, 0], data_to_plot[:, 1], data_to_plot[:, 2], c=y_pred, cmap='viridis', alpha=0.7)
                          ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
                          legend = self.figure.colorbar(scatter, ax=ax, shrink=0.7)
                          legend.set_label('Predicted Cluster')
                     elif plot_dim == 2:
                          ax = self.figure.add_subplot(111)
                          scatter = ax.scatter(data_to_plot[:, 0], data_to_plot[:, 1], c=y_pred, cmap='viridis', alpha=0.7)
                          ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
                          legend = self.figure.colorbar(scatter, ax=ax)
                          legend.set_label('Predicted Cluster')
                     else: # 1D plot
                          ax = self.figure.add_subplot(111)
                          ax.scatter(data_to_plot[:, 0], np.zeros_like(data_to_plot[:, 0]), c=y_pred, cmap='viridis', alpha=0.7)
                          ax.set_xlabel(xlabel); ax.set_yticks([])
                          # Consider adding legend/colorbar even for 1D

                     ax.set_title(f"{self.current_model_name}: Clusters{plot_title_suffix}")
                 else:
                      self.figure.text(0.5, 0.5, 'Clustering plot needs Cluster Labels & Features.', ha='center', va='center')


            elif task_type == 'dim_reduction':
                 # PCA: Explained Variance Plot
                 if isinstance(self.current_model, PCA):
                     explained_variance_ratio = self.current_model.explained_variance_ratio_
                     ax = self.figure.add_subplot(111)
                     n_comp_pca = len(explained_variance_ratio)
                     comp_range = range(1, n_comp_pca + 1)
                     ax.bar(comp_range, explained_variance_ratio, alpha=0.8, label='Individual Variance')
                     ax.plot(comp_range, np.cumsum(explained_variance_ratio), 'r-o', label='Cumulative Variance')
                     ax.set_xlabel("Principal Component")
                     ax.set_ylabel("Explained Variance Ratio")
                     ax.set_title(f"PCA Explained Variance (Total: {np.sum(explained_variance_ratio):.3f})")
                     ax.set_xticks(comp_range) # Ensure integer ticks
                     # Only show integer ticks if not too many components
                     if n_comp_pca <= 15: ax.set_xticks(comp_range)
                     else: ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True)) # Auto ticks otherwise

                     ax.legend(); ax.grid(True, axis='y', linestyle='--', alpha=0.5)
                 # LDA, t-SNE, UMAP: Scatter plot of embedding
                 elif isinstance(self.current_model, (LinearDiscriminantAnalysis, TSNE)) or \
                      (HAS_UMAP and isinstance(self.current_model, umap.UMAP)):

                      if data_to_plot is not None:
                          # Use y_test for coloring if available and classification/clustering context makes sense
                          # For LDA, y_test is definitely relevant. For tSNE/UMAP, it's for visual inspection.
                          color_data = self.y_test if self.y_test is not None else 'blue' # Default color if no y
                          color_label = 'Actual Class/Label' if self.y_test is not None else ''
                          cmap = 'viridis' if self.y_test is not None else None

                          plot_dim = 2 if n_components >= 2 else 1
                          if n_components >= 3: plot_dim = 3

                          if plot_dim == 3:
                              ax = self.figure.add_subplot(111, projection='3d')
                              scatter = ax.scatter(data_to_plot[:, 0], data_to_plot[:, 1], data_to_plot[:, 2], c=color_data, cmap=cmap, alpha=0.7)
                              ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2"); ax.set_zlabel("Component 3")
                              if self.y_test is not None:
                                   legend = self.figure.colorbar(scatter, ax=ax, shrink=0.7)
                                   legend.set_label(color_label)
                          elif plot_dim == 2:
                              ax = self.figure.add_subplot(111)
                              scatter = ax.scatter(data_to_plot[:, 0], data_to_plot[:, 1], c=color_data, cmap=cmap, alpha=0.7)
                              ax.set_xlabel("Component 1"); ax.set_ylabel("Component 2")
                              if self.y_test is not None:
                                  legend = self.figure.colorbar(scatter, ax=ax)
                                  legend.set_label(color_label)
                          else: # 1D Plot
                               ax = self.figure.add_subplot(111)
                               # Jitter y-values slightly for better visibility if using y_test colors
                               y_plot = np.zeros_like(data_to_plot[:, 0])
                               if self.y_test is not None:
                                   y_plot = self.y_test + (np.random.rand(len(self.y_test)) * 0.1)
                               scatter = ax.scatter(data_to_plot[:, 0], y_plot, c=color_data, cmap=cmap, alpha=0.7)
                               ax.set_xlabel("Component 1")
                               if self.y_test is not None: ax.set_ylabel("Class/Label (jittered)")
                               else: ax.set_yticks([])


                          ax.set_title(f"{self.current_model_name} Embedding of Test Data")
                          ax.grid(True, linestyle='--', alpha=0.5)
                      else:
                           self.figure.text(0.5, 0.5, f'{self.current_model_name}: Transformed data not available for plotting.', ha='center', va='center')

                 else:
                      self.figure.text(0.5, 0.5, 'Visualization not available for this reduction type.', ha='center', va='center')

            else: # Elbow plot handled separately
                 if self.current_model_name != "K-Means Elbow": # Avoid overwriting elbow plot immediately
                    self.figure.text(0.5, 0.5, 'Visualization not implemented for this task type.', ha='center', va='center')


            # Final adjustments and drawing
            # Use try-except for tight_layout as it can fail in some cases (e.g., empty plots)
            try:
                self.figure.tight_layout()
            except ValueError:
                pass # Ignore tight_layout errors if plot is empty or invalid
            self.canvas.draw()

        except Exception as e:
             self.show_error(f"Error updating visualization: {e}")
             import traceback
             traceback.print_exc()
             self.figure.clear()
             self.figure.text(0.5, 0.5, f"Error creating plot:\n{e}", color='red', ha='center', va='center', wrap=True)
             self.canvas.draw()


    def update_metrics(self, y_pred, X_transformed, task_type):
        """Update metrics display (Adds Silhouette, handles transformed data)"""
        metrics_title = f"--- {self.current_model_name} Performance Metrics ---"
        metrics_text = metrics_title + "\n\n"

        # Display parameters used for this run
        if self.last_trained_params:
             metrics_text += f"Parameters Used:\n{self.last_trained_params}\n\n"

        try:
            # Use X_test for calculations requiring original features, X_transformed where needed
            X_eval = self.X_test
            y_eval = self.y_test

            if y_eval is None and task_type in ['regression', 'classification']:
                 metrics_text += "Test target data (y_test) not available.\n"
                 self.metrics_text.setText(metrics_text)
                 return

            if X_eval is None and task_type != 'dim_reduction': # Dim reduction might only show variance
                 metrics_text += "Test feature data (X_test) not available.\n"
                 self.metrics_text.setText(metrics_text)
                 return

            if task_type == 'regression':
                 if y_pred is not None:
                     mse = mean_squared_error(y_eval, y_pred)
                     mae = mean_absolute_error(y_eval, y_pred)
                     rmse = np.sqrt(mse)
                     r2 = float('nan')
                     try: r2 = r2_score(y_eval, y_pred)
                     except Exception as r2_err: print(f"Could not calculate R2 score: {r2_err}")

                     metrics_text += f"Mean Squared Error (MSE): {mse:.4f}\n"
                     metrics_text += f"Root Mean Squared Error (RMSE): {rmse:.4f}\n"
                     metrics_text += f"Mean Absolute Error (MAE): {mae:.4f}\n"
                     metrics_text += f"R² Score: {r2:.4f}\n"
                 else: metrics_text += "Predictions not available for metric calculation.\n"

            elif task_type == 'classification':
                 if y_pred is not None:
                     accuracy = accuracy_score(y_eval, y_pred)
                     metrics_text += f"Accuracy: {accuracy:.4f}\n\n"
                     try:
                         conf_matrix = confusion_matrix(y_eval, y_pred)
                         labels = np.unique(np.concatenate((y_eval.astype(str), y_pred.astype(str)))) # Ensure labels are consistent str type for pandas
                         cm_df = pd.DataFrame(conf_matrix, index=[f"True:{l}" for l in labels], columns=[f"Pred:{l}" for l in labels])
                         metrics_text += "Confusion Matrix:\n" + cm_df.to_string() + "\n"
                     except Exception as cm_error:
                          metrics_text += f"Could not generate Confusion Matrix: {cm_error}\n"
                          metrics_text += f"Y Test unique: {np.unique(y_eval)}, Y Pred unique: {np.unique(y_pred)}\n"
                 else: metrics_text += "Predictions not available for metric calculation.\n"

            elif task_type == 'clustering':
                 metrics_text += "Clustering Metrics:\n"
                 if isinstance(self.current_model, KMeans) and hasattr(self.current_model, 'inertia_'):
                     metrics_text += f" - Inertia (Train Set): {self.current_model.inertia_:.4f}\n"

                 # Silhouette Score requires features and predicted labels
                 if y_pred is not None and X_eval is not None and len(np.unique(y_pred)) > 1:
                     try:
                         # Use X_test (original or scaled features) for silhouette calculation
                         sil_score = silhouette_score(X_eval, y_pred)
                         metrics_text += f" - Silhouette Score (Test Set): {sil_score:.4f}\n"
                     except ValueError as sil_err:
                          metrics_text += f" - Could not calculate Silhouette Score: {sil_err}\n"
                     except Exception as e:
                          metrics_text += f" - Error calculating Silhouette Score: {e}\n"
                 elif len(np.unique(y_pred)) <= 1:
                      metrics_text += " - Silhouette Score requires more than 1 cluster.\n"
                 else:
                      metrics_text += " - Silhouette Score requires features (X_test) and cluster predictions.\n"

            elif task_type == 'dim_reduction':
                  metrics_text += "Dimensionality Reduction Results:\n"
                  if isinstance(self.current_model, PCA):
                     n_comp_pca = self.current_model.n_components_
                     metrics_text += f" - PCA Components: {n_comp_pca}\n"
                     metrics_text += f" - Explained Variance Ratio:\n {np.round(self.current_model.explained_variance_ratio_, 4)}\n"
                     metrics_text += f" - Total Explained Variance: {np.sum(self.current_model.explained_variance_ratio_):.4f}\n"
                  elif isinstance(self.current_model, LinearDiscriminantAnalysis):
                       n_comp_lda = self.current_model.n_components_
                       metrics_text += f" - LDA Components: {n_comp_lda}\n"
                       if hasattr(self.current_model, 'explained_variance_ratio_'):
                           metrics_text += f" - Explained Variance Ratio (LDA):\n {np.round(self.current_model.explained_variance_ratio_, 4)}\n"
                           metrics_text += f" - Total Explained Variance (LDA): {np.sum(self.current_model.explained_variance_ratio_):.4f}\n"
                       # Add class separation metric? Could compute manually based on transformed data + y_test
                       if X_transformed is not None and y_eval is not None:
                           # Example: Compute Silhouette on transformed data using true labels
                            try:
                                if len(np.unique(y_eval)) > 1:
                                    lda_sil = silhouette_score(X_transformed, y_eval)
                                    metrics_text += f" - Silhouette Score (Transformed Data, True Labels): {lda_sil:.4f} (Indicates class separation)\n"
                            except Exception as lda_sil_e:
                                 metrics_text += f" - Could not compute Silhouette on LDA output: {lda_sil_e}\n"

                  elif isinstance(self.current_model, TSNE) or (HAS_UMAP and isinstance(self.current_model, umap.UMAP)):
                       metrics_text += f" - {self.current_model_name} completed.\n"
                       if X_transformed is not None:
                           metrics_text += f" - Output Embedding Shape: {X_transformed.shape}\n"
                       if hasattr(self.current_model, 'kl_divergence_') and self.current_model.kl_divergence_ is not None: # t-SNE specific
                            metrics_text += f" - Final KL Divergence (t-SNE): {self.current_model.kl_divergence_:.4f}\n"
                       # Add Silhouette using true labels for qualitative assessment of embedding separation
                       if X_transformed is not None and y_eval is not None and len(np.unique(y_eval)) > 1:
                           try:
                                embed_sil = silhouette_score(X_transformed, y_eval)
                                metrics_text += f" - Silhouette Score (Embedding, True Labels): {embed_sil:.4f} (Indicates class separation in embedding)\n"
                           except Exception as embed_sil_e:
                                metrics_text += f" - Could not compute Silhouette on embedding: {embed_sil_e}\n"

                  else: metrics_text += "Metrics not implemented for this reduction type.\n"
            else:
                 metrics_text += "Metrics calculation not implemented for this task type.\n"

        except Exception as e:
             metrics_text += f"\nError calculating metrics: {str(e)}\n"
             import traceback
             traceback.print_exc()

        self.metrics_text.setText(metrics_text)


    # --- NN Methods (largely unchanged, ensure they don't interfere with new CV) ---
    def _update_layer_display(self):
        if not self.layer_config:
            self.layer_list_widget.setText("No layers added yet.")
            return
    
        display_text = ""
        for i, layer_conf in enumerate(self.layer_config):
            params_str_list = []
            for k, v in layer_conf["params"].items():
                if k == "kernel_regularizer_l2":
                    params_str_list.append(f"L2={v:.0e}") # scientific notation for small L2
                elif isinstance(v, tuple):
                    params_str_list.append(f"{k.replace('_', ' ')}={v}")
                elif isinstance(v, bool):
                    params_str_list.append(f"{k.replace('_', ' ')}={v}")
                else:
                    params_str_list.append(f"{k.replace('_', ' ')}={v}")
            params_str = ", ".join(params_str_list)
            display_text += f"{i+1}: {layer_conf['type']}({params_str})\n"
        self.layer_list_widget.setText(display_text)

    def _clear_layers(self):
        """Clears the neural network layer configuration. (Unchanged)"""
        self.layer_config = []; self._update_layer_display()
        self.status_bar.showMessage("Neural network layers cleared.")

    @pyqtSlot(str)
    def _update_loss_options(self, task_type):
        """Update loss options based on NN task type. (Unchanged)"""
        self.nn_loss_combo.clear(); self.nn_task_type = task_type
        if task_type == "Classification": self.nn_loss_combo.addItems(['categorical_crossentropy','sparse_categorical_crossentropy','binary_crossentropy','hinge','squared_hinge'])
        elif task_type == "Regression": self.nn_loss_combo.addItems(['mean_squared_error','mean_absolute_error','huber_loss','mean_squared_logarithmic_error'])
        else: self.nn_loss_combo.addItem("Select Task Type First")

    def add_layer_dialog_detailed(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        dialog_layout = QVBoxLayout(dialog)

        # Layer Type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Layer Type:"))
        layer_type_combo = QComboBox()
        layer_types = ["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout", "LSTM", "GRU"]
        layer_type_combo.addItems(layer_types)
        type_layout.addWidget(layer_type_combo)
        dialog_layout.addLayout(type_layout)

        # Dynamic Parameters Area
        params_container = QWidget()
        self.current_layer_params_layout = QGridLayout(params_container) # Store to access widgets
        dialog_layout.addWidget(params_container)

        # Initial params display
        self._on_layer_type_changed_in_dialog(layer_type_combo.currentText())
        layer_type_combo.currentTextChanged.connect(
            lambda text: self._on_layer_type_changed_in_dialog(text)
        )

        # OK and Cancel buttons
        btn_layout = QHBoxLayout()
        btn_ok = QPushButton("Add Layer")
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(lambda: self._accept_add_layer_dialog(dialog, layer_type_combo.currentText()))
        btn_cancel.clicked.connect(dialog.reject)
        btn_layout.addWidget(btn_ok)
        btn_layout.addWidget(btn_cancel)
        dialog_layout.addLayout(btn_layout)

        dialog.exec()

    def _on_layer_type_changed_in_dialog(self, layer_type):
        # Clear previous params from self.current_layer_params_layout
        for i in reversed(range(self.current_layer_params_layout.count())):
            widget_to_remove = self.current_layer_params_layout.itemAt(i).widget()
            self.current_layer_params_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

        # Add new params
        if layer_type == "Dense": self._add_dense_params_to_dialog()
        elif layer_type == "Conv2D": self._add_conv2d_params_to_dialog()
        elif layer_type == "MaxPooling2D": self._add_maxpooling2d_params_to_dialog()
        elif layer_type == "Dropout": self._add_dropout_params_to_dialog()
        elif layer_type == "LSTM" or layer_type == "GRU": self._add_lstm_gru_params_to_dialog(layer_type)
        # Flatten has no common params other than name, handled by default
        
    def _add_dense_params_to_dialog(self):
        layout = self.current_layer_params_layout
        self.dlg_dense_units = QSpinBox(); self.dlg_dense_units.setRange(1, 8192); self.dlg_dense_units.setValue(128)
        self.dlg_dense_activation = QComboBox(); self.dlg_dense_activation.addItems(["relu", "sigmoid", "tanh", "linear", "softmax"])
        self.dlg_dense_l2_reg = QDoubleSpinBox(); self.dlg_dense_l2_reg.setRange(0.0, 0.1); self.dlg_dense_l2_reg.setValue(0.0); self.dlg_dense_l2_reg.setDecimals(5)
        
        layout.addWidget(QLabel("Units:"), 0, 0); layout.addWidget(self.dlg_dense_units, 0, 1)
        layout.addWidget(QLabel("Activation:"), 1, 0); layout.addWidget(self.dlg_dense_activation, 1, 1)
        layout.addWidget(QLabel("L2 Regularization (0 for none):"), 2, 0); layout.addWidget(self.dlg_dense_l2_reg, 2, 1)

    def _add_conv2d_params_to_dialog(self):
        layout = self.current_layer_params_layout
        self.dlg_conv2d_filters = QSpinBox(); self.dlg_conv2d_filters.setRange(1, 1024); self.dlg_conv2d_filters.setValue(32)
        self.dlg_conv2d_kernel_size = QLineEdit("3,3"); # Use parse_int_tuple later
        self.dlg_conv2d_strides = QLineEdit("1,1");
        self.dlg_conv2d_padding = QComboBox(); self.dlg_conv2d_padding.addItems(["valid", "same"])
        self.dlg_conv2d_activation = QComboBox(); self.dlg_conv2d_activation.addItems(["relu", "sigmoid", "tanh", "linear"])
        self.dlg_conv2d_l2_reg = QDoubleSpinBox(); self.dlg_conv2d_l2_reg.setRange(0.0, 0.1); self.dlg_conv2d_l2_reg.setValue(0.0); self.dlg_conv2d_l2_reg.setDecimals(5)

        layout.addWidget(QLabel("Filters:"), 0, 0); layout.addWidget(self.dlg_conv2d_filters, 0, 1)
        layout.addWidget(QLabel("Kernel Size (h,w):"), 1, 0); layout.addWidget(self.dlg_conv2d_kernel_size, 1, 1)
        layout.addWidget(QLabel("Strides (h,w):"), 2, 0); layout.addWidget(self.dlg_conv2d_strides, 2, 1)
        layout.addWidget(QLabel("Padding:"), 3, 0); layout.addWidget(self.dlg_conv2d_padding, 3, 1)
        layout.addWidget(QLabel("Activation:"), 4, 0); layout.addWidget(self.dlg_conv2d_activation, 4, 1)
        layout.addWidget(QLabel("L2 Regularization (0 for none):"), 5, 0); layout.addWidget(self.dlg_conv2d_l2_reg, 5, 1)

    def _add_maxpooling2d_params_to_dialog(self):
        layout = self.current_layer_params_layout
        self.dlg_pool_size = QLineEdit("2,2")
        self.dlg_pool_strides = QLineEdit("") # Optional, defaults to pool_size
        self.dlg_pool_padding = QComboBox(); self.dlg_pool_padding.addItems(["valid", "same"])

        layout.addWidget(QLabel("Pool Size (h,w):"), 0, 0); layout.addWidget(self.dlg_pool_size, 0, 1)
        layout.addWidget(QLabel("Strides (h,w) [optional]:"), 1, 0); layout.addWidget(self.dlg_pool_strides, 1, 1)
        layout.addWidget(QLabel("Padding:"), 2, 0); layout.addWidget(self.dlg_pool_padding, 2, 1)

    def _add_dropout_params_to_dialog(self):
        layout = self.current_layer_params_layout
        self.dlg_dropout_rate = QDoubleSpinBox(); self.dlg_dropout_rate.setRange(0.0, 0.99); self.dlg_dropout_rate.setValue(0.5); self.dlg_dropout_rate.setDecimals(2)
        layout.addWidget(QLabel("Dropout Rate:"), 0, 0); layout.addWidget(self.dlg_dropout_rate, 0, 1)

    def _add_lstm_gru_params_to_dialog(self, layer_type): # layer_type is "LSTM" or "GRU"
        layout = self.current_layer_params_layout
        self.dlg_rnn_units = QSpinBox(); self.dlg_rnn_units.setRange(1, 2048); self.dlg_rnn_units.setValue(64)
        self.dlg_rnn_activation = QComboBox(); self.dlg_rnn_activation.addItems(["tanh", "sigmoid", "relu"])
        self.dlg_rnn_recurrent_activation = QComboBox(); self.dlg_rnn_recurrent_activation.addItems(["sigmoid", "tanh", "relu"])
        self.dlg_rnn_return_sequences = QCheckBox("Return Sequences")
        
        layout.addWidget(QLabel("Units:"), 0, 0); layout.addWidget(self.dlg_rnn_units, 0, 1)
        layout.addWidget(QLabel("Activation:"), 1, 0); layout.addWidget(self.dlg_rnn_activation, 1, 1)
        layout.addWidget(QLabel("Recurrent Activation:"), 2, 0); layout.addWidget(self.dlg_rnn_recurrent_activation, 2, 1)
        layout.addWidget(self.dlg_rnn_return_sequences, 3, 0, 1, 2)


    def _accept_add_layer_dialog(self, dialog, layer_type):
        params = {}
        if layer_type == "Dense":
            params["units"] = self.dlg_dense_units.value()
            params["activation"] = self.dlg_dense_activation.currentText()
            if self.dlg_dense_l2_reg.value() > 0: params["kernel_regularizer_l2"] = self.dlg_dense_l2_reg.value()
        elif layer_type == "Conv2D":
            params["filters"] = self.dlg_conv2d_filters.value()
            params["kernel_size"] = parse_int_tuple(self.dlg_conv2d_kernel_size.text(), (3,3))
            params["strides"] = parse_int_tuple(self.dlg_conv2d_strides.text(), (1,1))
            params["padding"] = self.dlg_conv2d_padding.currentText()
            params["activation"] = self.dlg_conv2d_activation.currentText()
            if self.dlg_conv2d_l2_reg.value() > 0: params["kernel_regularizer_l2"] = self.dlg_conv2d_l2_reg.value()
        elif layer_type == "MaxPooling2D":
            params["pool_size"] = parse_int_tuple(self.dlg_pool_size.text(), (2,2))
            strides_text = self.dlg_pool_strides.text().strip()
            if strides_text: params["strides"] = parse_int_tuple(strides_text)
            params["padding"] = self.dlg_pool_padding.currentText()
        elif layer_type == "Dropout":
            params["rate"] = self.dlg_dropout_rate.value()
        elif layer_type == "LSTM" or layer_type == "GRU":
            params["units"] = self.dlg_rnn_units.value()
            params["activation"] = self.dlg_rnn_activation.currentText()
            params["recurrent_activation"] = self.dlg_rnn_recurrent_activation.currentText()
            params["return_sequences"] = self.dlg_rnn_return_sequences.isChecked()
        # Flatten needs no specific params beyond name
        
        self.layer_config.append({"type": layer_type, "params": params})
        self._update_layer_display()
        dialog.accept()

    def _remove_last_layer(self):
        if self.layer_config:
            self.layer_config.pop()
            self._update_layer_display()
            self.status_bar.showMessage("Last layer removed.")
        else:
            self.status_bar.showMessage("No layers to remove.")

    def save_nn_architecture(self):
        if not self.layer_config:
            self.show_error("No architecture to save.")
            return
        file_name, _ = QFileDialog.getSaveFileName(self, "Save NN Architecture", "", "JSON files (*.json)")
        if file_name:
            try:
                import json
                with open(file_name, 'w') as f:
                    json.dump(self.layer_config, f, indent=4)
                self.status_bar.showMessage(f"Architecture saved to {file_name}")
            except Exception as e:
                self.show_error(f"Error saving architecture: {e}")

    def load_nn_architecture(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load NN Architecture", "", "JSON files (*.json)")
        if file_name:
            try:
                import json
                with open(file_name, 'r') as f:
                    self.layer_config = json.load(f)
                self._update_layer_display()
                self.status_bar.showMessage(f"Architecture loaded from {file_name}")
            except Exception as e:
                self.show_error(f"Error loading architecture: {e}")
                self.layer_config = [] # Clear if loading failed
                self._update_layer_display()

    def create_training_params_group(self):
        """Create group widget for NN training parameters (Unchanged)"""
        # --- Uses the same logic as the original provided code ---
        # Includes Batch Size, Epochs, Learning Rate spins
        # ... (Code from original file for create_training_params_group) ...
        # Simplified for brevity:
        widget = QWidget()
        layout = QVBoxLayout(widget); layout.setContentsMargins(0,0,0,0)
        # Batch Size
        batch_layout = QHBoxLayout(); batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox(); self.batch_size_spin.setRange(1, 2048); self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin); layout.addLayout(batch_layout)
        # Epochs
        epochs_layout = QHBoxLayout(); epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox(); self.epochs_spin.setRange(1, 1000); self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin); layout.addLayout(epochs_layout)
        # Learning Rate
        lr_layout = QHBoxLayout(); lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox(); self.lr_spin.setRange(1e-6, 1.0); self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001); self.lr_spin.setDecimals(6)
        lr_layout.addWidget(self.lr_spin); layout.addLayout(lr_layout)
        return widget

    def create_neural_network_enhanced(self, input_shape, num_classes=None, base_model_output=None):
        """
        Create neural network based on current layer configuration or pre-trained model.
        If base_model_output is provided, it's used as the input for custom top layers.
        """
        
        if self.pretrained_model_combo.currentText() != "None" and self.pretrained_group.isChecked() and base_model_output is not None:
            # Building on top of a pre-trained model
            x = base_model_output
            # Add any custom top layers from layer_config IF SPECIFIED AS TOP LAYERS
            # For simplicity now, we'll assume layer_config is mainly for scratch models
            # or a very simple head. A more advanced setup would allow defining a complex head.
            
            # Example: Add a GlobalAveragePooling2D if the output is 4D (features maps)
            if len(x.shape) == 4: # (batch, height, width, channels)
                 x = layers.GlobalAveragePooling2D()(x)
            elif len(x.shape) == 3 and x.shape[-1] != 1 and x.shape[1] > 1 : # (batch, timesteps, features) for RNN, ensure it's not (batch, features, 1)
                 x = layers.GlobalAveragePooling1D()(x) # Or Flatten, or an RNN layer

            # Add a Dense output layer based on task
            if self.nn_task_type == "Classification":
                if num_classes is None:
                    raise ValueError("num_classes must be provided for classification task.")
                if num_classes == 2: # Binary classification
                    final_activation = 'sigmoid'
                    output_units = 1
                elif num_classes > 2: # Multi-class classification
                    final_activation = 'softmax'
                    output_units = num_classes
                else: # num_classes == 1 implies regression-like output for classification
                    final_activation = 'sigmoid' # Or linear if model handles logits
                    output_units = 1

                # Check if the last configured layer is already an output layer
                is_output_defined = False
                if self.layer_config:
                    last_layer_conf = self.layer_config[-1]
                    if last_layer_conf["type"] == "Dense":
                        if last_layer_conf["params"].get("units") == output_units and \
                           last_layer_conf["params"].get("activation") == final_activation:
                            is_output_defined = True
                
                if not is_output_defined or not self.layer_config: # Add default if no layers or last isn't output
                    x = layers.Dense(output_units, activation=final_activation, name="output_dense")(x)
                else: # Use configured layers as the head
                    model_top = models.Sequential(name="custom_top")
                    for i, layer_conf in enumerate(self.layer_config):
                        # This part of logic for applying layer_config on top of pre-trained needs care
                        # For now, let's assume layer_config defines the *entire* top if used with pre-trained
                        # A simpler approach is to just add a default dense layer after GAP for pre-trained.
                        # The current code structure is a bit mixed. Let's stick to default head for pre-trained for now.
                        pass # Placeholder for complex head from layer_config. For now, simple Dense layer above.

            elif self.nn_task_type == "Regression":
                # Regression typically has 1 output unit, linear activation
                is_output_defined = False
                if self.layer_config:
                    last_layer_conf = self.layer_config[-1]
                    if last_layer_conf["type"] == "Dense" and \
                       last_layer_conf["params"].get("units") == 1 and \
                       (last_layer_conf["params"].get("activation") == "linear" or not last_layer_conf["params"].get("activation")):
                        is_output_defined = True

                if not is_output_defined or not self.layer_config:
                    x = layers.Dense(1, activation='linear', name="output_dense")(x)
                else:
                     pass # Similar to classification, complex head needs careful handling.
            
            # The input to this part is base_model.input, output is x
            # So, we need to link base_model.input to this custom top
            # This is typically done when defining the overall model earlier
            return x # Return the tensor that is the output of the custom head

        else: # Building model from scratch using layer_config
            if not self.layer_config:
                self.show_error("No layers defined for the neural network.")
                return None

            model = models.Sequential(name="ML_Course_NN_Scratch")
            first_layer = True
            for i, layer_conf in enumerate(self.layer_config):
                layer_type = layer_conf["type"]
                params = layer_conf["params"].copy() # Make a copy to modify
                
                # Handle L2 regularization
                l2_reg_val = params.pop("kernel_regularizer_l2", 0.0)
                regularizer = None
                if l2_reg_val > 0:
                    regularizer = tf.keras.regularizers.l2(l2_reg_val)

                current_layer = None
                if layer_type == "Dense":
                    if regularizer: params["kernel_regularizer"] = regularizer
                    if first_layer: current_layer = layers.Dense(**params, input_shape=input_shape)
                    else: current_layer = layers.Dense(**params)
                elif layer_type == "Conv2D":
                    if regularizer: params["kernel_regularizer"] = regularizer
                    if first_layer: current_layer = layers.Conv2D(**params, input_shape=input_shape)
                    else: current_layer = layers.Conv2D(**params)
                elif layer_type == "MaxPooling2D":
                    current_layer = layers.MaxPooling2D(**params)
                elif layer_type == "Flatten":
                    current_layer = layers.Flatten()
                elif layer_type == "Dropout":
                    current_layer = layers.Dropout(**params)
                elif layer_type == "LSTM":
                    if first_layer: current_layer = layers.LSTM(**params, input_shape=input_shape)
                    else: current_layer = layers.LSTM(**params)
                elif layer_type == "GRU":
                    if first_layer: current_layer = layers.GRU(**params, input_shape=input_shape)
                    else: current_layer = layers.GRU(**params)
                
                if current_layer:
                    model.add(current_layer)
                    first_layer = False
                else:
                    self.show_error(f"Unknown layer type or config error: {layer_type}")
                    return None
            
            # Add final output layer if not already defined by user in layer_config
            # Check if the last layer in config is an appropriate output layer
            is_output_defined_by_user = False
            if self.layer_config:
                last_conf = self.layer_config[-1]
                if last_conf["type"] == "Dense":
                    if self.nn_task_type == "Classification":
                        needed_units = 1 if num_classes == 2 else num_classes
                        needed_act = 'sigmoid' if num_classes == 2 else 'softmax'
                        if last_conf["params"].get("units") == needed_units and \
                           last_conf["params"].get("activation") == needed_act:
                            is_output_defined_by_user = True
                    elif self.nn_task_type == "Regression":
                        if last_conf["params"].get("units") == 1 and \
                           (last_conf["params"].get("activation") == "linear" or not last_conf["params"].get("activation")):
                           is_output_defined_by_user = True
            
            if not is_output_defined_by_user:
                if self.nn_task_type == "Classification":
                    if num_classes is None: raise ValueError("num_classes for classification.")
                    final_activation = 'sigmoid' if num_classes == 2 else 'softmax'
                    output_units = 1 if num_classes == 2 else num_classes
                    model.add(layers.Dense(output_units, activation=final_activation, name="output_dense_auto"))
                elif self.nn_task_type == "Regression":
                    model.add(layers.Dense(1, activation='linear', name="output_dense_auto"))
            return model

    def _create_optimizer(self):
        opt_name = self.nn_optimizer_combo.currentText()
        lr = self.lr_spin.value()
        optimizer = None

        if opt_name == "Adam":
            optimizer = optimizers.Adam(learning_rate=lr,
                                        beta_1=self.adam_beta1_spin.value(),
                                        beta_2=self.adam_beta2_spin.value(),
                                        epsilon=self.adam_epsilon_spin.value(),
                                        amsgrad=self.adam_amsgrad_cb.isChecked())
        elif opt_name == "SGD":
            optimizer = optimizers.SGD(learning_rate=lr,
                                       momentum=self.sgd_momentum_spin.value(),
                                       nesterov=self.sgd_nesterov_cb.isChecked())
        elif opt_name == "RMSprop":
            optimizer = optimizers.RMSprop(learning_rate=lr,
                                           rho=self.rmsprop_rho_spin.value(),
                                           momentum=self.rmsprop_momentum_spin.value(),
                                           epsilon=self.rmsprop_epsilon_spin.value())
        else: # Default to Adam
            optimizer = optimizers.Adam(learning_rate=lr)
        return optimizer

    def _create_callbacks(self, initial_lr):
        keras_callbacks = []
        
        # 1. UI Progress Callback (already exists, ensure it's instantiated)
        keras_callbacks.append(self.create_progress_callback()) # Assumes this method exists and works

        # 2. Early Stopping
        if self.early_stopping_cb.isChecked():
            es_callback = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',  # Could be val_accuracy for classification
                patience=self.early_stopping_patience_spin.value(),
                verbose=1,
                restore_best_weights=True # Important!
            )
            keras_callbacks.append(es_callback)
            self.status_bar.showMessage("Early stopping enabled.", 2000)

        # 3. Learning Rate Scheduler
        scheduler_type = self.lr_scheduler_combo.currentText()
        if scheduler_type == "Step Decay":
            # Example: decay LR by a factor every N epochs
            def step_decay_schedule(epoch, lr):
                initial_lrate = initial_lr
                drop = 0.5  # Factor to drop by
                epochs_drop = 10.0 # Drop every 10 epochs
                lrate = initial_lrate * np.power(drop, np.floor((1+epoch)/epochs_drop))
                return lrate
            lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(step_decay_schedule, verbose=1)
            keras_callbacks.append(lr_scheduler_callback)
            self.status_bar.showMessage("Step decay LR scheduler enabled.", 2000)
        elif scheduler_type == "Exponential Decay":
            # lr = lr0 * exp(-kt)
            def exp_decay_schedule(epoch, lr):
                k = 0.1 # Decay rate
                return initial_lr * np.exp(-k * epoch)
            lr_scheduler_callback = tf.keras.callbacks.LearningRateScheduler(exp_decay_schedule, verbose=1)
            keras_callbacks.append(lr_scheduler_callback)
            self.status_bar.showMessage("Exponential decay LR scheduler enabled.", 2000)

        # 4. Gradient Tracker Callback
        self.gradient_tracker_callback = GradientTrackerCallback(self) # Pass GUI instance for potential updates
        keras_callbacks.append(self.gradient_tracker_callback)
        self.plot_gradients_btn.setEnabled(True) # Enable button to plot gradients later

        return keras_callbacks

    def _get_nn_data(self):
        """Prepares X_train, y_train, X_test, y_test for NN, including reshaping and encoding."""
        if self.X_train is None or self.y_train is None or self.X_test is None or self.y_test is None:
            self.show_error("Data not fully processed and split. Please process data first.")
            return None, None, None, None, None, None

        X_train_nn = self.X_train.copy()
        y_train_nn = self.y_train.copy()
        X_test_nn = self.X_test.copy()
        y_test_nn = self.y_test.copy()

        # Determine input shape for the first layer (auto-detect if possible)
        # This needs to be more robust, considering data type (image, sequence, tabular)
        # For now, assume tabular or needs user to configure layers appropriately (e.g. Flatten for images)
        input_shape = X_train_nn.shape[1:] # (features,) or (height, width, channels) or (timesteps, features)
        
        # Reshaping for CNN/RNN if first layer is Conv2D/LSTM/GRU
        # This is tricky to automate perfectly. User's first layer choice guides this.
        if self.layer_config:
            first_layer_type = self.layer_config[0]['type']
            if first_layer_type == "Conv2D":
                # Try to infer image dimensions if data is flat
                if len(input_shape) == 1: # Flat data (num_features)
                    # Heuristic: if num_features is a perfect square, assume grayscale image
                    # Or if divisible by 3, assume RGB. This is very basic.
                    # User should ideally provide data in (H, W, C) format.
                    num_features = input_shape[0]
                    sqrt_dim = int(np.sqrt(num_features))
                    if sqrt_dim * sqrt_dim == num_features: # Grayscale
                        height, width, channels = sqrt_dim, sqrt_dim, 1
                        input_shape = (height, width, channels)
                        X_train_nn = X_train_nn.reshape((-1, height, width, channels))
                        X_test_nn = X_test_nn.reshape((-1, height, width, channels))
                        self.status_bar.showMessage(f"Reshaped flat data to {input_shape} for Conv2D.", 3000)
                    elif num_features % 3 == 0 and int(np.sqrt(num_features/3))**2 == num_features/3 : # RGB
                        sqrt_dim_rgb = int(np.sqrt(num_features/3))
                        height, width, channels = sqrt_dim_rgb, sqrt_dim_rgb, 3
                        input_shape = (height, width, channels)
                        X_train_nn = X_train_nn.reshape((-1, height, width, channels))
                        X_test_nn = X_test_nn.reshape((-1, height, width, channels))
                        self.status_bar.showMessage(f"Reshaped flat data to {input_shape} for Conv2D (assumed RGB).", 3000)
                    else:
                        self.show_error(f"First layer is Conv2D, but input data is flat ({num_features} features) and not easily reshaped to HWC image. Ensure data is pre-shaped or add Flatten if input is image-like but flattened.")
                        return None, None, None, None, None, None
                elif len(input_shape) == 2: # (H,W) -> add channel
                    input_shape = (*input_shape, 1)
                    X_train_nn = np.expand_dims(X_train_nn, axis=-1)
                    X_test_nn = np.expand_dims(X_test_nn, axis=-1)
                elif len(input_shape) != 3:
                     self.show_error(f"Conv2D expects 3D input (H,W,C), got {len(input_shape)}D. Please check data or add Flatten layer if data is already image-like.")
                     return None, None, None, None, None, None


            elif first_layer_type in ["LSTM", "GRU"]:
                if len(input_shape) == 1: # (features,) -> assume (timesteps=1, features)
                    input_shape = (1, input_shape[0])
                    X_train_nn = X_train_nn.reshape((-1, 1, X_train_nn.shape[1]))
                    X_test_nn = X_test_nn.reshape((-1, 1, X_test_nn.shape[1]))
                    self.status_bar.showMessage(f"Reshaped flat data to {input_shape} for RNN (assumed 1 timestep).", 3000)
                elif len(input_shape) != 2: # Expects (timesteps, features)
                    self.show_error(f"RNN layer expects 2D input (timesteps, features), got {len(input_shape)}D. Please check data format.")
                    return None, None, None, None, None, None
        
        # Target variable processing
        num_classes = None
        if self.nn_task_type == "Classification":
            unique_labels = np.unique(y_train_nn)
            num_classes = len(unique_labels)
            
            if num_classes == 2: # Binary classification
                # Ensure labels are 0 and 1
                # If they are not, map them (e.g. using LabelEncoder first, then ensure 0/1)
                # For sigmoid output, target should be (batch_size, 1)
                if not (np.all(np.isin(unique_labels, [0,1])) and len(unique_labels) == 2):
                    self.status_bar.showMessage("Binary classification: mapping labels to 0/1.", 2000)
                    # A simple mapping, assuming sorted unique_labels[0] -> 0, unique_labels[1] -> 1
                    # More robust: use sklearn.preprocessing.LabelEncoder
                    label_map = {label: i for i, label in enumerate(sorted(unique_labels))}
                    y_train_nn = np.array([label_map[y] for y in y_train_nn])
                    y_test_nn = np.array([label_map[y] for y in y_test_nn])

            elif num_classes > 2: # Multi-class classification
                # One-hot encode if loss is categorical_crossentropy
                # If sparse_categorical_crossentropy, integer labels are fine.
                if self.nn_loss_combo.currentText() == 'categorical_crossentropy':
                    self.status_bar.showMessage("Multi-class: One-hot encoding labels.", 2000)
                    y_train_nn = tf.keras.utils.to_categorical(y_train_nn, num_classes=num_classes)
                    y_test_nn = tf.keras.utils.to_categorical(y_test_nn, num_classes=num_classes)
            else: # num_classes = 1, could be problematic for classification
                self.show_error("Classification task detected only 1 class. Check your target variable.")
                return None, None, None, None, None, None
        
        if X_train_nn is not None and y_train_nn is not None:
            sample_size = min(32, X_train_nn.shape[0]) # Take a small sample
            self.X_train_for_grads = X_train_nn[:sample_size]
            self.y_train_for_grads = y_train_nn[:sample_size]

        return X_train_nn, y_train_nn, X_test_nn, y_test_nn, input_shape, num_classes


    def train_neural_network_enhanced(self):
        if (self.X_train is None or self.y_train is None or
            self.X_test is None or self.y_test is None):
            self.show_error("Data not fully processed and split. Please process data first.")
            return

        if not self.layer_config and not (self.pretrained_group.isChecked() and self.pretrained_model_combo.currentText() != "None"):
            self.show_error("No layers defined and no pre-trained model selected. Please define an architecture.")
            return

        try:
            self.status_bar.showMessage("Preparing NN data and model...")
            self.progress_bar.setRange(0,0) # Indeterminate
            QApplication.processEvents()

            X_train_nn, y_train_nn, X_test_nn, y_test_nn, input_shape, num_classes = self._get_nn_data()
            if X_train_nn is None: return # Error handled in _get_nn_data

            # --- Model Creation ---
            model = None
            base_model_name = self.pretrained_model_combo.currentText()

            if self.pretrained_group.isChecked() and base_model_name != "None":
                self.status_bar.showMessage(f"Loading pre-trained model: {base_model_name}...")
                QApplication.processEvents()
                
                # Input shape for pre-trained models often fixed, e.g., (224, 224, 3)
                # We need to ensure our input_shape matches or adapt data/model.
                # For simplicity, let's assume input_shape from _get_nn_data is what we'll use
                # or we resize for standard models.
                
                # This is a simplified pre-trained loading. Real applications need more robust input handling.
                target_size = None # (height, width)
                if base_model_name == "VGG16":
                    base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
                    target_size = (224,224) # VGG16 typical input
                elif base_model_name == "ResNet50":
                    base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
                    target_size = (224,224)
                elif base_model_name == "MobileNetV2":
                    base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
                    target_size = (224,224)
                else:
                    self.show_error(f"Pre-trained model {base_model_name} not implemented yet.")
                    return

                # Check if input_shape needs resizing for standard models
                if len(input_shape) == 3 and (input_shape[0] != target_size[0] or input_shape[1] != target_size[1]):
                    self.status_bar.showMessage(f"Resizing input images to {target_size} for {base_model_name}.", 3000)
                    # This requires tf.image.resize, best done within a tf.data pipeline or a Lambda layer
                    # For simplicity here, we'll assume data is already correct or create a resizing layer
                    resize_layer = layers.Resizing(target_size[0], target_size[1])
                    inputs = tf.keras.Input(shape=input_shape)
                    resized_inputs = resize_layer(inputs)
                    base_output_tensor = base(resized_inputs)
                    base_model_input_tensor = inputs
                else:
                    base_model_input_tensor = base.input
                    base_output_tensor = base.output

                # Fine-tuning: unfreeze some layers
                if self.finetune_cb.isChecked():
                    base.trainable = True
                    num_layers_to_unfreeze = self.finetune_layers_spin.value()
                    if num_layers_to_unfreeze > 0 and num_layers_to_unfreeze < len(base.layers):
                        for layer in base.layers[:-num_layers_to_unfreeze]:
                            layer.trainable = False
                        self.status_bar.showMessage(f"Fine-tuning: Unfroze last {num_layers_to_unfreeze} layers of {base_model_name}.", 3000)
                    else: # Unfreeze all or none
                         self.status_bar.showMessage(f"Fine-tuning: All layers of {base_model_name} are trainable.", 3000)

                else: # Feature extraction
                    base.trainable = False
                    self.status_bar.showMessage(f"Feature extraction: All layers of {base_model_name} are frozen.", 3000)

                # Add custom head
                head_output = self.create_neural_network_enhanced(input_shape=None, num_classes=num_classes, base_model_output=base_output_tensor)
                model = models.Model(inputs=base_model_input_tensor, outputs=head_output, name=f"{base_model_name}_TransferLearned")

            else: # Model from scratch
                 model = self.create_neural_network_enhanced(input_shape, num_classes)

            if model is None:
                self.progress_bar.setRange(0,100); self.progress_bar.setValue(0)
                return # Error creating model

            model.summary(print_fn=lambda x: self.metrics_text.append(x)) # Show summary in metrics text
            
            # --- Optimizer and Compilation ---
            optimizer = self._create_optimizer()
            loss_function = self.nn_loss_combo.currentText()
            
            compile_metrics = []
            if self.nn_task_type == "Classification":
                # For sparse labels (integers), use SparseCategoricalAccuracy
                if loss_function == 'sparse_categorical_crossentropy':
                    compile_metrics.append(tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'))
                else: # For one-hot or binary (0/1)
                    compile_metrics.append('accuracy')
                # Add F1 score for classification if possible (tf.addons or custom)
                # For simplicity, stick to accuracy for now.
            elif self.nn_task_type == "Regression":
                compile_metrics.extend(['mae', 'mse']) # Mean Absolute Error, Mean Squared Error
            
            model.compile(optimizer=optimizer, loss=loss_function, metrics=compile_metrics)
            self.status_bar.showMessage("Model compiled successfully.")
            QApplication.processEvents()

            # --- Callbacks ---
            keras_callbacks = self._create_callbacks(initial_lr=self.lr_spin.value())

            # --- Image Augmentation ---
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            
            history = None
            if self.image_augmentation_group.isChecked() and len(input_shape) == 3 : # Only for image data
                self.status_bar.showMessage("Setting up image augmentation...")
                QApplication.processEvents()
                
                datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=self.aug_rotation_spin.value() if self.aug_rotation_cb.isChecked() else 0,
                    width_shift_range=0.1, # Example, could add UI for these
                    height_shift_range=0.1,
                    zoom_range=self.aug_zoom_spin.value() if self.aug_zoom_cb.isChecked() else 0.0,
                    horizontal_flip=self.aug_flip_cb.isChecked(),
                    fill_mode='nearest'
                )
                # Need to fit datagen if using featurewise_center or featurewise_std_normalization
                # datagen.fit(X_train_nn) # Not needed for these basic augmentations

                self.status_bar.showMessage(f"Training NN with augmentation for {epochs} epochs...")
                QApplication.processEvents()
                history = model.fit(
                    datagen.flow(X_train_nn, y_train_nn, batch_size=batch_size),
                    steps_per_epoch=len(X_train_nn) // batch_size,
                    epochs=epochs,
                    validation_data=(X_test_nn, y_test_nn),
                    callbacks=keras_callbacks,
                    verbose=0 # Progress handled by callback
                )
            else: # No augmentation or not image data
                self.status_bar.showMessage(f"Training NN for {epochs} epochs...")
                QApplication.processEvents()
                history = model.fit(
                    X_train_nn, y_train_nn,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(X_test_nn, y_test_nn), # Use test set as validation
                    callbacks=keras_callbacks,
                    verbose=0 # Progress handled by callback
                )

            self.current_model = model
            self.current_model_name = model.name
            self.last_task_type = self.nn_task_type # Store task type

            self.status_bar.showMessage("NN training complete. Evaluating on test set...")
            QApplication.processEvents()

            # --- Evaluation ---
            loss, *eval_metrics = model.evaluate(X_test_nn, y_test_nn, verbose=0)
            
            metrics_text_eval = f"\n--- Test Set Evaluation ({model.name}) ---\n"
            metrics_text_eval += f"Test Loss: {loss:.4f}\n"
            for metric_name, metric_value in zip(model.metrics_names[1:], eval_metrics): # Skip loss as it's already printed
                metrics_text_eval += f"Test {metric_name.replace('_', ' ').title()}: {metric_value:.4f}\n"

            # Detailed classification metrics if applicable
            if self.nn_task_type == "Classification":
                y_pred_proba = model.predict(X_test_nn)
                if num_classes == 2 or (y_pred_proba.shape[-1] == 1 and loss_function != 'sparse_categorical_crossentropy'): # Binary or single output sigmoid
                    y_pred_classes = (y_pred_proba > 0.5).astype("int32")
                else: # Multi-class
                    y_pred_classes = np.argmax(y_pred_proba, axis=1)
                
                # If y_test_nn was one-hot encoded, convert it back for classification_report
                y_test_for_report = y_test_nn
                if len(y_test_nn.shape) > 1 and y_test_nn.shape[1] > 1: # One-hot
                    y_test_for_report = np.argmax(y_test_nn, axis=1)

                try:
                    from sklearn.metrics import classification_report, f1_score
                    report = classification_report(y_test_for_report, y_pred_classes)
                    metrics_text_eval += "\nClassification Report:\n" + report + "\n"
                    
                    # F1-score (macro average example)
                    f1 = f1_score(y_test_for_report, y_pred_classes, average='macro', zero_division=0)
                    metrics_text_eval += f"F1 Score (Macro): {f1:.4f}\n"

                except Exception as report_err:
                    metrics_text_eval += f"Could not generate classification report: {report_err}\n"
            
            self.metrics_text.append(metrics_text_eval)
            self.plot_training_history(history) # Plot loss/accuracy curves

            self.progress_bar.setRange(0,100); self.progress_bar.setValue(100)
            self.status_bar.showMessage(f"Neural Network '{model.name}' trained and evaluated.")

        except Exception as e:
            self.show_error(f"Error during NN training or evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            self.progress_bar.setRange(0,100); self.progress_bar.setValue(0)
            self.status_bar.showMessage("NN training/evaluation failed.")
            self.current_model = None
            self.current_model_name = "Neural Network (Error)"

    def create_progress_callback(self):
    # Ensure this is defined within your MLCourseGUI class
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, gui_progress_bar, gui_status_bar, total_epochs):
                super().__init__()
                self.gui_progress_bar = gui_progress_bar
                self.gui_status_bar = gui_status_bar
                self.total_epochs = total_epochs

            def on_train_begin(self, logs=None):
                self.gui_progress_bar.setRange(0, self.total_epochs)
                self.gui_progress_bar.setValue(0)

            def on_epoch_end(self, epoch, logs=None):
                self.gui_progress_bar.setValue(epoch + 1)
                logs_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                self.gui_status_bar.showMessage(f"Epoch {epoch+1}/{self.total_epochs} complete. {logs_str}")
                QApplication.processEvents() # Allow UI to update

            def on_train_end(self, logs=None):
                self.gui_progress_bar.setValue(self.total_epochs)
                self.gui_status_bar.showMessage("Training finished.", 3000)
                QApplication.processEvents()

        return ProgressCallback(self.progress_bar, self.status_bar, self.epochs_spin.value())


    def plot_training_history(self, history):
        if history is None or not history.history:
            self.figure.clear()
            self.figure.text(0.5, 0.5, 'No training history to plot.', ha='center', va='center')
            self.canvas.draw()
            return

        self.figure.clear()
        history_dict = history.history
    
    # Determine primary metric (accuracy or mae/mse)
        primary_metric = None
        val_primary_metric = None
        if 'accuracy' in history_dict:
            primary_metric = 'accuracy'
            if 'val_accuracy' in history_dict: val_primary_metric = 'val_accuracy'
        elif 'sparse_categorical_accuracy' in history_dict: # Keras might use this name
            primary_metric = 'sparse_categorical_accuracy'
            if 'val_sparse_categorical_accuracy' in history_dict: val_primary_metric = 'val_sparse_categorical_accuracy'
        elif 'mae' in history_dict: # For regression
            primary_metric = 'mae'
            if 'val_mae' in history_dict: val_primary_metric = 'val_mae'
        elif 'mse' in history_dict: # Fallback for regression
            primary_metric = 'mse'
            if 'val_mse' in history_dict: val_primary_metric = 'val_mse'

        epochs_range = range(1, len(history_dict['loss']) + 1)

    # Plot 1: Loss
        ax1 = self.figure.add_subplot(1, 2, 1 if primary_metric else 1) # If no metric, loss takes full space
        ax1.plot(epochs_range, history_dict['loss'], label='Training Loss', marker='.')
        if 'val_loss' in history_dict:
            ax1.plot(epochs_range, history_dict['val_loss'], label='Validation Loss', marker='.')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot 2: Primary Metric (if available)
        if primary_metric:
            ax2 = self.figure.add_subplot(1, 2, 2)
            ax2.plot(epochs_range, history_dict[primary_metric], label=f'Training {primary_metric.title()}', marker='.')
            if val_primary_metric and val_primary_metric in history_dict:
                ax2.plot(epochs_range, history_dict[val_primary_metric], label=f'Validation {primary_metric.title()}', marker='.')
            ax2.set_title(f'Training and Validation {primary_metric.title()}')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel(primary_metric.title())
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
        
        try:
            self.figure.tight_layout()
        except Exception:
            pass
        self.canvas.draw()



    # --- Utility Methods (Unchanged except clear_data) ---
    def show_error(self, message):
        """Show error message dialog (Unchanged)"""
        QMessageBox.critical(self, "Error", message)

    def clear_data(self):
        """Clear all loaded and processed data (Ensure feature_names cleared)."""
        self.original_data = None
        self.X = None; self.y = None
        self.X_train = None; self.X_test = None
        self.y_train = None; self.y_test = None
        if hasattr(self, 'target_column_name'): del self.target_column_name
        if hasattr(self, 'feature_names'): del self.feature_names # Clear feature names
        self.status_bar.showMessage("Data cleared.")
        self.clear_results()

    def clear_results(self):
         """Clear model, metrics, and visualization (Unchanged)."""
         self.current_model = None
         self.current_model_name = None
         self.last_trained_params = {}
         self.last_task_type = 'unknown'
         self.metrics_text.clear()
         self.figure.clear()
         # Add try-except for draw in case canvas is invalid state
         try:
             self.canvas.draw()
         except Exception as draw_err:
             print(f"Warning: Error during canvas draw on clear_results: {draw_err}")
         self.progress_bar.setValue(0)
    
    


# --- Main Application Runner (Unchanged) ---
def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    # Optional High DPI scaling attributes
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    # QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()