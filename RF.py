import sys
import gc
import textwrap
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

try:
    import openpyxl
except ImportError:
    openpyxl = None

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QGroupBox, QLabel, QFileDialog, QMessageBox, QTextBrowser, QProgressBar,
    QSpinBox, QDialog, QLineEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter


def MSC(data_x, ref_spectrum=None):
    """
    Performs Multiplicative Scatter Correction.
    - If ref_spectrum is None, it's calculated from data_x (for training set).
    - If ref_spectrum is provided, it's used to correct data_x (for test set).
    """
    if ref_spectrum is None:
        ref_spectrum = np.mean(data_x, axis=0)

    corrected_x = np.zeros_like(data_x)
    for i in range(data_x.shape[0]):
        lin = LinearRegression().fit(ref_spectrum.reshape(-1, 1), data_x[i, :].reshape(-1, 1))
        k = lin.coef_[0, 0]
        b = lin.intercept_[0]
        corrected_x[i, :] = (data_x[i, :] - b) / (k if np.abs(k) > 1e-9 else 1e-9)
    return corrected_x, ref_spectrum


def SG_smoothing(data_x, window_length=11, polyorder=2):
    """ Performs Savitzky-Golay smoothing. """
    if data_x.shape[1] < window_length:
        window_length = data_x.shape[1] - (1 if data_x.shape[1] % 2 == 0 else 2)
    if window_length < 3: return data_x

    return savgol_filter(data_x, window_length=window_length, polyorder=polyorder, deriv=0, axis=1)


def predict_with_interval(model, X, percentile=95):
    selector = model.named_steps.get('selector')
    regressor = model.named_steps.get('regressor')
    if not hasattr(regressor, 'estimators_'):
        preds = model.predict(X)
        return preds, preds, preds
    if selector:
        X_transformed = selector.transform(X)
    else:
        X_transformed = X
    individual_tree_preds = np.array([tree.predict(X_transformed) for tree in regressor.estimators_])
    mean_predictions = np.mean(individual_tree_preds, axis=0)
    lower_percent = (100 - percentile) / 2
    upper_percent = 100 - lower_percent
    lower_bound = np.percentile(individual_tree_preds, lower_percent, axis=0)
    upper_bound = np.percentile(individual_tree_preds, upper_percent, axis=0)
    return mean_predictions, lower_bound, upper_bound


class ShapPlotDialog(QDialog):
    def __init__(self, figure, parent=None):
        super(ShapPlotDialog, self).__init__(parent)
        self.setWindowTitle("SHAP Summary Plot")
        self.setModal(True)
        self.canvas = FigureCanvas(figure)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.resize(800, 600)


class RFTrainingThread(QThread):
    update_log = pyqtSignal(str)
    training_finished = pyqtSignal(object)
    progress_updated = pyqtSignal(int)

    def __init__(self, X_train, y_train, n_trees_min, n_trees_max, cv_folds, max_depth_list):
        super().__init__()
        self.X_train = np.array(X_train, copy=True)
        self.y_train = np.array(y_train, copy=True)
        self.n_trees_min = n_trees_min
        self.n_trees_max = n_trees_max
        self.cv_folds = cv_folds
        self.max_depth_list = max_depth_list
        self.abort_flag = False
        self._base_model = None
        self._search = None

    def run(self):
        try:
            if self.abort_flag: return
            self._safe_log("🎯 Starting base model training for feature selection...")
            self._base_model = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42,
                                                     verbose=0)
            self._base_model.fit(self.X_train, self.y_train)
            if self.abort_flag: return
            self.progress_updated.emit(30)

            self._safe_log("🔍 Performing feature selection...")
            selector = SelectFromModel(self._base_model, prefit=True, threshold="1.25*median")
            X_reduced = selector.transform(self.X_train)
            if self.abort_flag: return
            self.progress_updated.emit(50)

            self._safe_log(f"⚙️ Starting hyperparameter optimization with {self.cv_folds}-fold CV...")

            param_dist = {
                'n_estimators': randint(self.n_trees_min, self.n_trees_max),
                'max_depth': self.max_depth_list,
                'min_samples_split': randint(2, 10),
                'min_samples_leaf': randint(1, 5),
                'max_features': ['sqrt', 'log2']
            }

            rf_search_estimator = RandomForestRegressor(n_jobs=-1, random_state=42)
            self._search = RandomizedSearchCV(
                estimator=rf_search_estimator, param_distributions=param_dist,
                n_iter=20, cv=self.cv_folds, scoring='r2', verbose=0, n_jobs=1
            )
            self._search.fit(X_reduced, self.y_train)
            if self.abort_flag: return
            self.progress_updated.emit(80)

            final_model = Pipeline([('selector', selector), ('regressor', self._search.best_estimator_)])

            if not self.abort_flag:
                self.training_finished.emit(final_model)
                self.progress_updated.emit(100)
                self._safe_log("✅ Training completed!")

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self._safe_log(f"❌ Training error: {str(e)}\nDetails:\n{error_details}")
            if not self.abort_flag: self.training_finished.emit(None)
        finally:
            del self._base_model, self._search, X_reduced, rf_search_estimator
            gc.collect()

    def _safe_log(self, msg):
        try:
            if not self.signalsBlocked(): self.update_log.emit(str(msg)[:500])
        except (RuntimeError, Exception) as e:
            print(f"Log Error (Thread): {e}")

    def request_abort(self):
        self._safe_log("🛑 Abort requested...")
        self.abort_flag = True


class ForestTraining(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_data = None
        self.model = None
        self.X_train, self.y_train, self.X_test, self.y_test = [None] * 4
        self.X_train_processed, self.X_test_processed = None, None
        self.thread = None
        self._init_ui()

    def _init_ui(self):
        self.setWindowTitle("Intelligent Spectral Analysis System v3.0 - Random Forest Trainer")
        self.setGeometry(100, 100, 1350, 1380)
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)
        sidebar = QWidget()
        sidebar.setFixedWidth(320)
        sidebar.setStyleSheet(
            "QWidget { background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #2c3e50, stop:1 #34495e); border-right: 1px solid #3a506b; }")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 20, 15, 20)
        sidebar_layout.setSpacing(20)
        title = QLabel("Control Panel")
        title.setStyleSheet("color: #7fcdff; font: bold 20px 'Segoe UI'; padding-bottom: 10px; border: none;")
        title.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(title)
        control_group = QGroupBox("Data & Training")
        control_group.setStyleSheet(
            "QGroupBox { color: white; font: bold 14px 'Segoe UI'; border: 1px solid #4a5568; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px 5px 5px; }")
        control_layout = QVBoxLayout()
        control_layout.setSpacing(15)
        self.btn_load = QPushButton("📂 Load Data")
        self.btn_load.setStyleSheet(self.button_style())
        self.btn_load.clicked.connect(self._load_data)
        control_layout.addWidget(self.btn_load)

        # Parameter Group
        params_group = QGroupBox("")
        params_group.setStyleSheet(
            "QGroupBox { color: white; font: bold 13px 'Segoe UI'; border: none; margin-top: 0px;} QGroupBox::title {subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px 5px 0px;}")
        params_layout = QVBoxLayout(params_group)
        params_layout.setSpacing(10)

        trees_group_layout = QHBoxLayout()
        trees_label = QLabel("Trees (Min/Max):")
        trees_label.setStyleSheet("color: white; font: 12px 'Segoe UI'; border: none;")
        self.spin_trees_min = QSpinBox()
        self.spin_trees_min.setRange(50, 1000);
        self.spin_trees_min.setValue(200)
        self.spin_trees_min.setStyleSheet(self.spinbox_style())
        self.spin_trees_max = QSpinBox()
        self.spin_trees_max.setRange(100, 5000);
        self.spin_trees_max.setValue(500)
        self.spin_trees_max.setStyleSheet(self.spinbox_style())
        trees_group_layout.addWidget(trees_label);
        trees_group_layout.addWidget(self.spin_trees_min);
        trees_group_layout.addWidget(self.spin_trees_max)
        params_layout.addLayout(trees_group_layout)
        self.spin_trees_min.valueChanged.connect(lambda val: self.spin_trees_max.setMinimum(val + 1))
        self.spin_trees_max.valueChanged.connect(lambda val: self.spin_trees_min.setMaximum(val - 1))

        depth_label = QLabel("Max Depth List:")
        depth_label.setStyleSheet("color: white; font: 12px 'Segoe UI'; border: none;")
        self.line_max_depth = QLineEdit("None, 10, 15, 20")
        self.line_max_depth.setToolTip("Comma-separated integers or 'None'")
        self.line_max_depth.setStyleSheet(
            "QLineEdit { background-color: #3a506b; color: white; border: 1px solid #4a5568; border-radius: 4px; padding: 5px; font: 12px 'Segoe UI'; }")
        params_layout.addWidget(depth_label)
        params_layout.addWidget(self.line_max_depth)

        cv_layout = QHBoxLayout()
        cv_label = QLabel("CV Folds:")
        cv_label.setStyleSheet("color: white; font: 12px 'Segoe UI'; border: none;")
        self.spin_cv_folds = QSpinBox()
        self.spin_cv_folds.setRange(2, 20);
        self.spin_cv_folds.setValue(5)
        self.spin_cv_folds.setStyleSheet(self.spinbox_style())
        cv_layout.addWidget(cv_label)
        cv_layout.addWidget(self.spin_cv_folds)
        params_layout.addLayout(cv_layout)

        control_layout.addWidget(params_group)

        self.btn_train = QPushButton("🚀 Start Training")
        self.btn_train.setStyleSheet(
            self.button_style() + "QPushButton { background: rgba(76, 175, 80, 0.3); color: #98e09b; border-color: rgba(76, 175, 80, 0.4); } QPushButton:hover { background: rgba(76, 175, 80, 0.4); } QPushButton:pressed { background: rgba(76, 175, 80, 0.5); }")
        self.btn_train.clicked.connect(self._start_training)
        self.btn_train.setEnabled(False)
        control_layout.addWidget(self.btn_train)
        self.btn_abort = QPushButton("🛑 Abort Training")
        self.btn_abort.setStyleSheet(
            self.button_style() + "QPushButton { background: rgba(244, 67, 54, 0.3); color: #f89a92; border-color: rgba(244, 67, 54, 0.4); } QPushButton:hover { background: rgba(244, 67, 54, 0.4); } QPushButton:pressed { background: rgba(244, 67, 54, 0.5); }")
        self.btn_abort.clicked.connect(self._abort_training)
        self.btn_abort.setEnabled(False)
        control_layout.addWidget(self.btn_abort)
        self.btn_shap = QPushButton("🧠 Calculate SHAP Values")
        self.btn_shap.setStyleSheet(
            self.button_style() + "QPushButton { background: rgba(255, 154, 139, 0.3); color: #ff9a8b; border-color: rgba(255, 154, 139, 0.4); } QPushButton:hover { background: rgba(255, 154, 139, 0.4); }")
        self.btn_shap.clicked.connect(self._show_shap_popup)
        self.btn_shap.setEnabled(False)
        control_layout.addWidget(self.btn_shap)
        self.progress = QProgressBar()
        self.progress.setStyleSheet(
            "QProgressBar { border: 1px solid #4a5568; border-radius: 5px; background-color: #3a506b; text-align: center; color: white; font: 12px 'Segoe UI'; } QProgressBar::chunk { background-color: #2196F3; border-radius: 5px; background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1e88e5, stop:1 #64b5f6); }")
        self.progress.setRange(0, 100);
        self.progress.setValue(0)
        control_layout.addWidget(self.progress)
        control_group.setLayout(control_layout)
        sidebar_layout.addWidget(control_group)
        sidebar_layout.addStretch(1)
        main_layout.addWidget(sidebar)
        right_panel = QWidget();
        right_panel.setStyleSheet("background: #f8f9fa;")
        right_layout = QVBoxLayout(right_panel);
        right_layout.setContentsMargins(20, 20, 20, 20);
        right_layout.setSpacing(15)
        vis_group = QGroupBox("Analysis Results")
        vis_group.setStyleSheet(
            "QGroupBox { font: bold 14px 'Segoe UI'; color: #343a40; border: 1px solid #dee2e6; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px 5px 5px; }")
        vis_layout = QVBoxLayout()
        self.figure = Figure(figsize=(8, 18), dpi=100)
        self.figure.patch.set_facecolor('#f8f9fa')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("border: none;")
        vis_layout.addWidget(self.canvas)
        vis_group.setLayout(vis_layout)
        right_layout.addWidget(vis_group, stretch=2)
        log_group = QGroupBox("System Log")
        log_group.setStyleSheet(
            "QGroupBox { font: bold 14px 'Segoe UI'; color: #343a40; border: 1px solid #dee2e6; border-radius: 5px; margin-top: 10px; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px 5px 5px; }")
        log_layout = QVBoxLayout()
        self.log_area = QTextBrowser()
        self.log_area.setStyleSheet(
            "QTextBrowser { background: white; border: 1px solid #e9ecef; border-radius: 4px; padding: 10px; font-family: 'Consolas', 'Courier New', monospace; font-size: 12px; color: #495057; line-height: 1.4; }")
        self.log_area.setReadOnly(True)
        self.log_area.setPlaceholderText("Welcome to Spectral Analysis System! Logs will appear here.")
        log_layout.addWidget(self.log_area)
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group, stretch=1)
        main_layout.addWidget(right_panel, stretch=1)

    def button_style(self):
        return "QPushButton { background: rgba(78, 205, 196, 0.2); color: #4ecdc4; border: 1px solid rgba(78, 205, 196, 0.4); border-radius: 5px; padding: 10px; font: bold 13px 'Segoe UI'; min-height: 30px; } QPushButton:hover { background: rgba(78, 205, 196, 0.3); border: 1px solid rgba(78, 205, 196, 0.6); } QPushButton:pressed { background: rgba(78, 205, 196, 0.4); } QPushButton:disabled { background-color: #50687c; color: #8ea4b8; border-color: #5f7a93; }"

    def spinbox_style(self):
        return "QSpinBox { background-color: #3a506b; color: white; border: 1px solid #4a5568; border-radius: 4px; padding: 5px; font: 13px 'Segoe UI'; } QSpinBox:focus { border: 1px solid #4ecdc4; }"

    def _start_training(self):
        if not isinstance(self.X_train, np.ndarray):
            QMessageBox.warning(self, "No Data", "Please load data before training.");
            return
        if self.thread and self.thread.isRunning():
            QMessageBox.warning(self, "Busy", "Training is in progress.");
            return

        try:
            n_trees_min, n_trees_max = self.spin_trees_min.value(), self.spin_trees_max.value()
            cv_folds = self.spin_cv_folds.value()

            depth_text = self.line_max_depth.text().strip()
            if not depth_text: raise ValueError("Max Depth List cannot be empty.")
            max_depth_list = []
            for item in depth_text.split(','):
                item = item.strip().lower()
                if item == 'none':
                    max_depth_list.append(None)
                else:
                    max_depth_list.append(int(item))
        except ValueError as e:
            QMessageBox.warning(self, "Parameter Error",
                                f"Invalid Max Depth List: {e}. Please use comma-separated integers or 'None'.")
            return

        self._log("\n🚀 Initiating training process...")
        self.btn_train.setEnabled(False);
        self.btn_abort.setEnabled(True);
        self.btn_load.setEnabled(False);
        self.btn_shap.setEnabled(False)
        self.progress.setValue(0);
        self.figure.clear();
        self.canvas.draw();
        self.model = None

        try:
            self._log("🛠️ Applying spectral preprocessing pipeline...")


            self._log("   - Applying Multiplicative Scatter Correction (MSC)...")
            X_train_msc, ref_spec = MSC(self.X_train)
            X_test_msc, _ = MSC(self.X_test, ref_spectrum=ref_spec)


            self._log("   - Applying Savitzky-Golay Smoothing...")
            X_train_sg = SG_smoothing(X_train_msc)
            X_test_sg = SG_smoothing(X_test_msc)


            self._log("   - Applying StandardScaler...")
            scaler = StandardScaler()
            self.X_train_processed = scaler.fit_transform(X_train_sg)
            self.X_test_processed = scaler.transform(X_test_sg)

            self.scaler = scaler

            self._log("✅ Preprocessing complete.")
        except Exception as e:
            self._log(f"❌ Error during preprocessing: {e}")
            self.btn_train.setEnabled(True);
            self.btn_abort.setEnabled(False);
            self.btn_load.setEnabled(True)
            return

        self._log(f"   - Hyperparameter search range for trees: {n_trees_min} to {n_trees_max}")
        self._log(f"   - Max Depth search space: {max_depth_list}")
        self._log(f"   - Using {cv_folds}-fold cross-validation.")

        self.thread = RFTrainingThread(self.X_train_processed, self.y_train, n_trees_min, n_trees_max, cv_folds,
                                       max_depth_list)

        self.thread.update_log.connect(self._log)
        self.thread.progress_updated.connect(self.progress.setValue)
        self.thread.training_finished.connect(self._on_training_done)
        self.thread.finished.connect(self._training_thread_finished)
        self.thread.start()

    def _load_data(self):
        try:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Data File",
                "",
                "Data Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)"
            )
            if not path:
                self._log("ℹ️ File selection cancelled.")
                return

            self._log(f"🔄 Loading data from: {path}...")
            QApplication.processEvents()

            df = None
            if path.lower().endswith(('.xlsx', '.xls')):
                if openpyxl is None:
                    raise ImportError(
                        "Reading Excel files requires the 'openpyxl' library. Please install it (pip install openpyxl).")
                df = pd.read_excel(path, header=0, engine='openpyxl')

            else:
                encodings = ['utf-8', 'gbk', 'latin1', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(path, header=0, encoding=encoding)
                        self._log(f"✓ Detected encoding: {encoding}")
                        break
                    except (UnicodeDecodeError, Exception):
                        continue
                else:
                    raise ValueError(
                        "Failed to automatically detect file encoding. Please ensure the file is UTF-8, GBK, or Latin1 encoded.")

            if df is None:
                raise ValueError("Could not load dataframe.")
            if df.empty:
                raise ValueError("The loaded file is empty.")
            if df.shape[1] < 2:
                raise ValueError("Data must contain at least two columns (features + 1 label column).")

            df = df.apply(pd.to_numeric, errors='coerce')
            if df.isnull().values.any():
                nan_cols = df.columns[df.isnull().any()].tolist()
                self._log(f"⚠️ Warning: Found missing values in columns: {nan_cols}. Filling with column medians.")
                df.fillna(df.median(), inplace=True)

            label_col = df.columns[-1]
            X = df.iloc[:, :-1].values.astype(np.float32)
            y = df.iloc[:, -1].values.astype(np.float32)

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2,
                                                                                    random_state=42)

            if len(self.X_train) == 0 or len(self.X_test) == 0:
                raise ValueError("Not enough data to perform train/test split. Need at least 2 samples.")

            self.current_data = {
                'features': df.columns[:-1].tolist()
            }
            self.btn_train.setEnabled(True)
            self.btn_load.setEnabled(False)
            self.btn_shap.setEnabled(False)
            self._log(textwrap.dedent(f"""
            ------------------------------
            ✅ Data loaded successfully!
            Shape: {df.shape[0]:,} rows, {df.shape[1]} columns
            Label Column: '{label_col}'
            Train/Test Split: {len(self.X_train)} / {len(self.X_test)}
            ------------------------------"""))
            self.figure.clear()
            self.canvas.draw()

        except (ImportError, ValueError) as e:
            QMessageBox.critical(self, "Data Error", str(e))
            self._log(f"❌ Data Error: {str(e)}")
            self._reset_state()
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            QMessageBox.critical(self, "Error",
                                 f"An unexpected error occurred during data loading:\n{str(e)}")
            self._log(f"❌ Unexpected Error: {str(e)}\n{error_details}")
            self._reset_state()
    def _on_training_done(self, model):
        if self.thread is None or model is None: self._log("❌ Training failed or aborted."); return
        try:
            self.model = model
            self._log("\n🔄 Evaluating model performance...")
            QApplication.processEvents()

            y_pred_test, pi_lower, pi_upper = predict_with_interval(model, self.X_test_processed)
            y_pred_train = model.predict(self.X_train_processed)

            r2_test = r2_score(self.y_test, y_pred_test);
            rmse_test = np.sqrt(mean_squared_error(self.y_test, y_pred_test));
            mae_test = mean_absolute_error(self.y_test, y_pred_test)
            r2_train = r2_score(self.y_train, y_pred_train);
            rmse_train = np.sqrt(mean_squared_error(self.y_train, y_pred_train));
            mae_train = mean_absolute_error(self.y_train, y_pred_train)

            log_output = textwrap.dedent(f"""
                ------------------------------
                📊 Final Evaluation Results:
                --- Training Set ---
                ➤ R² Score: {r2_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}
                --- Test Set ---
                ➤ R² Score: {r2_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}
                ------------------------------
                """)
            self._log(log_output)
            self._plot_results(y_pred_test, pi_lower, pi_upper)
            self._save_model()
            self.btn_shap.setEnabled(True)
        except Exception as e:
            import traceback;
            error_details = traceback.format_exc();
            self._log(f"❌ Error during post-training: {str(e)}\n{error_details}");
            QMessageBox.critical(self, "Post-Training Error", f"Error during evaluation or plotting:\n{str(e)}")

    def _show_shap_popup(self):
        if not self.model:
            QMessageBox.warning(self, "SHAP Error", "A trained model is required.")
            return
        self._log("\n🧠 Calculating SHAP values (this may take a moment)...")
        self.btn_shap.setEnabled(False);
        QApplication.processEvents()
        try:
            selector = self.model.named_steps['selector'];
            regressor = self.model.named_steps['regressor']


            X_train_subset_transformed = selector.transform(self.X_train_processed[:100])
            X_test_subset_transformed = selector.transform(self.X_test_processed[:100])

            all_feature_names = np.array(self.current_data['features']);
            selected_feature_names = all_feature_names[selector.get_support()]

            explainer = shap.TreeExplainer(regressor, X_train_subset_transformed)
            shap_values = explainer.shap_values(X_test_subset_transformed)

            self._log("✅ SHAP calculation complete. Displaying plot in new window...")
            plt.close('all')
            shap.summary_plot(shap_values, X_test_subset_transformed, feature_names=selected_feature_names, show=False)
            shap_figure = plt.gcf()
            dialog = ShapPlotDialog(shap_figure, self)
            dialog.exec_()
        except Exception as e:
            import traceback;
            error_msg = f"SHAP calculation failed: {str(e)}\n\n{traceback.format_exc()}";
            QMessageBox.critical(self, "SHAP Error", error_msg);
            self._log(f"❌ {error_msg}")
        finally:
            self.btn_shap.setEnabled(True)

    def _plot_results(self, y_pred_test, pi_lower, pi_upper):
        if self.y_test is None or y_pred_test is None:
            self._log("ℹ️ Skipping plotting: Missing data.");
            return
        try:
            self.figure.clear();
            self.figure.patch.set_facecolor('#f8f9fa')
            residuals_test = self.y_test - y_pred_test
            ax1 = self.figure.add_subplot(221)
            ax1.scatter(self.y_test, y_pred_test, alpha=0.6, edgecolors='w', s=25, c='#007bff', label='Mean Prediction')
            lims = [np.nanmin(self.y_test), np.nanmax(self.y_test)]
            ax1.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label='Ideal (y=x)')
            if pi_lower is not None and pi_upper is not None:
                sort_indices = np.argsort(y_pred_test)
                ax1.fill_between(y_pred_test[sort_indices], pi_lower[sort_indices], pi_upper[sort_indices],
                                 color='#007bff', alpha=0.2, label='95% PI')
            ax1.set_xlabel("Actual Values", fontsize=18);
            ax1.set_ylabel("Predicted Values", fontsize=18);
            ax1.set_title("Prediction vs Actual", fontsize=18)
            ax1.grid(True, linestyle='--', alpha=0.6);
            ax1.tick_params(labelsize=10);
            ax1.legend(fontsize=9)
            ax2 = self.figure.add_subplot(222)
            stats.probplot(residuals_test, dist="norm", plot=ax2)
            ax2.get_lines()[0].set_markerfacecolor('mediumpurple');
            ax2.get_lines()[0].set_markeredgecolor('indigo');
            ax2.get_lines()[1].set_color('darkred')
            ax2.set_title("Normal Q-Q Plot of Residuals", fontsize=18);
            ax2.set_xlabel("Theoretical Quantiles", fontsize=18);
            ax2.set_ylabel("Sample Quantiles", fontsize=18)
            ax2.tick_params(labelsize=10);
            ax2.grid(True, linestyle='--', alpha=0.6)
            ax3 = self.figure.add_subplot(223)
            ax3.scatter(y_pred_test, residuals_test, alpha=0.6, edgecolors='w', s=25, c='#ff7f0e');
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.75)
            ax3.set_xlabel("Predicted Value", fontsize=18);
            ax3.set_ylabel("Residual", fontsize=18);
            ax3.set_title("Residual Plot", fontsize=18)
            ax3.grid(True, linestyle='--', alpha=0.6);
            ax3.tick_params(labelsize=10)
            ax4 = self.figure.add_subplot(224)
            ax4.hist(residuals_test, bins=30, alpha=0.7, color='mediumseagreen', edgecolor='black');
            ax4.set_xlabel("Residual Value", fontsize=18);
            ax4.set_ylabel("Frequency", fontsize=18)
            ax4.set_title("Residual Distribution", fontsize=18);
            ax4.grid(True, linestyle='--', alpha=0.6, axis='y');
            ax4.tick_params(labelsize=10)
            self.figure.tight_layout(pad=3.0);
            self.canvas.draw();
            self._log("📊 Plots updated successfully.")
        except Exception as plot_err:
            import traceback;
            error_details = traceback.format_exc();
            self._log(f"❌ Plotting Error: {str(plot_err)}\n{error_details}")

    def _training_thread_finished(self):
        self._log("🧵 Training thread finished.");
        self.btn_train.setEnabled(True);
        self.btn_abort.setEnabled(False);
        self.btn_load.setEnabled(True)
        self.progress.setValue(100 if self.model else 0);
        self.thread = None

    def _save_model(self):
        if not self.model: return
        default_fn = f"RF_Model_{pd.Timestamp.now():%Y%m%d_%H%M}.joblib"
        path, _ = QFileDialog.getSaveFileName(self, "Save Trained Model", default_fn, "Joblib (*.joblib)")
        if not path: self._log("ℹ️ Model save cancelled."); return
        try:
            self._log(f"💾 Saving model to: {path}...")
            y_pred = self.model.predict(self.X_test_processed)
            performance = {'R2': r2_score(self.y_test, y_pred),
                           'RMSE': np.sqrt(mean_squared_error(self.y_test, y_pred))}
            save_data = {'model': self.model, 'scaler': self.scaler,
                         'metadata': {'feature_names': self.current_data['features'],
                                      'selected_feature_indices': self.model.named_steps['selector'].get_support(
                                          indices=True).tolist(),
                                      'save_date': pd.Timestamp.now().isoformat(), },
                         'performance_on_test_set': performance}
            joblib.dump(save_data, path, compress=3)
            self._log(f"✅ Model saved successfully!");
            QMessageBox.information(self, "Save Successful", f"Model saved to:\n{path}")
        except Exception as e:
            import traceback;
            error_details = traceback.format_exc();
            QMessageBox.critical(self, "Save Error", f"Failed to save model:\n{str(e)}");
            self._log(f"❌ Save Error: {str(e)}\n{error_details}")

    def _abort_training(self):
        if self.thread and self.thread.isRunning(): self._log(
            "⏹️ Sending abort signal..."); self.thread.request_abort(); self.btn_abort.setEnabled(False)

    def _log(self, message):
        self.log_area.append(str(message).strip());
        self.log_area.ensureCursorVisible();
        QApplication.processEvents()

    def _reset_state(self):
        self.current_data, self.model = None, None;
        self.X_train, self.y_train, self.X_test, self.y_test = [None] * 4;
        self.X_train_processed, self.X_test_processed = None, None
        self.btn_train.setEnabled(False);
        self.btn_load.setEnabled(True);
        self.btn_shap.setEnabled(False)
        self.progress.setValue(0);
        self.figure.clear();
        self.canvas.draw()

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            reply = QMessageBox.question(self, 'Confirm Exit', "Training is in progress. Exit anyway?",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                self._abort_training(); self.thread.wait(500); event.accept()
            else:
                event.ignore()
        else:
            event.accept()


if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            import ctypes; ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
        except (ImportError, Exception):
            pass
    app = QApplication(sys.argv)
    window = ForestTraining()
    window.show()
    sys.exit(app.exec_())
