import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QGroupBox, QTabWidget, QTextBrowser, QGridLayout, QFileDialog, QMessageBox, QSpinBox
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.signal import savgol_filter
import joblib



class SpectralPreprocessor:
    def __init__(self, window_length=11, polyorder=2):
        self.scaler = StandardScaler()
        self.mean_spectrum_ = None
        self.window_length = window_length
        self.polyorder = polyorder

    def _msc(self, data_x):
        corrected_X = np.zeros_like(data_x)
        for i in range(data_x.shape[0]):
            lin = LinearRegression().fit(self.mean_spectrum_.reshape(-1, 1), data_x[i, :].reshape(-1, 1))
            k, b = lin.coef_[0, 0], lin.intercept_[0]
            corrected_X[i, :] = (data_x[i, :] - b) / (k if np.abs(k) > 1e-9 else 1e-9)
        return corrected_X

    def _sg_smoothing(self, data_x):
        wl = min(self.window_length, data_x.shape[1])
        if wl % 2 == 0: wl -= 1
        if wl < 3: return data_x
        po = min(self.polyorder, wl - 1)
        return savgol_filter(data_x, window_length=wl, polyorder=po, deriv=0, axis=1)

    def fit_transform(self, X_train):
        self.mean_spectrum_ = np.mean(X_train, axis=0)

        X_train_msc = self._msc(X_train)
        X_train_smoothed = self._sg_smoothing(X_train_msc)
        X_train_scaled = self.scaler.fit_transform(X_train_smoothed)
        return X_train_scaled

    def transform(self, X_test):
        if self.mean_spectrum_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet. Call fit_transform first.")

        X_test_msc = self._msc(X_test)
        X_test_smoothed = self._sg_smoothing(X_test_msc)
        X_test_scaled = self.scaler.transform(X_test_smoothed)
        return X_test_scaled





class PLSAppleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PLS Model Training")
        self.setGeometry(100, 100, 1800, 1000)

        # Main layout
        self.main_layout = QHBoxLayout()

        # Left sidebar
        left_sidebar = QWidget()
        left_sidebar.setFixedWidth(450)
        left_sidebar.setStyleSheet("""
            QWidget {
                background: #f0f2f5;
                border-right: 1px solid #d1d5db;
            }
        """)
        sidebar_layout = QVBoxLayout(left_sidebar)
        sidebar_layout.setContentsMargins(30, 30, 30, 30)
        sidebar_layout.setSpacing(30)

        # Title
        title = QLabel("Training Control")
        title.setStyleSheet("""
            QLabel {
                color: #1a202c;
                font: bold 38px 'Segoe UI';
                padding-bottom: 15px;
                border-bottom: 3px solid #38b2ac;
            }
        """)
        sidebar_layout.addWidget(title)

        button_stylesheet = f"""
            QPushButton {{
                background-color: #4ecdc4;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 15px 20px;
                font: bold 30px 'Segoe UI';
                min-height: 70px;
            }}
            QPushButton:hover {{ background-color: #38b2ac; }}
            QPushButton:pressed {{ background-color: #2c7a7b; }}
            QPushButton:disabled {{ background-color: #bdc3c7; color: #7f8c8d; }}
        """

        groupbox_stylesheet = """
            QGroupBox {
                border: 1px solid #d1d5db;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 20px; 
                font: bold 28px 'Segoe UI'; 
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 10px;
                color: #2d3748;
            }
        """

        spinbox_stylesheet = """
            QSpinBox {
                padding: 10px;
                font: bold 24px 'Segoe UI';
                border: 1px solid #d1d5db;
                border-radius: 8px;
                background-color: white;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 30px;
                height: 20px;
            }
        """

        # Data import button
        import_group = QGroupBox("Data Import")
        import_group.setStyleSheet(groupbox_stylesheet)
        import_layout = QVBoxLayout()
        self.import_button = QPushButton("Import Data")
        self.import_button.setStyleSheet(button_stylesheet)
        self.import_button.clicked.connect(self.import_data)
        import_layout.addWidget(self.import_button)
        import_group.setLayout(import_layout)
        sidebar_layout.addWidget(import_group)

        # Training button
        train_group = QGroupBox("Training")
        train_group.setStyleSheet(groupbox_stylesheet)
        train_layout = QVBoxLayout()
        train_layout.setSpacing(20)

        # Latent Variables setting
        param_layout_lv = QHBoxLayout()
        param_label_lv = QLabel("Max Latent Variables:")
        param_label_lv.setStyleSheet("font: 24px 'Segoe UI'; color: #2d3748;")
        self.max_lv_spinbox = QSpinBox()
        self.max_lv_spinbox.setRange(1, 200)
        self.max_lv_spinbox.setValue(20)
        self.max_lv_spinbox.setStyleSheet(spinbox_stylesheet)
        param_layout_lv.addWidget(param_label_lv)
        param_layout_lv.addWidget(self.max_lv_spinbox)
        train_layout.addLayout(param_layout_lv)

        # CV folds setting
        param_layout_cv = QHBoxLayout()
        param_label_cv = QLabel("CV Folds:")
        param_label_cv.setStyleSheet("font: 24px 'Segoe UI'; color: #2d3748;")
        self.cv_folds_spinbox = QSpinBox()
        self.cv_folds_spinbox.setRange(2, 20)
        self.cv_folds_spinbox.setValue(5)
        self.cv_folds_spinbox.setStyleSheet(spinbox_stylesheet)
        param_layout_cv.addWidget(param_label_cv)
        param_layout_cv.addWidget(self.cv_folds_spinbox)
        train_layout.addLayout(param_layout_cv)

        self.train_button = QPushButton("Start Training")
        self.train_button.setStyleSheet(button_stylesheet)
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setEnabled(False)
        train_layout.addWidget(self.train_button)
        train_group.setLayout(train_layout)
        sidebar_layout.addWidget(train_group)

        # Save model button
        save_group = QGroupBox("Save Model")
        save_group.setStyleSheet(groupbox_stylesheet)
        save_layout = QVBoxLayout()
        self.save_button = QPushButton("Save Model")
        save_button_stylesheet = button_stylesheet.replace("#4ecdc4", "#45b7d1").replace("#38b2ac", "#3da8bf").replace(
            "#2c7a7b", "#3597ad")
        self.save_button.setStyleSheet(save_button_stylesheet)
        self.save_button.clicked.connect(self.save_model)
        self.save_button.setEnabled(False)
        save_layout.addWidget(self.save_button)
        save_group.setLayout(save_layout)
        sidebar_layout.addWidget(save_group)

        sidebar_layout.addStretch()
        self.main_layout.addWidget(left_sidebar)

        self.init_visualization_tabs()
        self.setLayout(self.main_layout)

        self.data = None
        self.X = None
        self.y = None
        self.wavelengths = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.pls_best = None
        self.preprocessor = None  # To store the fitted preprocessor

    def init_visualization_tabs(self):
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane { border: 0; }
            QTabBar::tab {
                background: #e9ecef;
                color: #495057;
                padding: 14px 22px; 
                border: 1px solid #dee2e6;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font: 20px 'Segoe UI'; 
            }
            QTabBar::tab:selected {
                background: white;
                color: #4ecdc4;
                font-weight: bold;
            }
        """)
        self.create_main_result_tab()
        self.create_scatter_plot_tab()
        self.create_vip_tab()
        self.create_training_curve_tab()
        self.main_layout.addWidget(self.tab_widget)

    def create_main_result_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(25, 25, 25, 25)
        layout.setSpacing(20)
        metrics_group = QGroupBox("Model Performance Metrics")
        metrics_group.setStyleSheet("""
               QGroupBox { 
                   font: bold 26px 'Segoe UI'; 
                   color: #343a40;
                   border: 1px solid #ced4da; border-radius: 8px;
                   margin-top: 10px; background-color: white; 
               }
               QGroupBox::title { 
                   subcontrol-origin: margin; subcontrol-position: top left;
                   padding: 5px 10px; color: #495057; 
               }
           """)
        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(25)
        metrics_layout.setContentsMargins(20, 30, 20, 20)
        color_r2, color_rmse, color_mae = "#28a745", "#dc3545", "#fd7e14"
        self.metric_cards = {
            'train_r2': self.create_metric_card("Training R²", "-", color_r2),
            'train_rmse': self.create_metric_card("Training RMSE", "-", color_rmse),
            'train_mae': self.create_metric_card("Training MAE", "-", color_mae),
            'test_r2': self.create_metric_card("Test R²", "-", color_r2),
            'test_rmse': self.create_metric_card("Test RMSE", "-", color_rmse),
            'test_mae': self.create_metric_card("Test MAE", "-", color_mae),
        }
        metrics_layout.addWidget(self.metric_cards['train_r2'], 0, 0)
        metrics_layout.addWidget(self.metric_cards['train_rmse'], 0, 1)
        metrics_layout.addWidget(self.metric_cards['train_mae'], 0, 2)
        metrics_layout.addWidget(self.metric_cards['test_r2'], 1, 0)
        metrics_layout.addWidget(self.metric_cards['test_rmse'], 1, 1)
        metrics_layout.addWidget(self.metric_cards['test_mae'], 1, 2)
        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)
        log_group = QGroupBox("Training Log & Details")
        log_group.setStyleSheet(metrics_group.styleSheet())
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(15, 30, 15, 15)
        self.result_text = QTextBrowser()
        self.result_text.setStyleSheet("""
               QTextBrowser { 
                   background-color: #ffffff; border: 1px solid #dee2e6;
                   border-radius: 5px; padding: 15px;
                   font-family: 'Consolas', 'Courier New', monospace;
                   font-size: 22px;
                   color: #343a40; 
               }
           """)
        self.result_text.setMinimumHeight(200)
        log_layout.addWidget(self.result_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        layout.addStretch()
        self.tab_widget.addTab(tab, "📊 Results Summary")

    def create_metric_card(self, title, value, color):
        card = QWidget()
        card.setStyleSheet("background: white; border-radius: 8px; padding: 20px; border: 1px solid #dee2e6;")
        layout = QVBoxLayout(card)
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {color}; font: bold 24px 'Segoe UI';")
        value_label = QLabel(value)
        value_label.setStyleSheet("font: bold 42px 'Segoe UI'; color: #495057;")
        value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        return card

    def create_scatter_plot_tab(self):
        self.scatter_tab = QWidget()
        layout = QVBoxLayout(self.scatter_tab)
        self.scatter_fig = Figure(figsize=(8, 6), dpi=100)
        self.scatter_canvas = FigureCanvas(self.scatter_fig)
        layout.addWidget(self.scatter_canvas)
        self.tab_widget.addTab(self.scatter_tab, "📈 Prediction Scatter")

    def create_vip_tab(self):
        self.vip_tab = QWidget()
        layout = QVBoxLayout(self.vip_tab)
        self.vip_fig = Figure(figsize=(10, 6), dpi=100)
        self.vip_canvas = FigureCanvas(self.vip_fig)
        layout.addWidget(self.vip_canvas)
        self.tab_widget.addTab(self.vip_tab, "⭐ Feature Importance (VIP)")

    def create_training_curve_tab(self):
        self.curve_tab = QWidget()
        layout = QVBoxLayout(self.curve_tab)
        self.curve_fig = Figure(figsize=(8, 6), dpi=100)
        self.curve_canvas = FigureCanvas(self.curve_fig)
        layout.addWidget(self.curve_canvas)
        self.tab_widget.addTab(self.curve_tab, "📉 CV Curves")

    def update_visualizations(self, y_train_pred, y_test_pred, r2_train, rmse_train, mae_train, r2_test, rmse_test,
                              mae_test):
        try:
            self.metric_cards['train_r2'].layout().itemAt(1).widget().setText(f"{r2_train:.4f}")
            self.metric_cards['train_rmse'].layout().itemAt(1).widget().setText(f"{rmse_train:.4f}")
            self.metric_cards['train_mae'].layout().itemAt(1).widget().setText(f"{mae_train:.4f}")
            self.metric_cards['test_r2'].layout().itemAt(1).widget().setText(f"{r2_test:.4f}")
            self.metric_cards['test_rmse'].layout().itemAt(1).widget().setText(f"{rmse_test:.4f}")
            self.metric_cards['test_mae'].layout().itemAt(1).widget().setText(f"{mae_test:.4f}")
        except Exception as e:
            print(f"Error updating metric cards: {e}")
        if self.pls_best is not None and self.X is not None:
            self.plot_scatter(y_train_pred, y_test_pred)
            self.plot_vip()
            self.plot_training_curve()
        else:
            for fig in [self.scatter_fig, self.vip_fig, self.curve_fig]: fig.clear()
            for canvas in [self.scatter_canvas, self.vip_canvas, self.curve_canvas]: canvas.draw()

    def plot_scatter(self, y_train_pred, y_test_pred):
        self.scatter_fig.clear()
        ax = self.scatter_fig.add_subplot(111)
        ax.scatter(self.y_train, y_train_pred, c='#4ecdc4', edgecolors='k', alpha=0.7, label='Training Set')
        ax.scatter(self.y_test, y_test_pred, c='#ff6b6b', edgecolors='k', alpha=0.7, label='Test Set')
        max_val = max(np.max(self.y_train), np.max(y_train_pred), np.max(self.y_test), np.max(y_test_pred))
        min_val = min(np.min(self.y_train), np.min(y_train_pred), np.min(self.y_test), np.min(y_test_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
        ax.set_xlabel('Actual Value', fontsize=14);
        ax.set_ylabel('Predicted Value', fontsize=14);
        ax.set_title('Actual vs Predicted Values', fontsize=16)
        ax.legend(fontsize=12);
        ax.tick_params(axis='both', which='major', labelsize=12);
        ax.grid(True, linestyle='--', alpha=0.6)
        self.scatter_fig.tight_layout();
        self.scatter_canvas.draw()

    def plot_vip(self):
        self.vip_fig.clear()
        ax = self.vip_fig.add_subplot(111)
        w, q = self.pls_best.x_weights_, self.pls_best.y_loadings_
        p, h = w.shape
        vips = np.zeros(p)
        s = np.diag(self.pls_best.x_scores_.T @ self.pls_best.x_scores_ @ q.T @ q).reshape(h, -1)
        total_s = np.sum(s)
        for i in range(p):
            weight = np.array([(w[i, j] / np.linalg.norm(w[:, j])) ** 2 for j in range(h)])
            vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
        sorted_idx = np.argsort(vips)[::-1][:20]
        sorted_vip_scores = vips[sorted_idx]
        vip_wavelengths = self.wavelengths[sorted_idx]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(sorted_vip_scores)))
        ax.barh(range(len(sorted_vip_scores)), sorted_vip_scores, color=colors, edgecolor='k')
        ax.set_yticks(range(len(sorted_vip_scores)));
        ax.set_yticklabels([f"{w:.1f} nm" for w in vip_wavelengths])
        ax.set_xlabel('VIP Score', fontsize=14);
        ax.set_title('Top 20 Important Wavelengths (VIP)', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=12);
        ax.invert_yaxis();
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        self.vip_fig.tight_layout();
        self.vip_canvas.draw()

    def plot_training_curve(self):
        self.curve_fig.clear()
        ax = self.curve_fig.add_subplot(111)
        ax2 = ax.twinx()
        max_lv_from_user = self.max_lv_spinbox.value()
        if self.X_train is None:
            self.curve_canvas.draw()
            return

        n_components_range = range(1, min(max_lv_from_user + 1, self.X_train.shape[1] + 1))
        if not n_components_range: self.curve_canvas.draw(); return
        r2_scores, rmse_scores = [], []

        user_cv_folds = self.cv_folds_spinbox.value()
        kf = KFold(n_splits=min(user_cv_folds, self.X_train.shape[0]), shuffle=True, random_state=42)

        for n in n_components_range:
            pls = PLSRegression(n_components=n)
            if kf.get_n_splits() > self.X_train.shape[0]: r2_scores.append(np.nan); rmse_scores.append(np.nan); continue
            r2_cv = cross_val_score(pls, self.X_train, self.y_train, cv=kf, scoring='r2')
            rmse_cv = np.sqrt(
                -cross_val_score(pls, self.X_train, self.y_train, cv=kf, scoring='neg_mean_squared_error'))
            r2_scores.append(np.mean(r2_cv))
            rmse_scores.append(np.mean(rmse_cv))

        ax.plot(n_components_range, r2_scores, 'o-', color='#4ecdc4', lw=2, label='R² (CV)')
        ax2.plot(n_components_range, rmse_scores, 's-', color='#ff6b6b', lw=2, label='RMSE (CV)')
        ax.set_xlabel('Number of Latent Variables', fontsize=14);
        ax.set_ylabel('R² Score', color='#4ecdc4', fontsize=14);
        ax2.set_ylabel('RMSE', color='#ff6b6b', fontsize=14)
        ax.tick_params(axis='y', labelcolor='#4ecdc4', labelsize=12);
        ax2.tick_params(axis='y', labelcolor='#ff6b6b', labelsize=12);
        ax.tick_params(axis='x', labelsize=12)
        ax.set_title('Cross-Validation Performance', fontsize=16);
        lines, labels = ax.get_legend_handles_labels();
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='center right', fontsize=12);
        ax.grid(True, linestyle='--', alpha=0.6);
        self.curve_fig.tight_layout();
        self.curve_canvas.draw()

    def import_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "Excel Files (*.xlsx)")
        if file_name:
            try:
                self.data = pd.read_excel(file_name, header=0)
                self.X = self.data.iloc[:, :-1].values
                self.y = self.data.iloc[:, -1].values
                self.wavelengths = self.data.columns[:-1].astype(float)
                self.train_button.setEnabled(True)
                info_html = f"""
                <html><head><style>
                    .data-preview {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-family: 'Segoe UI'; font-size: 16px; }}
                    .data-preview th, .data-preview td {{ padding: 8px 12px; border: 1px solid #ddd; }}
                    .data-preview th {{ background-color: #4ecdc4; color: white; font-weight: bold; font-size: 16px; }}
                    .success-header {{ color: #28a745; font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                    .stats-list {{ margin: 10px 0; color: #495057; font-size: 16px; }}
                </style></head><body>
                    <div class="success-header">[✓] Data Import Successful</div>
                    <div class="stats-list">
                        • <b>File Path:</b> {file_name}<br/>
                        • <b>Number of Samples:</b> {self.data.shape[0]}<br/>
                        • <b>Number of Spectral Features:</b> {self.X.shape[1]}<br/>
                        • <b>Wavelength Range:</b> {self.wavelengths[0]:.1f} - {self.wavelengths[-1]:.1f} nm
                    </div>
                    <div style="font-weight: bold; margin: 12px 0 8px 0; color: #2c3e50; font-size: 16px;">Data Preview (First 5 Rows):</div>
                    {self.data.head().to_html(index=False, classes='data-preview')}
                </body></html>"""
                self.result_text.setHtml(info_html)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to import data: {str(e)}")

    def start_training(self):
        if self.X is None or self.y is None:
            QMessageBox.warning(self, "Warning", "Please import data first.")
            return

        self.result_text.setHtml("")
        self.result_text.append("\n" + "=" * 50 + "\n--- Starting Training Process ---")
        QApplication.processEvents()

        try:
            self.result_text.append("\n[1] Splitting Data (80% Train / 20% Test)...")
            QApplication.processEvents()
            self.X_train_raw, self.X_test_raw, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=0.20, random_state=42
            )
            self.result_text.append(
                f"  - Train set: {self.X_train_raw.shape[0]} samples | Test set: {self.X_test_raw.shape[0]} samples")

            self.result_text.append("\n[2] Preprocessing:")
            self.preprocessor = SpectralPreprocessor()

            self.result_text.append("  - Fitting preprocessor on training data...")
            QApplication.processEvents()
            self.X_train = self.preprocessor.fit_transform(self.X_train_raw)

            self.result_text.append("  - Transforming test data with fitted preprocessor...")
            QApplication.processEvents()
            self.X_test = self.preprocessor.transform(self.X_test_raw)
            self.result_text.append("--- Preprocessing Complete ---")

            user_cv_folds = self.cv_folds_spinbox.value()
            if self.X_train.shape[0] < user_cv_folds:
                raise ValueError(
                    f"Not enough samples in training set ({self.X_train.shape[0]}) for {user_cv_folds}-fold CV.")

            user_max_lv = self.max_lv_spinbox.value()
            best_n_components = self.pls_kfold_optimization(
                self.X_train, self.y_train, max_lv=user_max_lv, n_splits=user_cv_folds
            )

            self.result_text.append(f"\n[4] Training Final Model with {best_n_components} LVs...")
            QApplication.processEvents()
            self.pls_best = PLSRegression(n_components=best_n_components)
            self.pls_best.fit(self.X_train, self.y_train)
            self.result_text.append("--- Final Model Trained ---")

            self.result_text.append("\n[5] Evaluating Model Performance...")
            QApplication.processEvents()
            y_train_pred = self.pls_best.predict(self.X_train).flatten()
            y_test_pred = self.pls_best.predict(self.X_test).flatten()

            r2_train = r2_score(self.y_train, y_train_pred)
            rmse_train = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            mae_train = mean_absolute_error(self.y_train, y_train_pred)
            r2_test = r2_score(self.y_test, y_test_pred)
            rmse_test = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            mae_test = mean_absolute_error(self.y_test, y_test_pred)

            result_str = f"""
            <hr><b>--- Final Evaluation Results ---</b><br>
            <b>Best Latent Variables:</b> {best_n_components}<br><br>
            <b>--- Training Set ---</b><br>
            <b>R²:</b> {r2_train:.4f}<br>
            <b>RMSE:</b> {rmse_train:.4f}<br>
            <b>MAE:</b> {mae_train:.4f}<br><br>
            <b>--- Test Set ---</b><br>
            <b>R²:</b> {r2_test:.4f}<br>
            <b>RMSE:</b> {rmse_test:.4f}<br>
            <b>MAE:</b> {mae_test:.4f}<hr>
            """
            self.result_text.append(result_str)
            self.save_button.setEnabled(True)

            self.update_visualizations(y_train_pred, y_test_pred, r2_train, rmse_train, mae_train, r2_test, rmse_test,
                                       mae_test)
            self.result_text.append("\n--- All Plots and Metrics Updated ---")

        except Exception as e:
            import traceback
            error_msg = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
            self.result_text.append(f"\n<b style='color:red;'>--- ERROR ---</b>\n{error_msg}")
            QMessageBox.critical(self, "Training Error", error_msg)
            self.save_button.setEnabled(False)
            self.pls_best = None

    def pls_kfold_optimization(self, X, y, max_lv, n_splits):
        actual_n_splits = min(n_splits, X.shape[0])
        if actual_n_splits < 2:
            raise ValueError(f"Not enough samples ({X.shape[0]}) for CV.")

        kf = KFold(n_splits=actual_n_splits, shuffle=True, random_state=42)
        best_n_components, best_rmse = -1, np.inf

        max_components = min(max_lv, X.shape[1], X.shape[0] - int(X.shape[0] / actual_n_splits))
        if max_components < 1:
            raise ValueError(f"Cannot perform CV, max components ({max_components}) is less than 1.")

        self.result_text.append(
            f"\n[3] Optimizing Latent Variables (Max LV={max_components}, CV={actual_n_splits}-Fold):")
        QApplication.processEvents()

        for n in range(1, max_components + 1):
            pls = PLSRegression(n_components=n)
            cv_neg_mse = cross_val_score(pls, X, y, cv=kf, scoring='neg_mean_squared_error')
            avg_rmse = np.sqrt(-np.mean(cv_neg_mse)) if not np.any(np.isnan(cv_neg_mse)) else np.inf
            if n % 4 == 0 or n == 1 or n == max_components:
                self.result_text.append(f"  - LV={n}: Avg. CV RMSE = {avg_rmse:.4f}")
                QApplication.processEvents()
            if avg_rmse < best_rmse:
                best_rmse, best_n_components = avg_rmse, n

        if best_n_components == -1:
            raise ValueError("PLS optimization failed. Could not find a valid number of components.")

        self.result_text.append(
            f"--- Optimization Complete ---\n==> Selected Best LV = {best_n_components} (Min CV RMSE = {best_rmse:.4f})")
        QApplication.processEvents()
        return best_n_components



    def save_model(self):

        if self.pls_best is None or self.preprocessor is None:
            QMessageBox.warning(self, "Warning", "Please train the model first.")
            return

        file_name, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "Joblib Files (*.joblib)")
        if file_name:
            try:

                model_to_save = {
                    'model': self.pls_best,
                    'preprocessor': self.preprocessor
                }
                joblib.dump(model_to_save, file_name)
                QMessageBox.information(self, "Success", f"Model and preprocessor saved successfully to {file_name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model: {str(e)}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PLSAppleApp()
    ex.show()
    sys.exit(app.exec_())