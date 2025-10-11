import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import textwrap
import joblib
import openpyxl


try:
    import shap
    import matplotlib.pyplot as plt
except ImportError:
    shap = None
    plt = None

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QMessageBox, QGroupBox, QTextBrowser, QSpinBox,
    QDoubleSpinBox, QDialog, QLineEdit
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class EnhancedMLP(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes):
        super(EnhancedMLP, self).__init__()

        layers = []
        in_features = input_dim
        dropout_rates = [0.5, 0.4, 0.3]


        for i, out_features in enumerate(hidden_layer_sizes):
            layers.append(nn.Linear(in_features, out_features))


            if i == len(hidden_layer_sizes) - 1:
                layers.append(nn.LayerNorm(out_features))
            else:
                layers.append(nn.BatchNorm1d(out_features))

            layers.append(nn.LeakyReLU(0.1))


            if i < len(dropout_rates):
                layers.append(nn.Dropout(dropout_rates[i]))

            in_features = out_features


        layers.append(nn.Linear(in_features, 1))

        self.layers = nn.Sequential(*layers)


        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        return self.layers(x)


def predict_with_uncertainty(model, data_tensor, n_samples=50):
    model.train()
    predictions = np.zeros((n_samples, len(data_tensor)))
    with torch.no_grad():
        for i in range(n_samples):
            preds = model(data_tensor).squeeze().cpu().numpy()
            predictions[i, :] = preds
    mean_predictions = np.mean(predictions, axis=0)
    lower_bound = np.percentile(predictions, 2.5, axis=0)
    upper_bound = np.percentile(predictions, 97.5, axis=0)
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


class MlpApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectral DM Value Analysis System - MLP")
        self.setGeometry(100, 100, 1350, 1380)
        self.model = None
        self.scaler = None
        self.current_data = None
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.train_losses = []
        self.feature_names = None
        self.initUI()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        self.colors = {"sidebar_bg_start": "#2c3e50", "sidebar_bg_end": "#34495e", "sidebar_border": "#3a506b",
                       "sidebar_text": "#ecf0f1", "sidebar_title": "#7fcdff", "group_border_dark": "#4a5568",
                       "widget_bg_dark": "#434C5E", "accent_primary": "#4ecdc4", "accent_positive": "#4CAF50",
                       "accent_negative": "#f44336", "accent_shap": "#ff9a8b", "panel_bg": "#f8f9fa",
                       "panel_text": "#343a40", "group_border_light": "#dee2e6", "plot_scatter": "#007bff",
                       "plot_line": "#dc3545", "plot_train_loss": "#17a2b8", "plot_val_loss": "#ffc107",
                       "plot_residuals": "#fd7e14", "plot_hist": "#28a745", "log_bg_light": "white",
                       "log_border_light": "#e9ecef", "log_text_light": "#495057"}
        sidebar = QWidget();
        sidebar.setFixedWidth(320);
        sidebar.setStyleSheet(
            f""" QWidget {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {self.colors['sidebar_bg_start']}, stop:1 {self.colors['sidebar_bg_end']}); border-right: 1px solid {self.colors['sidebar_border']}; color: {self.colors['sidebar_text']}; }} """)
        sidebar_layout = QVBoxLayout(sidebar);
        sidebar_layout.setContentsMargins(20, 20, 20, 20);
        sidebar_layout.setSpacing(20)
        title = QLabel("Control Panel");
        title.setFont(QFont('Segoe UI', 16, QFont.Bold));
        title.setStyleSheet(f"color: {self.colors['sidebar_title']}; border: none; padding-bottom: 5px;");
        title.setAlignment(Qt.AlignCenter);
        sidebar_layout.addWidget(title)
        control_group = QGroupBox("Data & Model Control");
        control_group.setStyleSheet(self.groupbox_style_dark());
        control_layout = QVBoxLayout();
        control_layout.setSpacing(15);
        control_layout.setContentsMargins(15, 20, 15, 15)
        self.btn_load = QPushButton("📂 Load Spectral Data");
        self.btn_load.setStyleSheet(self.button_style(accent=self.colors['accent_primary']));
        self.btn_load.clicked.connect(self.load_data);
        control_layout.addWidget(self.btn_load)
        self.btn_train = QPushButton("🚀 Train Model");
        self.btn_train.setStyleSheet(self.button_style(accent=self.colors['accent_positive']));
        self.btn_train.clicked.connect(self.train_model);
        self.btn_train.setEnabled(False);
        control_layout.addWidget(self.btn_train)
        self.btn_shap = QPushButton("🧠 Calculate SHAP Values");
        self.btn_shap.setStyleSheet(self.button_style(accent=self.colors['accent_shap']));
        self.btn_shap.clicked.connect(self.show_shap_popup);
        self.btn_shap.setEnabled(False);
        control_layout.addWidget(self.btn_shap)

        # Parameter Layout
        param_layout = QVBoxLayout();
        param_layout.setSpacing(10);

        # Epochs control
        epoch_layout = QHBoxLayout();
        lbl_epochs = QLabel("Epochs:");
        lbl_epochs.setFont(QFont('Segoe UI', 10));
        lbl_epochs.setStyleSheet("border: none; background: transparent;");
        self.spin_epochs = QSpinBox();
        self.spin_epochs.setRange(10, 5000);
        self.spin_epochs.setValue(200);
        self.spin_epochs.setSingleStep(10);
        self.spin_epochs.setAlignment(Qt.AlignCenter);
        self.spin_epochs.setStyleSheet(self.spinbox_style());
        epoch_layout.addWidget(lbl_epochs);
        epoch_layout.addWidget(self.spin_epochs);
        param_layout.addLayout(epoch_layout);

        # Learning Rate control
        lr_layout = QHBoxLayout();
        lbl_lr = QLabel("Learning Rate:");
        lbl_lr.setFont(QFont('Segoe UI', 10));
        lbl_lr.setStyleSheet("border: none; background: transparent;");
        self.spin_lr = QDoubleSpinBox();
        self.spin_lr.setDecimals(5)
        self.spin_lr.setSingleStep(0.0001)
        self.spin_lr.setValue(0.001)
        self.spin_lr.setRange(0.00001, 0.1)
        self.spin_lr.setStyleSheet(self.spinbox_style());
        lr_layout.addWidget(lbl_lr);
        lr_layout.addWidget(self.spin_lr);
        param_layout.addLayout(lr_layout);


        layer_layout = QVBoxLayout()
        lbl_layers = QLabel("Dense Layer Sizes:");
        lbl_layers.setFont(QFont('Segoe UI', 10));
        lbl_layers.setStyleSheet("border: none; background: transparent; margin-top: 5px;");
        self.line_layer_sizes = QLineEdit("256, 128, 64, 32")
        self.line_layer_sizes.setToolTip("Enter comma-separated integers for hidden layer sizes.")
        self.line_layer_sizes.setStyleSheet(f"""
            QLineEdit {{
                background-color: {self.colors['widget_bg_dark']}; color: {self.colors['sidebar_text']};
                border: 1px solid {self.colors['group_border_dark']}; border-radius: 4px;
                padding: 5px; font-family: 'Segoe UI'; font-size: 13px;
            }}
            QLineEdit:focus {{ border: 1px solid {self.colors['accent_primary']}; }}
        """)
        layer_layout.addWidget(lbl_layers)
        layer_layout.addWidget(self.line_layer_sizes)
        param_layout.addLayout(layer_layout)

        control_layout.addLayout(param_layout)

        # Data Info Layout
        data_info_layout = QVBoxLayout();
        data_info_layout.setSpacing(5);
        features_layout = QHBoxLayout();
        lbl_feat_title = QLabel("Features:");
        lbl_feat_title.setFont(QFont('Segoe UI', 10));
        lbl_feat_title.setStyleSheet("border: none; background: transparent;");
        self.lbl_features_value = QLabel("N/A");
        self.lbl_features_value.setFont(QFont('Segoe UI', 10, QFont.Bold));
        self.lbl_features_value.setStyleSheet("border: none; background: transparent;");
        self.lbl_features_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter);
        features_layout.addWidget(lbl_feat_title);
        features_layout.addWidget(self.lbl_features_value);
        data_info_layout.addLayout(features_layout);
        samples_layout = QHBoxLayout();
        lbl_samp_title = QLabel("Samples:");
        lbl_samp_title.setFont(QFont('Segoe UI', 10));
        lbl_samp_title.setStyleSheet("border: none; background: transparent;");
        self.lbl_samples_value = QLabel("N/A");
        self.lbl_samples_value.setFont(QFont('Segoe UI', 10, QFont.Bold));
        self.lbl_samples_value.setStyleSheet("border: none; background: transparent;");
        self.lbl_samples_value.setAlignment(Qt.AlignRight | Qt.AlignVCenter);
        samples_layout.addWidget(lbl_samp_title);
        samples_layout.addWidget(self.lbl_samples_value);
        data_info_layout.addLayout(samples_layout);
        control_layout.addLayout(data_info_layout);
        control_layout.addStretch(1);
        control_group.setLayout(control_layout);
        sidebar_layout.addWidget(control_group)
        sidebar_log_group = QGroupBox("Activity Log");
        sidebar_log_group.setStyleSheet(self.groupbox_style_dark());
        sidebar_log_layout = QVBoxLayout();
        sidebar_log_layout.setContentsMargins(15, 20, 15, 15);
        self.sidebar_log_display = QTextBrowser();
        self.sidebar_log_display.setReadOnly(True);
        self.sidebar_log_display.setPlaceholderText("Brief activity updates...");
        self.sidebar_log_display.setStyleSheet(
            f""" QTextBrowser {{ background: {self.colors['sidebar_bg_start']}; border: 1px solid {self.colors['group_border_dark']}; border-radius: 4px; padding: 10px; font-family: 'Consolas', 'Courier New', monospace; font-size: 12px; color: {self.colors['sidebar_text']}; line-height: 1.4; }} """);
        sidebar_log_layout.addWidget(self.sidebar_log_display);
        sidebar_log_group.setLayout(sidebar_log_layout);
        sidebar_layout.addWidget(sidebar_log_group, stretch=1);
        main_layout.addWidget(sidebar)
        right_panel = QWidget();
        right_panel.setStyleSheet(f"background: {self.colors['panel_bg']};");
        right_layout = QVBoxLayout(right_panel);
        right_layout.setContentsMargins(20, 20, 20, 20);
        right_layout.setSpacing(15)
        vis_group = QGroupBox("Analysis Results");
        vis_group.setStyleSheet(self.groupbox_style_light());
        vis_layout = QVBoxLayout();
        vis_layout.setContentsMargins(10, 15, 10, 10);
        self.figure = Figure(figsize=(8, 12), dpi=100);  # Adjusted size for 4 plots
        self.figure.patch.set_facecolor(self.colors['panel_bg']);
        self.canvas = FigureCanvas(self.figure);
        self.canvas.setStyleSheet("border: none;");
        self._setup_initial_plot();
        vis_layout.addWidget(self.canvas);
        vis_group.setLayout(vis_layout);
        right_layout.addWidget(vis_group, stretch=2)
        main_log_group = QGroupBox("System Log");
        main_log_group.setStyleSheet(self.groupbox_style_light());
        main_log_layout = QVBoxLayout();
        main_log_layout.setContentsMargins(10, 15, 10, 10);
        self.log = QTextBrowser();
        self.log.setReadOnly(True);
        self.log.setPlaceholderText("Detailed logs and training progress will appear here...");
        self.log.setStyleSheet(
            f""" QTextBrowser {{ background: {self.colors['log_bg_light']}; border: 1px solid {self.colors['log_border_light']}; border-radius: 4px; padding: 10px; font-family: 'Consolas', 'Courier New', monospace; font-size: 12px; color: {self.colors['log_text_light']}; line-height: 1.4; }} """);
        main_log_layout.addWidget(self.log);
        main_log_group.setLayout(main_log_layout);
        right_layout.addWidget(main_log_group, stretch=1);
        main_layout.addWidget(right_panel, stretch=1)

    def train_model(self):
        if self.current_data is None:
            QMessageBox.warning(self, "Warning", "Please load data first!")
            return


        try:
            layer_sizes_str = self.line_layer_sizes.text()
            if not layer_sizes_str.strip():
                raise ValueError("Layer sizes cannot be empty.")
            hidden_layer_sizes = [int(size.strip()) for size in layer_sizes_str.split(',') if size.strip()]
            if not hidden_layer_sizes:
                raise ValueError("Parsed layer sizes list is empty.")
        except ValueError as e:
            QMessageBox.critical(self, "Invalid Architecture",
                                 f"Could not parse layer sizes: {e}\nPlease provide comma-separated integers (e.g., 256, 128, 64).")
            return


        n_epochs = self.spin_epochs.value()
        learning_rate = self.spin_lr.value()

        self._append_to_main_log(f"\n🚀 Starting model training...")
        self._append_to_main_log(f"   - Architecture: {' -> '.join(map(str, hidden_layer_sizes))}")
        self._append_to_main_log(f"   - Hyperparameters: {n_epochs} epochs, LR={learning_rate}")

        self._append_to_sidebar_log("Training...")
        self.btn_train.setEnabled(False)
        self.btn_shap.setEnabled(False)
        QApplication.processEvents()
        self.train_losses = []
        try:
            self.scaler = RobustScaler()
            X_train_scaled = self.scaler.fit_transform(self.X_train.copy())
            X_test_scaled = self.scaler.transform(self.X_test.copy())


            self.model = EnhancedMLP(X_train_scaled.shape[1], hidden_layer_sizes)

            criterion = nn.HuberLoss()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.001)
            train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train_scaled),
                                                           torch.FloatTensor(self.y_train.reshape(-1, 1)))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
            self.model.train()
            for epoch in range(n_epochs):
                epoch_loss = 0
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = self.model(inputs.float()).squeeze()
                    loss = criterion(outputs, targets.float().squeeze())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    epoch_loss += loss.item()
                avg_epoch_loss = epoch_loss / len(train_loader)
                self.train_losses.append(avg_epoch_loss)
                if (epoch + 1) % 10 == 0:
                    self._append_to_main_log(f"      Epoch [{epoch + 1}/{n_epochs}], Loss: {avg_epoch_loss:.4f}")
                    QApplication.processEvents()
            self._append_to_sidebar_log("Training Done.")
            self._append_to_main_log("✅ Model training complete!")
            self._append_to_main_log("\n📊 Evaluating model performance...")
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            y_pred_test, pi_lower_test, pi_upper_test = predict_with_uncertainty(self.model, X_test_tensor)
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            y_pred_train, _, _ = predict_with_uncertainty(self.model, X_train_tensor)
            self.model.eval()
            y_test_np, y_train_np = self.y_test, self.y_train
            r2_test = r2_score(y_test_np, y_pred_test);
            rmse_test = np.sqrt(mean_squared_error(y_test_np, y_pred_test));
            mae_test = mean_absolute_error(y_test_np, y_pred_test)
            r2_train = r2_score(y_train_np, y_pred_train);
            rmse_train = np.sqrt(mean_squared_error(y_train_np, y_pred_train));
            mae_train = mean_absolute_error(y_train_np, y_pred_train)
            log_output = textwrap.dedent(f"""
                            ------------------------------
                            📊 Final Model Evaluation Results:
                            --- Training Set ---
                            ➤ R² Score        : {r2_train:.4f}
                            ➤ RMSE            : {rmse_train:.4f}
                            ➤ MAE             : {mae_train:.4f}
                            --- Test Set ---
                            ➤ R² Score        : {r2_test:.4f}
                            ➤ RMSE            : {rmse_test:.4f}
                            ➤ MAE             : {mae_test:.4f}
                            ------------------------------
                            """)
            self._append_to_main_log(log_output)
            self._update_plots(y_test_np, y_pred_test, pi_lower_test, pi_upper_test, self.train_losses)
            self._save_model()
            self.btn_shap.setEnabled(True)
        except Exception as e:
            import traceback
            error_msg = f"An error occurred during model training: {str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "Training Error", error_msg)
            self._append_to_main_log(f"❌ Training failed: {str(e)}")
            self._append_to_sidebar_log("Training Failed.")
        finally:
            self.btn_train.setEnabled(True)


    def show_shap_popup(self):
        if self.model is None or self.scaler is None or self.X_train is None:
            QMessageBox.warning(self, "SHAP Error", "Please train a model first.")
            return

        self._append_to_main_log("\n🧠 Calculating SHAP values... This may take a few minutes.")
        self._append_to_sidebar_log("SHAP running...")
        self.btn_shap.setEnabled(False)
        QApplication.processEvents()

        try:
            def predict_fn(x_unscaled):
                self.model.eval()
                x_scaled = self.scaler.transform(x_unscaled)
                x_tensor = torch.FloatTensor(x_scaled)
                with torch.no_grad():
                    output = self.model(x_tensor)
                return output.cpu().numpy()

            background_data = shap.kmeans(self.X_train, 50).data
            explainer = shap.KernelExplainer(predict_fn, background_data)
            data_to_explain_unscaled = self.X_test[:100]
            shap_values = explainer.shap_values(data_to_explain_unscaled)

            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            self._append_to_main_log("✅ SHAP calculation complete. Displaying plot in new window...")
            data_to_explain_scaled = self.scaler.transform(data_to_explain_unscaled)

            plt.close('all')

            shap.summary_plot(shap_values, data_to_explain_scaled, feature_names=self.feature_names, show=False,
                              plot_type='dot')

            shap_figure = plt.gcf()

            dialog = ShapPlotDialog(shap_figure, self)
            dialog.exec_()

        except Exception as e:
            import traceback
            error_msg = f"SHAP calculation failed: {str(e)}\n\n{traceback.format_exc()}"
            QMessageBox.critical(self, "SHAP Error", error_msg)
            self._append_to_main_log(f"❌ {error_msg}")
            self._append_to_sidebar_log("SHAP Failed.")
        finally:
            self.btn_shap.setEnabled(True)

    def _update_plots(self, y_true, y_pred, pi_lower, pi_upper, train_loss_history):
        if y_true is None or y_pred is None: self._setup_initial_plot(); return
        try:
            self.figure.clear();
            self.figure.patch.set_facecolor(self.colors['panel_bg'])
            ax1 = self.figure.add_subplot(2, 2, 1);
            ax1.set_facecolor(self.colors['panel_bg']);
            ax1.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', s=25, c=self.colors['plot_scatter'],
                        label='Test Predictions (Mean)');
            lim_min = min(np.nanmin(y_true) if y_true.size > 0 else 0,
                          np.nanmin(y_pred) if y_pred.size > 0 else 0) * 0.95;
            lim_max = max(np.nanmax(y_true) if y_true.size > 0 else 0,
                          np.nanmax(y_pred) if y_pred.size > 0 else 0) * 1.05;
            lims = [lim_min, lim_max];
            ax1.plot(lims, lims, color='k', linestyle='-', alpha=0.75, zorder=0, label='Ideal (y=x)');
            sort_indices = np.argsort(y_pred);
            sorted_preds = y_pred[sort_indices];
            sorted_lower = pi_lower[sort_indices];
            sorted_upper = pi_upper[sort_indices];
            ax1.fill_between(sorted_preds, sorted_lower, sorted_upper, color=self.colors['plot_scatter'], alpha=0.2,
                             label='95% PI');
            ax1.set_xlabel("Actual Values", fontsize=10);
            ax1.set_ylabel("Predicted Values", fontsize=10);
            ax1.set_title("Prediction vs Actual", fontsize=12);
            ax1.grid(True, linestyle=':', alpha=0.5);
            ax1.tick_params(axis='both', which='major', labelsize=8);
            ax1.legend(fontsize=8);
            ax1.set_xlim(lims);
            ax1.set_ylim(lims)
            ax2 = self.figure.add_subplot(2, 2, 2);
            ax2.set_facecolor(self.colors['panel_bg'])
            if train_loss_history: epochs = range(1, len(train_loss_history) + 1); ax2.plot(epochs, train_loss_history,
                                                                                            color=self.colors[
                                                                                                'plot_train_loss'],
                                                                                            marker='.', linestyle='-',
                                                                                            markersize=4,
                                                                                            label='Train Loss'); ax2.set_xlabel(
                "Epoch", fontsize=10); ax2.set_ylabel("Loss (Huber)", fontsize=10); ax2.set_title("Training Loss Curve",
                                                                                                  fontsize=12); ax2.grid(
                True, linestyle=':', alpha=0.5); ax2.tick_params(axis='both', which='major', labelsize=8); ax2.legend(
                fontsize=9)
            ax3 = self.figure.add_subplot(2, 2, 3);
            ax3.set_facecolor(self.colors['panel_bg']);
            residuals = y_true - y_pred;
            ax3.scatter(y_pred, residuals, alpha=0.6, edgecolors='w', s=25, c=self.colors['plot_residuals']);
            ax3.axhline(y=0, color=self.colors['plot_line'], linestyle='--', alpha=0.8);
            ax3.set_xlabel("Predicted Value", fontsize=10);
            ax3.set_ylabel("Residual", fontsize=10);
            ax3.set_title("Residual Plot", fontsize=12);
            ax3.grid(True, linestyle=':', alpha=0.5);
            ax3.tick_params(axis='both', which='major', labelsize=8)
            ax4 = self.figure.add_subplot(2, 2, 4);
            ax4.set_facecolor(self.colors['panel_bg']);
            ax4.hist(residuals, bins=30, alpha=0.75, color=self.colors['plot_hist'], edgecolor='black');
            ax4.set_xlabel("Residual Value", fontsize=10);
            ax4.set_ylabel("Frequency", fontsize=10);
            ax4.set_title("Residual Distribution", fontsize=12);
            ax4.grid(True, linestyle=':', alpha=0.5, axis='y');
            ax4.tick_params(axis='both', which='major', labelsize=8)
            self.figure.tight_layout(pad=2.0);
            self.canvas.draw()
        except Exception as plot_err:
            import traceback;
            error_msg = f"❌ Plotting Error: {plot_err}\n{traceback.format_exc()}";
            self._append_to_main_log(error_msg)
            self.figure.clear();
            ax = self.figure.add_subplot(111);
            ax.text(0.5, 0.5, f"Error during plotting:\n{plot_err}", ha='center', va='center',
                    color=self.colors['accent_negative'], wrap=True);
            self.canvas.draw()

    def _setup_initial_plot(self):
        self.figure.clear()
        for i in range(1, 5):
            ax = self.figure.add_subplot(2, 2, i);
            ax.set_facecolor(self.colors['panel_bg']);
            ax.text(0.5, 0.5, f"Plot Area {i}\n(Requires Training Data)", horizontalalignment='center',
                    verticalalignment='center', fontsize=10, color='grey', transform=ax.transAxes, wrap=True);
            ax.set_xticks([]);
            ax.set_yticks([])
        try:
            self.figure.tight_layout(pad=2.0)
        except Exception as e:
            print(f"Initial plot layout warning: {e}")
        self.canvas.draw()

    def spinbox_style(self):
        return f""" QSpinBox, QDoubleSpinBox {{ background-color: {self.colors['widget_bg_dark']}; color: {self.colors['sidebar_text']}; border: 1px solid {self.colors['group_border_dark']}; border-radius: 4px; padding: 5px; font-family: 'Segoe UI'; font-size: 13px; }} QSpinBox:focus, QDoubleSpinBox:focus {{ border: 1px solid {self.colors['accent_primary']}; }} QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right; width: 16px; border-left: 1px solid {self.colors['group_border_dark']}; }} QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right; width: 16px; border-left: 1px solid {self.colors['group_border_dark']}; }} """

    def button_style(self, accent):
        rgba_bg = f"rgba({int(accent[1:3], 16)}, {int(accent[3:5], 16)}, {int(accent[5:7], 16)}, 0.2)";
        rgba_hover = f"rgba({int(accent[1:3], 16)}, {int(accent[3:5], 16)}, {int(accent[5:7], 16)}, 0.3)";
        rgba_pressed = f"rgba({int(accent[1:3], 16)}, {int(accent[3:5], 16)}, {int(accent[5:7], 16)}, 0.4)";
        rgba_border = f"rgba({int(accent[1:3], 16)}, {int(accent[3:5], 16)}, {int(accent[5:7], 16)}, 0.4)";
        rgba_border_hover = f"rgba({int(accent[1:3], 16)}, {int(accent[3:5], 16)}, {int(accent[5:7], 16)}, 0.6)";
        return f""" QPushButton {{ background: {rgba_bg}; color: {accent}; border: 1px solid {rgba_border}; border-radius: 5px; padding: 10px; font: bold 13px 'Segoe UI'; min-height: 30px; text-align: left; padding-left: 15px; }} QPushButton:hover {{ background: {rgba_hover}; border: 1px solid {rgba_border_hover}; }} QPushButton:pressed {{ background: {rgba_pressed}; }} QPushButton:disabled {{ background-color: #50687c; color: #8ea4b8; border-color: #5f7a93; }} """

    def groupbox_style_dark(self):
        return f""" QGroupBox {{ color: {self.colors['sidebar_text']}; font: bold 14px 'Segoe UI'; border: 1px solid {self.colors['group_border_dark']}; border-radius: 5px; margin-top: 10px; padding-top: 15px; }} QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top center; padding: 0 5px 5px 5px; color: {self.colors['sidebar_title']}; }} """

    def groupbox_style_light(self):
        return f""" QGroupBox {{ font: bold 14px 'Segoe UI'; color: {self.colors['panel_text']}; border: 1px solid {self.colors['group_border_light']}; border-radius: 5px; margin-top: 10px; padding-top: 15px; }} QGroupBox::title {{ subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px 5px 5px; margin-left: 10px; color: {self.colors['panel_text']}; }} """

    def _append_to_main_log(self, message):
        if self.log: self.log.append(str(message)); self.log.ensureCursorVisible(); QApplication.processEvents()

    def _append_to_sidebar_log(self, message):
        if self.sidebar_log_display: self.sidebar_log_display.append(
            str(message)); self.sidebar_log_display.ensureCursorVisible(); QApplication.processEvents()

    def load_data(self):
        try:
            file_path, _ = QFileDialog.getOpenFileName(self, "Open Data File", "",
                                                       "Data Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls)")
            if not file_path: self._append_to_main_log("ℹ️ File selection cancelled."); return
            self._append_to_main_log(f"🔄 Loading data from: {file_path}...");
            self._append_to_sidebar_log(f"Loading: {file_path.split('/')[-1]}");
            QApplication.processEvents()
            df = pd.read_csv(file_path, header=0) if file_path.lower().endswith(
                ('.csv', '.CSV')) else pd.read_excel(file_path, header=0, engine='openpyxl')
            validation_msg = self._validate_data(df)
            if validation_msg != "ok": QMessageBox.warning(self, "Data Validation Failed",
                                                           validation_msg); self._append_to_main_log(
                f"❌ Data validation failed: {validation_msg}"); return
            y = df.iloc[:, -1].values;
            X = df.iloc[:, :-1].values;
            label_name = df.columns[-1];
            self.feature_names = df.columns[:-1].tolist()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2,
                                                                                    random_state=42)
            self.current_data = df;
            self._update_data_info(X.shape[1], X.shape[0])
            log_message_main = f"✅ Data loaded successfully!\n   - Features: {X.shape[1]}, Samples: {X.shape[0]}\n   - Label Column: '{label_name}'\n   - Split: Train {len(self.X_train)}, Test {len(self.X_test)}"
            self._append_to_main_log(log_message_main);
            self._append_to_sidebar_log(f"Loaded: {X.shape[0]} samples.")
            self.btn_train.setEnabled(True);
            if hasattr(self, 'btn_shap'): self.btn_shap.setEnabled(False)
            self.model = None;
            self.scaler = None;
            self.train_losses = [];
            self._setup_initial_plot()
        except ImportError as e:
            QMessageBox.warning(self, "Dependency Missing", str(e));
            self._append_to_main_log(f"❌ Error: {str(e)}")
        except Exception as e:
            import traceback;
            error_msg = f"Failed to load data: {str(e)}\n\n{traceback.format_exc()}";
            QMessageBox.critical(self, "Load Error", error_msg);
            self._append_to_main_log(f"❌ Load failed: {str(e)}");
            self._reset_ui_state()

    def _validate_data(self, df):
        try:
            if df is None or df.empty: return "Data is empty or failed to load."
            if df.shape[1] < 2: return "Data must contain at least two columns (features + 1 label column)."
            if not pd.api.types.is_numeric_dtype(
                    df.iloc[:, -1]): return f"The last column (label column '{df.columns[-1]}') must be numeric."
            if df.iloc[:, -1].isnull().any(): return f"Label column '{df.columns[-1]}' contains missing values."
            df_features_numeric = df.iloc[:, :-1].apply(lambda x: pd.to_numeric(x, errors='coerce'))
            if df_features_numeric.isnull().any().any():
                bad_cols = df.columns[:-1][df_features_numeric.isnull().any()].tolist()
                return f"Non-numeric feature columns detected after coercion: {bad_cols}. Please ensure all feature data is numeric."
            return "ok"
        except Exception as e:
            return f"Data validation error: {str(e)}"

    def _update_data_info(self, num_features, num_samples):
        self.lbl_features_value.setText(str(num_features));
        self.lbl_samples_value.setText(str(num_samples))

    def _reset_ui_state(self):
        self.btn_train.setEnabled(False)
        if hasattr(self, 'btn_shap'): self.btn_shap.setEnabled(False)
        self._update_data_info("N/A", "N/A");
        self._setup_initial_plot();
        self._append_to_sidebar_log("UI Reset.")

    def _save_model(self):
        try:
            save_path, _ = QFileDialog.getSaveFileName(self, "Save Model File", "", "Model File (*.pth);;All Files (*)")
            if save_path:
                if self.model is None or self.scaler is None:
                    QMessageBox.warning(self, "Warning", "Model or Scaler not available for saving.")
                    return


                hidden_layer_sizes = [int(size.strip()) for size in self.line_layer_sizes.text().split(',') if
                                      size.strip()]


                model_data_to_save = {
                    'model_state_dict': self.model.state_dict(),
                    'hidden_layer_sizes': hidden_layer_sizes,
                    'input_dim': self.X_train.shape[1]
                }


                torch.save(model_data_to_save, save_path)


                scaler_path = save_path.replace(".pth", "_scaler.pkl")
                joblib.dump(self.scaler, scaler_path)

                self._append_to_main_log(f"💾 Model and architecture saved to: {save_path}")
                self._append_to_main_log(f"💾 Scaler saved to: {scaler_path}")
                self._append_to_sidebar_log(f"Model Saved: {save_path.split('/')[-1]}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed: {str(e)}")
            self._append_to_main_log(f"❌ Save failed: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MlpApp()
    window.show()
    sys.exit(app.exec_())
