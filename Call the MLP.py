import sys
import pandas as pd
import joblib
import torch
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog, QTableWidget,
                             QTableWidgetItem, QLabel, QMessageBox, QGroupBox)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class EnhancedMLP(torch.nn.Module):


    def __init__(self, input_dim, hidden_layer_sizes):
        super(EnhancedMLP, self).__init__()

        layers = []
        in_features = input_dim
        dropout_rates = [0.5, 0.4, 0.3]

        for i, out_features in enumerate(hidden_layer_sizes):
            layers.append(torch.nn.Linear(in_features, out_features))

            if i == len(hidden_layer_sizes) - 1:
                layers.append(torch.nn.LayerNorm(out_features))
            else:
                layers.append(torch.nn.BatchNorm1d(out_features))

            layers.append(torch.nn.LeakyReLU(0.1))

            if i < len(dropout_rates):
                layers.append(torch.nn.Dropout(dropout_rates[i]))

            in_features = out_features

        layers.append(torch.nn.Linear(in_features, 1))

        self.layers = torch.nn.Sequential(*layers)

        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        return self.layers(x)




class MLPPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.scaler = None
        self.current_data = None
        self.initUI()

    def _get_button_style(self, color_hex):
        text_color = "white"
        font_style = "bold 26px 'Segoe UI'"
        border_radius = "18px"
        padding = "18px 8px"

        def darken_hex(hex_str, factor):
            if hex_str.startswith('#'): hex_str = hex_str[1:]
            if len(hex_str) == 6:
                r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
                r, g, b = max(0, int(r * factor)), max(0, int(g * factor)), max(0, int(b * factor))
                return f"#{r:02x}{g:02x}{b:02x}"
            return hex_str

        hover_bg = darken_hex(color_hex, 0.85)
        pressed_bg = darken_hex(color_hex, 0.7)
        return f"""
            QPushButton {{ background-color:{color_hex}; color:{text_color}; border:none; border-radius:{border_radius}; padding:{padding}; font:{font_style}; }}
            QPushButton:hover {{ background-color:{hover_bg}; }}
            QPushButton:pressed {{ background-color:{pressed_bg}; }}
            QPushButton:disabled {{ background-color:#bdc3c7; color:#7f8c8d; }}"""

    def initUI(self):
        self.setWindowTitle("MLP Model Prediction System")
        self.setGeometry(100, 100, 1200, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        left_panel = QGroupBox("Control Panel")
        left_panel.setFixedWidth(300)
        left_panel.setStyleSheet("""
               QGroupBox {
                   background-color: #f0f2f5; border: 0px; border-right: 1px solid #d1d5db;
                   padding-top: 55px; padding-left: 0px; padding-right: 0px; padding-bottom: 0px;
               }
               QGroupBox::title {
                   color: #1a202c; font: bold 30px 'Segoe UI'; subcontrol-origin: padding;
                   subcontrol-position: top left; padding: 10px 0px 0px 10px;
               }
           """)
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(10, 0, 10, 20)
        left_layout.setSpacing(15)
        self.btn_load_model = QPushButton("Load Model File (.pth)")
        self.btn_load_model.setStyleSheet(self._get_button_style(color_hex="#4ecdc4"))
        self.btn_load_model.clicked.connect(self.load_model)
        left_layout.addWidget(self.btn_load_model)
        self.btn_load_data = QPushButton("Load Prediction Data")
        self.btn_load_data.setStyleSheet(self._get_button_style(color_hex="#4ecdc4"))
        self.btn_load_data.clicked.connect(self.load_data)
        self.btn_load_data.setEnabled(False)
        left_layout.addWidget(self.btn_load_data)
        self.btn_predict = QPushButton("Perform Prediction")
        self.btn_predict.setStyleSheet(self._get_button_style(color_hex="#45b7d1"))
        self.btn_predict.clicked.connect(self.predict)
        self.btn_predict.setEnabled(False)
        left_layout.addWidget(self.btn_predict)
        self.btn_save = QPushButton("Save Results")
        self.btn_save.setStyleSheet(self._get_button_style(color_hex="#4ecdc4"))
        self.btn_save.clicked.connect(self.save_results)
        self.btn_save.setEnabled(False)
        left_layout.addWidget(self.btn_save)
        left_layout.addSpacing(10)
        self.model_info = QLabel("Loaded Model: None")
        self.model_info.setStyleSheet("color: #4a5568; font: 14px 'Segoe UI'; padding-top: 5px;")
        self.model_info.setWordWrap(True)
        left_layout.addWidget(self.model_info)
        self.data_info = QLabel("Loaded Data: 0 samples")
        self.data_info.setStyleSheet("color: #4a5568; font: 14px 'Segoe UI'; padding-top: 5px;")
        self.data_info.setWordWrap(True)
        left_layout.addWidget(self.data_info)
        left_layout.addStretch()
        left_panel.setLayout(left_layout)
        layout.addWidget(left_panel, 1)

        right_panel = QGroupBox("Prediction Results")
        right_panel.setStyleSheet("""
               QGroupBox {
                   background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px;
                   font: bold 16px 'Segoe UI'; color: #2d3748; margin-top: 10px;
               }
               QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 5px 10px; }
           """)
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(20, 30, 20, 20)
        right_layout.setSpacing(15)
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(["Prediction", "Confidence", "CI Lower", "CI Upper"])
        header = self.result_table.horizontalHeader()
        header.setSectionResizeMode(header.Stretch)
        self.result_table.setStyleSheet("""
               QTableWidget {
                   background-color: white; alternate-background-color: #fdfdfe; 
                   border: 1px solid #d1d5db; gridline-color: #e0e0e0; font: 24px 'Segoe UI'; color: #2d3748;
               }
               QTableWidget::item { padding: 6px; } 
               QHeaderView::section {
                   background-color: #e9ecef; color: #1a202c; padding: 8px; font: bold 24px 'Segoe UI';
                   border-top: 1px solid #d1d5db; border-bottom: 1px solid #d1d5db;
                   border-left: none; border-right: 1px solid #d1d5db;
               }
               QHeaderView::section:first { border-left: 1px solid #d1d5db; } 
               QTableCornerButton::section { background-color: #e9ecef; border: 1px solid #d1d5db; }
           """)
        right_layout.addWidget(self.result_table, 2)
        self.figure = plt.figure()
        self.figure.patch.set_facecolor('#f8f9fa')
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas, 1)
        right_panel.setLayout(right_layout)
        layout.addWidget(right_panel, 3)

    def load_model(self):
        try:
            model_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Model Files (*.pth)")
            if model_path:
                checkpoint = torch.load(model_path)

                input_dim = checkpoint['input_dim']
                hidden_layer_sizes = checkpoint['hidden_layer_sizes']

                self.model = EnhancedMLP(input_dim, hidden_layer_sizes)

                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()

                scaler_path = model_path.replace(".pth", "_scaler.pkl")
                self.scaler = joblib.load(scaler_path)

                model_display_path = model_path.split('/')[-1]
                arch_str = ' -> '.join(map(str, hidden_layer_sizes))
                self.model_info.setText(
                    f"Model: {model_display_path}\n"
                    f"Features: {input_dim}\n"
                    f"Architecture: {arch_str}"
                )
                self.btn_load_data.setEnabled(True)
                QMessageBox.information(self, "Success", "Model and scaler loaded successfully!")

                self.btn_predict.setEnabled(False)
                self.btn_save.setEnabled(False)
                self.result_table.setRowCount(0)
                self.figure.clear()
                self.canvas.draw()

        except FileNotFoundError:
            QMessageBox.critical(self, "Error",
                                 f"Scaler file not found. Ensure it's in the same directory as the model file.")
            self._reset_model_state()
        except KeyError as e:
            QMessageBox.critical(self, "Error",
                                 f"Model file is incompatible. It might be missing key information like '{e}'. Please retrain and save the model with the updated training script.")
            self._reset_model_state()
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error",
                                 f"An error occurred while loading the model: {str(e)}\n\n{traceback.format_exc()}")
            self._reset_model_state()

    def _reset_model_state(self):
        self.model, self.scaler = None, None
        self.model_info.setText("Loaded Model: None")
        self.btn_load_data.setEnabled(False)
        self.btn_predict.setEnabled(False)



    def load_data(self):
        try:
            if not self.model or not self.scaler:
                QMessageBox.warning(self, "Model Not Loaded", "Please load a model first.")
                return
            path, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "Data Files (*.csv *.xlsx)")
            if path:
                df = pd.read_csv(path) if path.lower().endswith('.csv') else pd.read_excel(path, engine='openpyxl')
                expected_features = self.model.layers[0].in_features
                if df.shape[1] != expected_features:
                    raise ValueError(
                        f"Feature count mismatch: model expects {expected_features}, data has {df.shape[1]}.")
                self.current_data = df.values.astype(np.float32)
                self.data_info.setText(f"Loaded Data: {len(df)} samples")
                self.btn_predict.setEnabled(True)
                QMessageBox.information(self, "Success", "Data for prediction loaded successfully!")
                self.btn_save.setEnabled(False)
                self.result_table.setRowCount(0)
                self.figure.clear()
                self.canvas.draw()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Data loading failed: {str(e)}")
            self.current_data = None
            self.data_info.setText("Loaded Data: 0 samples")
            self.btn_predict.setEnabled(False)

    def predict(self):
        try:
            if self.current_data is None:
                QMessageBox.warning(self, "No Data", "Please load data first.")
                return

            scaled_data = self.scaler.transform(self.current_data)
            tensor_data = torch.FloatTensor(scaled_data)

            n_samples_mc = 50
            self.model.train()
            with torch.no_grad():
                mc_outputs = [self.model(tensor_data) for _ in range(n_samples_mc)]
                mc_predictions_tensor = torch.cat(mc_outputs, dim=1)

            mc_predictions = mc_predictions_tensor.cpu().numpy()
            self.model.eval()

            mean_predictions = np.mean(mc_predictions, axis=1)
            std_dev = np.std(mc_predictions, axis=1)

            epsilon = 1e-8
            cv = std_dev / (np.abs(mean_predictions) + epsilon)
            confidence = np.maximum(0, 1 - cv)

            z_score = 1.96
            lower_bounds = mean_predictions - z_score * std_dev
            upper_bounds = mean_predictions + z_score * std_dev

            self.result_table.setRowCount(len(mean_predictions))
            for i, (pred, conf, lower, upper) in enumerate(
                    zip(mean_predictions, confidence, lower_bounds, upper_bounds)):
                self.result_table.setItem(i, 0, QTableWidgetItem(f"{pred:.4f}"))
                self.result_table.setItem(i, 1, QTableWidgetItem(f"{conf:.2%}"))
                self.result_table.setItem(i, 2, QTableWidgetItem(f"{lower:.4f}"))
                self.result_table.setItem(i, 3, QTableWidgetItem(f"{upper:.4f}"))

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.hist(mean_predictions, bins=20, edgecolor='black', alpha=0.7, color="#45b7d1")
            ax.set_xlabel("Prediction Value", fontsize=20)
            ax.set_ylabel("Frequency", fontsize=20)
            ax.set_title("Distribution of Mean Predictions", fontsize=20)
            ax.tick_params(axis='both', labelsize=16)
            self.figure.tight_layout()
            self.canvas.draw()
            self.btn_save.setEnabled(True)
            QMessageBox.information(self, "Success", "Prediction completed!")
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}\n\n{traceback.format_exc()}")
            self.btn_save.setEnabled(False)

    def save_results(self):
        try:
            if self.result_table.rowCount() == 0:
                QMessageBox.warning(self, "No Results", "No results to save.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Save Results", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")
            if path:
                results = []
                for row in range(self.result_table.rowCount()):
                    pred_value = float(self.result_table.item(row, 0).text())
                    conf_str = self.result_table.item(row, 1).text().replace('%', '')
                    conf_value = float(conf_str) / 100
                    lower_bound = float(self.result_table.item(row, 2).text())
                    upper_bound = float(self.result_table.item(row, 3).text())
                    results.append([pred_value, conf_value, lower_bound, upper_bound])

                df = pd.DataFrame(results, columns=["Prediction", "Confidence", "95%_CI_Lower", "95%_CI_Upper"])

                if path.lower().endswith('.csv'):
                    df.to_csv(path, index=False)
                else:
                    df.to_excel(path, index=False)
                QMessageBox.information(self, "Success", "Results saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save results: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MLPPredictorApp()
    window.show()
    sys.exit(app.exec_())