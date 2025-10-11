import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import copy

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QGroupBox, QFormLayout,
    QSpinBox, QDoubleSpinBox, QTextBrowser, QProgressBar, QFrame, QSizePolicy,
    QDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

Hp = {
    'kernel1': 3, 'kernel2': 5, 'kernel3': 7,
    'stride1': 1, 'stride2': 1, 'stride3': 1,
    'channel1': 32, 'channel2': 32, 'channel3': 64,
    'channel4': 64, 'lr': 0.0001, 'epochs': 200, 'batch_size': 32,
    'negative_slope': 0.365, 'dp': 0.2, 'weight_decay': 0.0001,
}


class DeepSpNet(nn.Module):
    def __init__(self, in_features, num_classes, params):
        super(DeepSpNet, self).__init__()

        self.params = params

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=params['channel1'], kernel_size=params['kernel1'],
                               stride=params['stride1'])
        self.conv2 = nn.Conv1d(in_channels=params['channel1'], out_channels=params['channel2'],
                               kernel_size=params['kernel2'], stride=params['stride2'])
        self.conv3 = nn.Conv1d(in_channels=params['channel2'], out_channels=params['channel3'],
                               kernel_size=params['kernel3'], stride=params['stride3'])

        len_after_conv1 = math.floor((in_features - params['kernel1']) / params['stride1']) + 1
        len_after_conv2 = math.floor((len_after_conv1 - params['kernel2']) / params['stride2']) + 1
        self.final_conv_length = math.floor((len_after_conv2 - params['kernel3']) / params['stride3']) + 1

        if self.final_conv_length <= 0:
            raise ValueError("Kernel sizes too large for input features, resulting in zero or negative output length.")

        self.fc4 = nn.Linear(in_features=params['channel3'] * self.final_conv_length, out_features=params['channel4'])
        self.bn4 = nn.BatchNorm1d(num_features=params['channel4'])
        self.fc5 = nn.Linear(in_features=params['channel4'], out_features=num_classes)

        self.drop = nn.Dropout(params['dp'])
        self.relu = nn.LeakyReLU(negative_slope=params['negative_slope'])

    def forward(self, x):
        out_conv1 = self.relu(self.conv1(x))
        out_conv2 = self.relu(self.conv2(out_conv1))
        out_conv3 = self.relu(self.conv3(out_conv2))
        out_flattened = out_conv3.view(-1, self.params['channel3'] * self.final_conv_length)
        out_fc4 = self.drop(self.relu(self.bn4(self.fc4(out_flattened))))
        output = self.fc5(out_fc4)
        return output


class TrainingThread(QThread):
    log_updated = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    epoch_metrics_updated = pyqtSignal(dict)
    training_finished = pyqtSignal(object)

    def __init__(self, train_data, test_data, params):
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.params = params
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None

    def run(self):
        try:
            self.log_updated.emit("<b>[1/5] Preparing data...</b>")

            y_full_train = self.train_data.iloc[:, -1]
            X_full_train = self.train_data.iloc[:, :-1]

            X_train, X_val, y_train, y_val = train_test_split(
                X_full_train, y_full_train, test_size=0.2, random_state=42, stratify=y_full_train
            )

            self.log_updated.emit(f"- Train/Validation split: {len(X_train)} / {len(X_val)} samples.")

            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            X_train_tensor = torch.unsqueeze(torch.from_numpy(X_train_scaled), 1).float()
            y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
            X_val_tensor = torch.unsqueeze(torch.from_numpy(X_val_scaled), 1).float()
            y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

            train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                                      batch_size=self.params['batch_size'], shuffle=True)

            self.log_updated.emit("<b>[2/5] Initializing model...</b>")
            num_classes = len(y_full_train.unique())
            in_features = X_full_train.shape[1]

            self.model = DeepSpNet(in_features, num_classes, self.params).to(self.device)

            class_counts = y_train.value_counts().sort_index().values
            class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
            class_weights = (class_weights / class_weights.sum() * num_classes).to(self.device)
            self.log_updated.emit(f"- Using Class Weights: {['{:.2f}'.format(w) for w in class_weights]}")

            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.Adam(self.model.parameters(), lr=self.params['lr'],
                                   weight_decay=self.params['weight_decay'])

            self.log_updated.emit(f"<b>[3/5] Starting training on {self.device}...</b><hr>")
            best_val_acc = 0.0
            best_model_wts = None

            for epoch in range(self.params['epochs']):
                self.model.train()
                train_loss, train_correct, train_total = 0, 0, 0

                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    train_total += y_batch.size(0)
                    train_correct += (predicted == y_batch).sum().item()

                train_acc = 100 * train_correct / train_total

                val_loss, val_acc = self._evaluate_epoch(X_val_tensor, y_val_tensor, criterion)

                self.log_updated.emit(
                    f"Epoch {epoch + 1}/{self.params['epochs']} -> Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

                self.progress_updated.emit(int((epoch + 1) * 100 / self.params['epochs']))
                self.epoch_metrics_updated.emit({'loss': val_loss, 'accuracy': val_acc})

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    self.log_updated.emit(
                        f"<font color='green'><b>   -> New best model found! Val Acc: {best_val_acc:.2f}%</b></font>")

            self.log_updated.emit("<hr><b>[4/5] Training complete. Loading best model...</b>")
            if best_model_wts:
                self.model.load_state_dict(best_model_wts)

            self.log_updated.emit("<b>[5/5] Performing final evaluation...</b>")
            train_metrics = self._evaluate_dataset(self.train_data, "Full Training Set")
            test_metrics = self._evaluate_dataset(self.test_data, "Test Set")

            results = {
                'model': self.model,
                'scaler': self.scaler,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics
            }
            self.training_finished.emit(results)

        except Exception as e:
            import traceback
            self.log_updated.emit(
                f"<font color='red'><b>Error in training thread:</b><br>{e}<br>{traceback.format_exc()}</font>")
            self.training_finished.emit(None)

    def _evaluate_epoch(self, X_tensor, y_tensor, criterion):
        self.model.eval()
        correct, total = 0, 0
        running_loss = 0.0
        with torch.no_grad():
            X_tensor, y_tensor = X_tensor.to(self.device), y_tensor.to(self.device)
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            running_loss = loss.item()

            _, predicted = torch.max(outputs, 1)
            total = y_tensor.size(0)
            correct = (predicted == y_tensor).sum().item()
        return running_loss, 100 * correct / total

    def _evaluate_dataset(self, data, dataset_name):
        y_true = data.iloc[:, -1].values
        X_data = data.iloc[:, :-1]

        X_scaled = self.scaler.transform(X_data)
        X_tensor = torch.unsqueeze(torch.from_numpy(X_scaled), 1).float().to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, y_pred_tensor = torch.max(outputs, 1)
            y_pred = y_pred_tensor.cpu().numpy()

        metrics = {
            'name': dataset_name,
            'accuracy': (y_true == y_pred).sum() / len(y_true),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'cm': confusion_matrix(y_true, y_pred),
            'y_true': y_true,
            'y_pred': y_pred
        }
        return metrics


class ConfusionMatrixDialog(QDialog):
    def __init__(self, cm, class_names, dataset_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Confusion Matrix - {dataset_name}")
        self.setMinimumSize(640, 520)

        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={"size": 12})
        ax.set_title(f'Confusion Matrix ({dataset_name})', fontsize=16)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        fig.tight_layout()
        layout = QVBoxLayout()
        layout.addWidget(canvas)
        self.setLayout(layout)


class TrainingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("1D CNN Training Platform")
        self.setGeometry(100, 100, 1600, 900)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        left_sidebar = QFrame()
        left_sidebar.setFixedWidth(550)
        left_sidebar.setStyleSheet("background: #f0f2f5; border-right: 1px solid #d1d5db;")
        sidebar_layout = QVBoxLayout(left_sidebar)
        sidebar_layout.setContentsMargins(30, 30, 30, 30)
        sidebar_layout.setSpacing(30)

        title = QLabel("Training Control")
        title.setStyleSheet(
            "color: #1a202c; font: bold 40px 'Segoe UI'; padding-bottom: 15px; border-bottom: 3px solid #38b2ac; margin-bottom: 15px;")
        sidebar_layout.addWidget(title)

        self.widgets = {}
        self.create_input_group(sidebar_layout)
        self.create_data_controls(sidebar_layout)
        self.create_training_controls(sidebar_layout)
        sidebar_layout.addStretch()
        main_layout.addWidget(left_sidebar)

        right_panel = QFrame()
        right_panel.setStyleSheet("background: #f8f9fa;")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(30, 30, 30, 30)
        right_layout.setSpacing(25)

        metrics_layout = QHBoxLayout()
        self.loss_card = self.create_metric_card("Validation Loss", "0.0000", "#ff6b6b")
        self.acc_card = self.create_metric_card("Validation Acc", "0.00%", "#4ecdc4")
        metrics_layout.addWidget(self.loss_card)
        metrics_layout.addWidget(self.acc_card)
        right_layout.addLayout(metrics_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar { height: 36px; border-radius: 18px; background: #e9ecef; color: #2d3748; font: bold 20px 'Segoe UI'; }
            QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4ecdc4, stop:1 #45b7d1); border-radius: 18px; }
        """)
        right_layout.addWidget(self.progress_bar)

        log_group = QGroupBox("Training Log")
        log_group.setStyleSheet(
            "QGroupBox { border: 1px solid #dee2e6; border-radius: 8px; margin-top: 10px; padding-top: 15px; font: bold 28px 'Segoe UI'; color: #6c757d; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        self.log_view = QTextBrowser()
        self.log_view.setStyleSheet(
            "background: white; border: none; border-radius: 6px; padding: 15px; font-family: 'Consolas', 'Courier New', monospace; font-size: 22px; color: #495057;")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.log_view)
        right_layout.addWidget(log_group, 1)

        main_layout.addWidget(right_panel, 1)

        self.train_data = None
        self.test_data = None
        self.training_results = None
        self.training_thread = None

    def create_input_group(self, layout):
        input_group = QGroupBox("Model Parameters")
        input_group.setStyleSheet(
            "QGroupBox { border: 1px solid #d1d5db; border-radius: 8px; margin-top: 10px; padding-top: 20px; color: #2d3748; font: bold 26px 'Segoe UI'; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 10px; left: 10px; } QLabel { color: #1a202c; font: bold 28px 'Segoe UI'; padding-top: 8px; }")
        form_layout = QFormLayout()
        form_layout.setVerticalSpacing(20)
        form_layout.setLabelAlignment(Qt.AlignRight)

        self.widgets['in_features'] = QSpinBox();
        self.widgets['in_features'].setRange(1, 10000)
        self.widgets['epochs'] = QSpinBox();
        self.widgets['epochs'].setRange(1, 5000);
        self.widgets['epochs'].setValue(Hp['epochs'])
        self.widgets['batch_size'] = QSpinBox();
        self.widgets['batch_size'].setRange(1, 1024);
        self.widgets['batch_size'].setValue(Hp['batch_size'])
        self.widgets['lr'] = QDoubleSpinBox();
        self.widgets['lr'].setDecimals(5);
        self.widgets['lr'].setRange(0.00001, 1.0);
        self.widgets['lr'].setSingleStep(0.0001);
        self.widgets['lr'].setValue(Hp['lr'])
        self.widgets['weight_decay'] = QDoubleSpinBox();
        self.widgets['weight_decay'].setDecimals(5);
        self.widgets['weight_decay'].setRange(0.0, 0.1);
        self.widgets['weight_decay'].setSingleStep(0.0001);
        self.widgets['weight_decay'].setValue(Hp['weight_decay'])
        self.widgets['dp'] = QDoubleSpinBox();
        self.widgets['dp'].setDecimals(2);
        self.widgets['dp'].setRange(0.0, 0.9);
        self.widgets['dp'].setSingleStep(0.05);
        self.widgets['dp'].setValue(Hp['dp'])
        self.widgets['channel4'] = QSpinBox();
        self.widgets['channel4'].setRange(1, 1024);
        self.widgets['channel4'].setValue(Hp['channel4'])
        self.widgets['kernel1'] = QSpinBox();
        self.widgets['kernel1'].setRange(1, 100);
        self.widgets['kernel1'].setValue(Hp['kernel1'])
        self.widgets['kernel2'] = QSpinBox();
        self.widgets['kernel2'].setRange(1, 100);
        self.widgets['kernel2'].setValue(Hp['kernel2'])
        self.widgets['kernel3'] = QSpinBox();
        self.widgets['kernel3'].setRange(1, 100);
        self.widgets['kernel3'].setValue(Hp['kernel3'])

        for name, widget in self.widgets.items(): self.style_spinbox(widget)

        form_layout.addRow("Input Features:", self.widgets['in_features'])
        form_layout.addRow("Training Epochs:", self.widgets['epochs'])
        form_layout.addRow("Batch Size:", self.widgets['batch_size'])
        form_layout.addRow("Learning Rate:", self.widgets['lr'])
        form_layout.addRow("L2 Regularization:", self.widgets['weight_decay'])
        form_layout.addRow("Dropout Rate:", self.widgets['dp'])
        form_layout.addRow("Dense Layer Units:", self.widgets['channel4'])
        form_layout.addRow("Kernel Size 1:", self.widgets['kernel1'])
        form_layout.addRow("Kernel Size 2:", self.widgets['kernel2'])
        form_layout.addRow("Kernel Size 3:", self.widgets['kernel3'])

        input_group.setLayout(form_layout)
        layout.addWidget(input_group)

    def start_training(self):
        if self.train_data is None or self.test_data is None:
            QMessageBox.warning(self, "Warning", "Please import both training and test sets first.")
            return

        if self.training_thread and self.training_thread.isRunning():
            QMessageBox.warning(self, "Busy", "A training process is already running.")
            return

        params = {key: widget.value() for key, widget in self.widgets.items()}
        params.update({
            'stride1': Hp['stride1'], 'stride2': Hp['stride2'], 'stride3': Hp['stride3'],
            'channel1': Hp['channel1'], 'channel2': Hp['channel2'], 'channel3': Hp['channel3'],
            'negative_slope': Hp['negative_slope']
        })

        if params['in_features'] != (self.train_data.shape[1] - 1):
            QMessageBox.warning(self, "Mismatch Error", "Input feature count does not match loaded data.")
            return

        self.log_view.clear()
        self.log_view.append("<b>Initializing training thread...</b>")
        self.progress_bar.setValue(0)

        self.train_button.setDisabled(True)
        self.save_button.setDisabled(True)
        self.import_train_button.setDisabled(True)
        self.import_test_button.setDisabled(True)

        self.training_thread = TrainingThread(self.train_data.copy(), self.test_data.copy(), params)

        self.training_thread.log_updated.connect(self._update_log)
        self.training_thread.progress_updated.connect(self._update_progress)
        self.training_thread.epoch_metrics_updated.connect(self._update_metrics)
        self.training_thread.training_finished.connect(self._on_training_finished)
        self.training_thread.finished.connect(self._on_thread_finished)

        self.training_thread.start()

    def _update_log(self, message):
        self.log_view.append(message)

    def _update_progress(self, value):
        self.progress_bar.setValue(value)

    def _update_metrics(self, metrics):
        self.loss_card.findChild(QLabel, "value").setText(f"{metrics['loss']:.4f}")
        self.acc_card.findChild(QLabel, "value").setText(f"{metrics['accuracy']:.2f}%")

    def _on_training_finished(self, results):
        if results:
            self.training_results = results
            self.log_view.append("<hr><b>Displaying final results...</b>")
            self._display_final_metrics(results['train_metrics'])
            self._display_final_metrics(results['test_metrics'])
        else:
            self.log_view.append("<hr><font color='red'><b>Training failed or was aborted.</b></font>")

    def _on_thread_finished(self):
        self.log_view.append("<hr><b>Thread has finished execution.</b>")
        self.train_button.setEnabled(True)
        if self.training_results:
            self.save_button.setEnabled(True)
        self.import_train_button.setEnabled(True)
        self.import_test_button.setEnabled(True)
        self.training_thread = None

    def _display_final_metrics(self, metrics):
        name = metrics['name']
        log_html = (f"<hr><b>--- Final Evaluation on {name} ---</b><br>"
                    f"<b>Accuracy:</b> <font color='#4ecdc4'>{metrics['accuracy']:.4f}</font><br>"
                    f"<b>Precision (Macro):</b> <font color='#4ecdc4'>{metrics['precision']:.4f}</font><br>"
                    f"<b>Recall (Macro):</b> <font color='#4ecdc4'>{metrics['recall']:.4f}</font><br>"
                    f"<b>F1-Score (Macro):</b> <font color='#4ecdc4'>{metrics['f1']:.4f}</font><br>")
        self.log_view.append(log_html)

        class_labels = np.unique(np.concatenate((metrics['y_true'], metrics['y_pred'])))
        class_names = [f'Class {label}' for label in class_labels]
        dialog = ConfusionMatrixDialog(metrics['cm'], class_names, name, self)
        dialog.exec_()

    def save_model(self):
        if not self.training_results or 'model' not in self.training_results:
            QMessageBox.warning(self, "Error", "No trained model available to save.")
            return

        file_name, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "PyTorch Model (*.pth)")
        if file_name:
            data_to_save = {
                'model_state_dict': self.training_results['model'].state_dict(),
                'scaler': self.training_results['scaler'],
                'model_params': self.training_results['model'].params,
                'num_classes': self.training_results['model'].fc5.out_features,
                'in_features': self.widgets['in_features'].value()
            }
            torch.save(data_to_save, file_name)
            QMessageBox.information(self, "Success", f"Model and scaler saved to {file_name}")

    def create_data_controls(self, layout):
        btn_group = QGroupBox("Data Operations")
        btn_group.setStyleSheet(
            "QGroupBox { border: 1px solid #d1d5db; border-radius: 8px; margin-top: 10px; padding-top: 30px; color: #2d3748; font: bold 26px 'Segoe UI'; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 10px; left: 10px; }")
        vbox = QVBoxLayout()
        vbox.setSpacing(15)
        self.import_train_button = self.create_control_button("Import Training Set", "#4ecdc4")
        self.import_train_button.clicked.connect(self.import_train_data)
        self.import_test_button = self.create_control_button("Import Test Set", "#4ecdc4")
        self.import_test_button.clicked.connect(self.import_test_data)
        vbox.addWidget(self.import_train_button)
        vbox.addWidget(self.import_test_button)
        btn_group.setLayout(vbox)
        layout.addWidget(btn_group)

    def create_training_controls(self, layout):
        btn_group = QGroupBox("Training Control")
        btn_group.setStyleSheet(
            "QGroupBox { border: 1px solid #d1d5db; border-radius: 8px; margin-top: 10px; padding-top: 30px; color: #2d3748; font: bold 26px 'Segoe UI'; } QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 10px; left: 10px; }")
        vbox = QVBoxLayout()
        vbox.setSpacing(15)
        self.train_button = self.create_control_button("Start Training", "#4ecdc4")
        self.train_button.clicked.connect(self.start_training)
        self.save_button = self.create_control_button("Save Model", "#45b7d1")
        self.save_button.clicked.connect(self.save_model)
        vbox.addWidget(self.train_button)
        vbox.addWidget(self.save_button)
        btn_group.setLayout(vbox)
        layout.addWidget(btn_group)

    def create_control_button(self, text, color="#4a5568"):
        btn = QPushButton(text)
        btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn.setMinimumHeight(70)
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {color}; color: black; border: none; border-radius: 8px; padding: 10px 15px; font: bold 28px 'Segoe UI'; }} QPushButton:hover {{ background-color: {self.adjust_color(color, -20)}; }} QPushButton:pressed {{ background-color: {self.adjust_color(color, -40)}; }} QPushButton:disabled {{ background-color: #bdc3c7; color: #555555; }}")
        return btn

    def create_metric_card(self, title, value, color):
        card = QFrame()
        card.setStyleSheet("background: white; border-radius: 10px; border: 1px solid #e0e0e0; padding: 25px;")
        layout = QVBoxLayout(card)
        layout.setSpacing(5)
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #6c757d; font: bold 28px 'Segoe UI'; border: none; padding: 0;")
        value_label = QLabel(value)
        value_label.setObjectName("value")
        value_label.setStyleSheet(f"color: {color}; font: bold 52px 'Segoe UI'; border: none; padding: 0;")
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        layout.addStretch()
        return card

    def style_spinbox(self, spinbox):
        spinbox.setStyleSheet(
            "QAbstractSpinBox { border: 1px solid #cbd5e0; border-radius: 8px; padding: 12px; background: #ffffff; color: #1f2937; font: 22px 'Segoe UI'; } QAbstractSpinBox:focus { border: 2px solid #4ecdc4; } QAbstractSpinBox::up-button, QAbstractSpinBox::down-button { subcontrol-origin: border; width: 32px; background-color: #f0f2f5; } QAbstractSpinBox::up-button { subcontrol-position: top right; border-top-right-radius: 8px; border-bottom: 1px solid #cbd5e0; } QAbstractSpinBox::down-button { subcontrol-position: bottom right; border-bottom-right-radius: 8px; } QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background-color: #e2e8f0; }")

    def import_train_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Training Data", "", "Excel Files (*.xls *.xlsx)")
        if file_name:
            try:
                self.train_data = pd.read_excel(file_name, index_col=0, header=None)
                self.widgets['in_features'].setValue(self.train_data.shape[1] - 1)
                self._update_log(f"<b>[Success] Training data loaded:</b> {self.train_data.shape[0]} samples.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Data loading failed: {str(e)}")

    def import_test_data(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Test Data", "", "Excel Files (*.xls *.xlsx)")
        if file_name:
            try:
                self.test_data = pd.read_excel(file_name, index_col=0, header=None)
                self._update_log(f"<b>[Success] Test data loaded:</b> {self.test_data.shape[0]} samples.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Data loading failed: {str(e)}")

    def adjust_color(self, color, amount):
        color = color.lstrip('#');
        rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
        new_rgb = tuple(max(0, min(255, c + amount)) for c in rgb)
        return "#%02x%02x%02x" % new_rgb


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrainingApp()
    window.show()
    sys.exit(app.exec_())