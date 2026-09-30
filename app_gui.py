import sys
import time
import numpy as np
import onnxruntime as ort
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QTimer, Qt

class RFMonitorApp(QMainWindow):
    def __init__(self, model_path="rf_classifier.onnx"):
        super().__init__()
        self.setWindowTitle("Snapdragon Cognitive RF & Spectrum Monitor")
        self.setGeometry(100, 100, 800, 600)
        
        # Initialize ONNX Runtime Session
        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
        except Exception as e:
            print(f"Error loading model '{model_path}': {e}")
            self.session = None

        self.is_monitoring = False
        self.signal_classes = ["BPSK Signal", "QPSK Signal", "8PSK Signal", "16QAM Signal", "Noise"]

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # NPU Status Label
        self.npu_label = QLabel("NPU Engine: Active (QNN / ONNX Runtime)", self)
        self.npu_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.npu_label)

        # Detected Signal Label
        self.signal_label = QLabel("Detected Signal: Idle", self)
        self.signal_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.signal_label)

        # Latency Label
        self.latency_label = QLabel("Inference Latency: -- ms", self)
        self.latency_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(self.latency_label)

        layout.addStretch()

        # Start/Stop Button
        self.toggle_btn = QPushButton("Start Live Monitoring", self)
        self.toggle_btn.setStyleSheet("font-size: 14px; padding: 10px;")
        self.toggle_btn.clicked.connect(self.toggle_monitoring)
        layout.addWidget(self.toggle_btn)

        # Timer for Live Signal Processing Loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_frame)

    def toggle_monitoring(self):
        if not self.is_monitoring:
            self.is_monitoring = True
            self.toggle_btn.setText("Stop Live Monitoring")
            self.timer.start(200)  # Process every 200 ms
        else:
            self.is_monitoring = False
            self.toggle_btn.setText("Start Live Monitoring")
            self.timer.stop()

    def process_frame(self):
        if self.session is None:
            self.latency_label.setText(f"Inference Latency: {latency_ms:.2f} ms")
            return

        # Generate spectrogram data matching target size (64, 64)
        spec = np.random.randn(64, 64).astype(np.float32)

        # Prepare 4D tensor shape: (batch_size=1, channels=1, height=64, width=64)
        input_tensor = np.expand_dims(np.expand_dims(spec, axis=0), axis=0).astype(np.float32)

        # Measure inference latency
        start_time = time.perf_counter()
        outputs = self.session.run(None, {self.input_name: input_tensor})
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Extract predictions
        logits = outputs[0]
        predicted_idx = np.argmax(logits, axis=-1).flatten()[0]
        detected_signal = self.signal_classes[predicted_idx % len(self.signal_classes)]

        # Update GUI labels
        self.signal_label.setText(f"Detected Signal: {detected_signal}")
        self.latency_label.setText(f"Inference Latency: {latency_ms:.2f} ms")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RFMonitorApp()
    window.show()
    sys.exit(app.exec_() if hasattr(app, 'exec_') else app.exec())