import os
import sys
import cv2
import numpy as np


def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
    return os.path.join(base_path, relative_path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if sys.platform == 'win32':
    os.environ["PATH"] += os.pathsep + os.path.dirname(sys.executable)

# 2. ПОПЫТКА ИМПОРТА TENSORFLOW
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    TF_AVAILABLE = True
    print("TensorFlow успешно инициализирован")
except ImportError as e:
    print(f"Критическая ошибка TensorFlow: {e}")
    TF_AVAILABLE = False

from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QComboBox,
                             QVBoxLayout, QHBoxLayout, QWidget, QLabel, QMessageBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt


class EmotionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Emotion Detector - Конкурсный проект")
        self.resize(800, 680)

        # Переменные состояния
        self.cap = None
        self.model = None
        self.labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.models_dir = 'models'

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        self.init_ui()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout()

        # --- Блок видео ---
        self.video_label = QLabel("Камера не активна")
        self.video_label.setStyleSheet("background-color: black; border: 3px solid #222; color: white;")
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.video_label, 0, Qt.AlignCenter)

        # --- Панель управления ---
        controls_group = QWidget()
        controls_layout = QHBoxLayout(controls_group)

        # Селектор моделей
        self.model_selector = QComboBox()
        self.model_selector.setMinimumWidth(250)
        self.refresh_models_list()
        self.model_selector.currentIndexChanged.connect(self.on_model_selected)

        # Кнопки
        self.btn_start = QPushButton("Запустить камеру")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.clicked.connect(self.start_camera)

        self.btn_stop = QPushButton("Остановить")
        self.btn_stop.setMinimumHeight(40)
        self.btn_stop.clicked.connect(self.stop_camera)

        controls_layout.addWidget(QLabel("Выбор модели:"))
        controls_layout.addWidget(self.model_selector)
        controls_layout.addStretch()
        controls_layout.addWidget(self.btn_start)
        controls_layout.addWidget(self.btn_stop)

        self.main_layout.addWidget(controls_group)
        self.central_widget.setLayout(self.main_layout)

        # Таймер видеопотока
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def refresh_models_list(self):
        self.model_selector.clear()
        self.model_selector.addItem("— Выберите модель из папки /models —")

        if os.path.exists(self.models_dir):
            files = [f for f in os.listdir(self.models_dir) if f.endswith(('.h5', '.keras'))]
            if not files:
                print("В папке models нет файлов моделей!")
            self.model_selector.addItems(files)

    def on_model_selected(self, index):
        if index <= 0 or not TF_AVAILABLE:
            self.model = None
            return

        model_name = self.model_selector.currentText()
        model_path = os.path.join(self.models_dir, model_name)

        try:
            # Загружаем модель
            self.model = load_model(model_path)
            print(f"Успешно загружена модель: {model_name}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось загрузить модель:\n{str(e)}")
            self.model_selector.setCurrentIndex(0)

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                QMessageBox.warning(self, "Ошибка", "Не удалось получить доступ к камере")
                self.cap = None
                return
            self.timer.start(30)

    def stop_camera(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.clear()
        self.video_label.setText("Камера остановлена")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Если модель выбрана — анализируем эмоции
            if self.model:
                frame = self.detect_emotions(frame)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def detect_emotions(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            try:
                roi = frame[y:y + h, x:x + w]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype('float32') / 255.0
                roi = np.expand_dims(roi, axis=0)

                # 3. Предсказание
                preds = self.model.predict(roi, verbose=0)
                idx = np.argmax(preds[0])
                label = self.labels[idx]
                prob = preds[0][idx] * 100

                # 4. Отрисовка
                color = (0, 255, 0)  # Зеленый
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} ({prob:.1f}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            except Exception as e:
                print(f"Ошибка анализа лица: {e}")

        return frame


if __name__ == "__main__":
    if not TF_AVAILABLE:
        print("ВНИМАНИЕ: Приложение запущено без поддержки TensorFlow (DLL Error)")

    app = QApplication(sys.argv)

    app.setStyle("Fusion")

    window = EmotionApp()
    window.show()
    sys.exit(app.exec_())