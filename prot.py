import cv2
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model

    print("TensorFlow загружен успешно.")
except ImportError:
    print("Ошибка: Установите библиотеку через 'pip install tensorflow'")
    exit()

# --- КОНФИГУРАЦИЯ ---
MODEL_PATH = 'models/model2.h5'
LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# 1. Загрузка модели
if not os.path.exists(MODEL_PATH):
    print(f"Ошибка: Файл модели не найден по пути {MODEL_PATH}")
    exit()

model = load_model(MODEL_PATH)
print(f"Модель {MODEL_PATH} загружена. Запуск камеры...")

# 2. Инициализация детектора
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. Захват видео
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        try:
            roi_color = frame[y:y + h, x:x + w]

            roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
            roi_resized = cv2.resize(roi_rgb, (48, 48))
            roi_normalized = roi_resized.astype('float32') / 255.0
            roi_input = np.expand_dims(roi_normalized, axis=0)

            # ПРЕДСКАЗАНИЕ
            preds = model.predict(roi_input, verbose=0)
            max_idx = np.argmax(preds[0])
            emotion = LABELS[max_idx]
            confidence = preds[0][max_idx] * 100

            # ОТРИСОВКА
            color = (0, 255, 0)  # Зеленый
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{emotion} {confidence:.1f}%", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        except Exception as e:
            print(f"Ошибка обработки кадра: {e}")

    # Вывод результата
    cv2.imshow('AI Emotion Detector (Press Q to Exit)', frame)

    # Выход по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()