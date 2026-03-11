from core.core_logic import build_new_model, get_datagen, IMG_SIZE, BATCH_SIZE
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import class_weight

# --- НАСТРОЙКИ ---
TRAIN_DIR = 'data/train'
MODEL_SAVE_PATH = 'models/model5.h5'
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# 1. ЗАГРУЗКА ДАННЫХ
train_gen = get_datagen(augment=True).flow_from_directory(
    TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE, class_mode='categorical',
    subset='training' # Если в core_logic прописан validation_split
)

# Расчет весов для борьбы с дисбалансом (исправляем ошибки в редких классах)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(weights))

# 2. ОБУЧЕНИЕ
model = build_new_model(num_classes=train_gen.num_classes)
print("Старт обучения...")

# Сохраняем историю обучения в переменную history
history = model.fit(
    train_gen,
    epochs=15,
    class_weight=class_weights
)

# 3. СОХРАНЕНИЕ И ГРАФИКИ
model.save(MODEL_SAVE_PATH)
print(f"Готово! Модель сохранена в {MODEL_SAVE_PATH}")

# Отрисовка графиков
plt.figure(figsize=(12, 5))

# График точности
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точность (Train)')
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Точность (Val)')
plt.title('Успешность обучения (Accuracy)')
plt.legend()

# График потерь
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Потери (Train)')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Потери (Val)')
plt.title('Функция потерь (Loss)')
plt.legend()

plt.savefig('results/training_charts.png')
print("Графики обучения сохранены в results/training_charts.png")
plt.show()