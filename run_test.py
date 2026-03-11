import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from core.core_logic import get_datagen, IMG_SIZE

TEST_DIR = 'data/test'
MODEL_PATH = 'models/model2.h5'

# 1. ЗАГРУЗКА
model = load_model(MODEL_PATH)
test_gen = get_datagen(augment=False).flow_from_directory(
    TEST_DIR, target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=32, class_mode='categorical',
    shuffle=False
)

# 2. ПРЕДСКАЗАНИЯ
print("Запуск финальных тестов...")
predictions = model.predict(test_gen)
y_pred = np.argmax(predictions, axis=1)
y_true = test_gen.classes
class_labels = list(test_gen.class_indices.keys())

# 3. МАТРИЦА ОШИБОК
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels, yticklabels=class_labels)
plt.title('Матрица ошибок (Confusion Matrix)')
plt.xlabel('Предсказано')
plt.ylabel('Реальность')
plt.savefig('results/confusion_matrix.png')
plt.show()

# 4. ТЕКСТОВАЯ СТАТИСТИКА
report = classification_report(y_true, y_pred, target_names=class_labels)
print(report)

with open('results/final_report.txt', 'w') as f:
    f.write(report)
print("Статистика сохранена в results/final_report.txt")