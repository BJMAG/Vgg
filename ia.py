import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='keras')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import classification_report, confusion_matrix

# === RUTAS ===
ruta_dataset = r"C:\Proyectos\ODIR\ODIR-organizado"
ruta_historial = os.path.join("C:\\Proyectos\\ODIR", "historial_entrenamiento.csv")
ruta_resultados = os.path.join("C:\\Proyectos\\ODIR", "resultados_odir")
ruta_modelo = os.path.join("C:\\Proyectos\\ODIR", "modelo_entrenado.h5")

# Crear carpeta de resultados si no existe
os.makedirs(ruta_resultados, exist_ok=True)

# === Clases a utilizar ===
clases_deseadas = ['Cataract', 'Diabetes', 'Normal']

# === Preprocesamiento con imágenes de 256x256 ===
data_gen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = data_gen.flow_from_directory(
    ruta_dataset,
    target_size=(256, 256),
    batch_size=16,
    classes=clases_deseadas,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_data = data_gen.flow_from_directory(
    ruta_dataset,
    target_size=(256, 256),
    batch_size=16,
    classes=clases_deseadas,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === Callback personalizado para detener si val_accuracy >= 0.85 ===
class DetenerAl85(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get("val_accuracy") is not None and logs["val_accuracy"] >= 0.85:
            print(f"\n✅ ¡Precisión de validación alcanzada! ({logs['val_accuracy']:.2%}) en la época {epoch + 1}. Deteniendo entrenamiento...")
            self.model.stop_training = True

# === Modelo CNN estilo VGG mejorado para 256x256 ===
model = Sequential([
    Input(shape=(256, 256, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# === Compilar modelo ===
model.compile(
    optimizer=Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# === ENTRENAMIENTO ===
print("🔄 Entrenando modelo... (se detendrá si llega a 85% en validación)")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[DetenerAl85()]
)

# === Guardar modelo entrenado ===
model.save(ruta_modelo)
print(f"💾 Modelo guardado en: {ruta_modelo}")

# === Guardar historial ===
df_historial = pd.DataFrame(history.history)
df_historial.to_csv(ruta_historial, index=False)
print("📁 Historial guardado en:", ruta_historial)

# === GRÁFICAS ===
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión por Época')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(ruta_resultados, "precision.png"))
plt.show()

plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida por Época')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(ruta_resultados, "perdida.png"))
plt.show()

# === EVALUACIÓN FINAL ===
val_data.reset()
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_data.classes
labels = list(val_data.class_indices.keys())

print("📋 Reporte de clasificación:")
print(classification_report(y_true, y_pred_classes, target_names=labels))

# Matriz de Confusión
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Matriz de Confusión")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.savefig(os.path.join(ruta_resultados, "matriz_confusion.png"))
plt.show()

