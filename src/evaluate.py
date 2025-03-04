import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import *

model = load_model(MODEL_PATH)

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(directory=VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
                                                color_mode='grayscale', class_mode='categorical', shuffle=False)

# Prédictions
y_pred = model.predict(val_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_generator.classes

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.show()

# Rapport
print(classification_report(y_true, y_pred_classes))
