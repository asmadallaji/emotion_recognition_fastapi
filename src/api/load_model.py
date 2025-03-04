from tensorflow.keras.models import load_model
from ..config import MODEL_PATH

model = load_model(MODEL_PATH)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
