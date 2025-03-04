# API de Reconnaissance des Émotions avec FastAPI et Docker
## 📦 Structure du projet

- `data/` : Contient les images organisées par dossiers d'émotions.
- `models/` : Contient le modèle entraîné après l'exécution.
- `src/` : Contient le code source pour l'entraînement et l'évaluation.


## 🚀 Installation et Lancement

### 1. Clonez le projet
```bash
git clone https://github.com/asmadallaji/emotion_recognition_fastapi.git
cd emotion_recognition_fastapi
```

### 2. Construction et Lancement avec Docker
```bash
docker-compose up --build -d
```

### 3. Entraînement du modèle
```bash
docker-compose run training python ./src/train.py
```

### 4. Tester l'API
- Ouvrez votre navigateur et accédez à l'URL: http://localhost:8000/docs
- Testez l'API en cliquant sur le bouton "Try it out" puis "Execute"
- Vous pouvez également tester l'API avec cURL:
```bash
curl -X 'POST' \
  'http://localhost:8000/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@chemin/vers/image.jpg'

```

### 5. Évaluation du modèle
```bash
docker-compose run training python ./src/evaluate.py
```
