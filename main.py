from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import uvicorn

# Definición de la clase para el input del modelo
class SentimentRequest(BaseModel):
    text: str

# Carga del modelo y el vectorizador
def load_model_and_vectorizer():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
            if not hasattr(model, 'predict'):
                raise ValueError("El archivo model.pkl no contiene un modelo válido.")
        
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
            if not hasattr(vectorizer, 'transform'):
                raise ValueError("El archivo vectorizer.pkl no contiene un vectorizador válido.")
        
        return model, vectorizer
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar el modelo y el vectorizador: {str(e)}")

model, vectorizer = load_model_and_vectorizer()

# Inicio de la API con FastAPI
app = FastAPI()

# Endpoint para análisis de sentimientos
@app.post("/sentiment/")
def predict_sentiment(request: SentimentRequest):
    try:
        # Transformar el texto usando el vectorizador
        vectorized_text = vectorizer.transform([request.text])
        
        # Realizar la predicción
        prediction = model.predict(vectorized_text)
        
        # Devolver la predicción como una respuesta
        sentiment_label = "positivo" if prediction[0] == 1 else "negativo"
        return {"sentiment": sentiment_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ejecución del servidor de la API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
