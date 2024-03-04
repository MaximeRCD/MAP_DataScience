from fastapi import FastAPI

app = FastAPI(
    title = "Application de détection de places de parking vides",
    description = "Un modèle d'apprentissage automatique pour détecter les places de parking libres à partir d'images satellitaires."
)


@app.get("/")
async def root():
    return {"message": "Il y a du boulot"}