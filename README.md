# 🎬 MovieMood API

API de recommandation de films hybride (Logistic + SVD) avec explications détaillées des recommandations.

## Prérequis

* Docker (version récente)
* Docker Compose (intégré avec Docker Desktop)
* Python 3.10+ (si tu veux exécuter localement sans Docker)

## Structure du projet

```
.
├── api/
│   ├── main.py
│   ├── routes/
│   ├── models.py
│   └── services/
├── data/
│   ├── movies_metadata.csv
│   └── ratings_small.csv
├── Dockerfile
├── docker-compose.yml
└── README.md
```

* `api/` → code FastAPI et services de recommandation
* `data/` → fichiers CSV avec films et évaluations
* `Dockerfile` → instructions pour construire l'image Docker
* `docker-compose.yml` → configuration pour lancer l'API et dépendances

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/Shad0wz93/MovieMood.git
cd MovieMood
```

### 2. Copier les données

Place les fichiers `movies_metadata.csv` et `ratings_small.csv` dans le dossier `data/`.

## Build de l'image Docker

```bash
docker-compose build ou docker compose build
```

Cette commande construit l'image Docker avec tous les packages nécessaires (`FastAPI`, `scikit-learn`, `pandas`, `surprise`, etc.).

## Lancement du service

```bash
docker-compose up ou docker compose up
```

* Le service démarre sur : `http://localhost:8000`
* L'API charge les modèles Logistic + SVD au démarrage (SVD fait 200 epochs, Logistic jusqu'à 200 itérations).
* Logs de chargement :

```
🚀 Chargement du modèle hybride...
✅ Modèle chargé avec succès
```

## Endpoints principaux

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Accueil et description de l'API |
| GET | `/health` | Vérifie l'état du service |
| POST | `/api/v1/predict` | Obtenir des recommandations top-K |
| POST | `/api/v1/explain` | Expliquer pourquoi ces films ont été recommandés |
| GET | `/api/v1/users` | Liste des IDs utilisateurs |
| GET | `/api/v1/users/{user_id}/seen` | Films déjà vus par un utilisateur |
| GET | `/api/v1/users/{user_id}/stats` | Statistiques d'un utilisateur |

* Documentation interactive : `http://localhost:8000/docs`
