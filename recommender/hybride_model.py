#---- Partie 1 : chargement && nettoyage  -------#

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
from sklearn.preprocessing import OneHotEncoder

from surprise import Dataset, Reader, SVD


THRESHOLD = 4.0  # seuil "il aime"


#-- chargement --#
df = pd.read_csv("../data/movies_metadata.csv", low_memory=False)
dr = pd.read_csv("../data/ratings_small.csv", low_memory=False)

# movies_metadata : garder seulement id + title
if "id" not in df.columns or "title" not in df.columns:
    raise ValueError("Colonnes 'id' ou 'title' absentes dans movies_metadata.csv")

movies = df[["id", "title"]].copy()

# nettoyage movies_metadata
movies["movieId"] = pd.to_numeric(movies["id"], errors="coerce")
movies = movies.dropna(subset=["movieId"])
movies["movieId"] = movies["movieId"].astype(int)
movies = movies[["movieId", "title"]]

print("🎬 Films après nettoyage :")
print(movies.head(5))


# vérifier colonnes dans ratings
required_columns = ["userId", "movieId", "rating", "timestamp"]
for c in required_columns:
    if c not in dr.columns:
        raise ValueError(f"Colonne manquante dans ratings_small.csv : {c}")

# garder les colonnes utiles
dr = dr[["userId", "movieId", "rating", "timestamp"]].copy()

# types numériques
dr["userId"] = pd.to_numeric(dr["userId"], errors="coerce")
dr["movieId"] = pd.to_numeric(dr["movieId"], errors="coerce")
dr["rating"] = pd.to_numeric(dr["rating"], errors="coerce")
dr["timestamp"] = pd.to_numeric(dr["timestamp"], errors="coerce")
dr = dr.dropna(subset=["userId", "movieId", "rating", "timestamp"])

# garder la note la plus récente pour chaque (user, movie)
dr = dr.sort_values("timestamp")
dr = dr.drop_duplicates(subset=["userId", "movieId"], keep="last")

# label binaire
dr["label"] = (dr["rating"] >= THRESHOLD).astype(int)

print("\n✅ Evaluations nettoyées :")
print(dr.head(5))


#---- Partie 2 : Modèle 1 – Régression logistique -------#

enc = OneHotEncoder(handle_unknown="ignore", dtype=float)
X = enc.fit_transform(dr[["userId", "movieId"]])
y = dr["label"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

clf = LogisticRegression(
    solver="liblinear",
    max_iter=200,
    class_weight="balanced"
)
clf.fit(X_train, y_train)

probs_test = clf.predict_proba(X_test)[:, 1]
preds_test = (probs_test >= 0.5).astype(int)

acc = accuracy_score(y_test, preds_test)
prec, rec, f1, _ = precision_recall_fscore_support(
    y_test, preds_test, average="binary", zero_division=0
)
auc = roc_auc_score(y_test, probs_test)

print("\n🎯 Résultats du modèle logistique :")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"AUC       : {auc:.4f}")


#---- Partie 3 : Modèle 2 – SVD (Filtrage collaboratif) -------#

reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(
    dr[["userId", "movieId", "rating"]],
    reader
)

trainset = data.build_full_trainset()

svd = SVD(
    n_factors=50,
    n_epochs=20,
    random_state=42
)

print("\n🚂 Entraînement du modèle SVD...")
svd.fit(trainset)
print("✅ Modèle SVD entraîné.")


#---- Partie 4 : films déjà vus par utilisateur -------#

def movies_seen(df_ratings):
    """
    userId -> set(movieId déjà notés)
    """
    seen = {}
    for _, row in df_ratings.iterrows():
        user = int(row["userId"])
        movie = int(row["movieId"])
        if user not in seen:
            seen[user] = set()
        seen[user].add(movie)
    return seen


seen_dict = movies_seen(dr)
print("\n👀 Nombre d'utilisateurs dans seen_dict :", len(seen_dict))


#---- Partie 5 : Fonctions de recommandation -------#

def recommend_logistic_for_user(user_id, clf, enc, dr, movies, seen_dict, topk=10, min_prob=0.6):
    """
    Modèle 1 : Logistique
    Recommande les films les plus probables d'être aimés.
    """
    user_id = int(user_id)

    if user_id not in dr["userId"].unique():
        print("❌ Utilisateur non trouvé.")
        return None

    seen_movies = seen_dict.get(user_id, set())
    all_movies = movies["movieId"].dropna().astype(int).unique()

    candidates = [m for m in all_movies if m not in seen_movies]
    if not candidates:
        print("😅 Il a déjà tout vu.")
        return None

    candidates = np.array(candidates, dtype=int)

    cand_df = pd.DataFrame({
        "userId": [user_id] * len(candidates),
        "movieId": candidates
    })

    X_cand = enc.transform(cand_df[["userId", "movieId"]])
    probs = clf.predict_proba(X_cand)[:, 1]

    mask = probs >= float(min_prob)
    if not np.any(mask):
        print(f"😕 Aucun film avec prob >= {min_prob}")
        return None

    probs = probs[mask]
    candidates_filtered = candidates[mask]

    order = np.argsort(-probs)[:topk]
    top_ids = candidates_filtered[order]
    top_probs = probs[order]

    recs = pd.DataFrame({
        "movieId": top_ids,
        "score_log": top_probs
    }).merge(movies, on="movieId", how="left")

    return recs


def recommend_cf_for_user(user_id, svd_model, dr, movies, seen_dict, topk=10):
    """
    Modèle 2 : CF SVD
    Recommande les films avec la meilleure note prédite.
    """
    user_id = int(user_id)

    if user_id not in dr["userId"].unique():
        print(f"❌ userId {user_id} n'existe pas dans ratings.")
        return None

    seen_movies = seen_dict.get(user_id, set())
    all_movies = movies["movieId"].unique()

    candidates = [m for m in all_movies if m not in seen_movies]
    if not candidates:
        print("😅 Aucun film candidat (tout vu).")
        return None

    preds = []
    for m in candidates:
        est = svd_model.predict(user_id, int(m)).est
        preds.append((m, est))

    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)[:topk]

    top_movie_ids = [p[0] for p in preds_sorted]
    top_scores = [p[1] for p in preds_sorted]

    recs = pd.DataFrame({
        "movieId": top_movie_ids,
        "score_pred": top_scores
    }).merge(movies, on="movieId", how="left")

    return recs


def hybrid_recommend_for_user(user_id, clf, enc, svd_model, dr, movies, seen_dict, topk=10, alpha=0.5):
    """
    Modèle 3 : Hybride
    - alpha * proba_logistique + (1 - alpha) * proba_CF_normalisée
    """
    user_id = int(user_id)

    if user_id not in dr["userId"].unique():
        print(f"❌ userId {user_id} n'existe pas.")
        return None

    seen_movies = seen_dict.get(user_id, set())
    all_movies = movies["movieId"].dropna().astype(int).unique()

    candidates = [m for m in all_movies if m not in seen_movies]
    if not candidates:
        print("😅 Aucun film candidat.")
        return None

    candidates = np.array(candidates, dtype=int)

    # Logistique
    cand_df = pd.DataFrame({
        "userId": [user_id] * len(candidates),
        "movieId": candidates
    })
    X_cand = enc.transform(cand_df[["userId", "movieId"]])
    probs_log = clf.predict_proba(X_cand)[:, 1]   # [0,1]

    # CF SVD : normalisation en [0,1]
    probs_svd = []
    for m in candidates:
        est = svd_model.predict(user_id, int(m)).est  # 0.5 - 5.0
        p = (est - 0.5) / (5.0 - 0.5)  # → [0,1]
        p = max(0.0, min(1.0, p))
        probs_svd.append(p)
    probs_svd = np.array(probs_svd)

    # Score hybride
    scores = alpha * probs_log + (1.0 - alpha) * probs_svd

    order = np.argsort(-scores)[:topk]
    top_ids = candidates[order]
    top_scores = scores[order]

    recs = pd.DataFrame({
        "movieId": top_ids,
        "score_hybride": top_scores
    }).merge(movies, on="movieId", how="left")

    return recs


#---- Partie 6 : Test & comparaison -------#

if __name__ == "__main__":
    user_test = 2  # change si tu veux

    print(f"\n=== Recommandations pour l'utilisateur {user_test} ===")

    recs_log = recommend_logistic_for_user(user_test, clf, enc, dr, movies, seen_dict, topk=10, min_prob=0.6)
    recs_cf  = recommend_cf_for_user(user_test, svd, dr, movies, seen_dict, topk=10)
    recs_hyb = hybrid_recommend_for_user(user_test, clf, enc, svd, dr, movies, seen_dict, topk=10, alpha=0.5)

    if recs_log is not None:
        print("\n🔷 M1 – Logistique :")
        print(recs_log[["title", "score_log"]])

    if recs_cf is not None:
        print("\n🔶 M2 – CF SVD :")
        print(recs_cf[["title", "score_pred"]])

    if recs_hyb is not None:
        print("\n🟣 M3 – Hybride :")
        print(recs_hyb[["title", "score_hybride"]])
# ---- Partie 1 : Imports & constantes -------#

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)
from sklearn.preprocessing import OneHotEncoder

from surprise import Dataset, Reader, SVD

import mlflow

THRESHOLD = 4.0          # seuil "j'aime" (rating >= 4)
ALPHA = 0.5              # poids du modèle logistique dans l'hybride (0.5 = 50/50)


# ---- Partie 2 : Chargement & nettoyage -------#

df = pd.read_csv("../data/movies_metadata.csv", low_memory=False)
dr = pd.read_csv("../data/ratings_small.csv", low_memory=False)

# Vérif colonnes
if "id" not in df.columns or "title" not in df.columns:
    raise ValueError("Colonnes 'id' ou 'title' absentes dans movies_metadata.csv")

movies = df[["id", "title"]].copy()
movies["movieId"] = pd.to_numeric(movies["id"], errors="coerce")
movies = movies.dropna(subset=["movieId"])
movies["movieId"] = movies["movieId"].astype(int)
movies = movies[["movieId", "title"]]

print("🎬 Films après nettoyage :")
print(movies.head(5))

required_columns = ["userId", "movieId", "rating", "timestamp"]
for c in required_columns:
    if c not in dr.columns:
        raise ValueError(f"Colonne manquante dans ratings_small.csv : {c}")

dr = dr[["userId", "movieId", "rating", "timestamp"]].copy()

dr["userId"] = pd.to_numeric(dr["userId"], errors="coerce")
dr["movieId"] = pd.to_numeric(dr["movieId"], errors="coerce")
dr["rating"] = pd.to_numeric(dr["rating"], errors="coerce")
dr["timestamp"] = pd.to_numeric(dr["timestamp"], errors="coerce")
dr = dr.dropna(subset=["userId", "movieId", "rating", "timestamp"])

# Garder la note la plus récente pour chaque (user, movie)
dr = dr.sort_values("timestamp")
dr = dr.drop_duplicates(subset=["userId", "movieId"], keep="last")

# Label binaire pour la classification
dr["label"] = (dr["rating"] >= THRESHOLD).astype(int)

print("\n✅ Évaluations nettoyées :")
print(dr.head(5))


# ---- Partie 3 : Split train/test au niveau des interactions -------#
# (on split le DataFrame complet, comme ça on garde userId/movieId pour le test)

train_df, test_df = train_test_split(
    dr,
    test_size=0.1,
    random_state=42,
    stratify=dr["label"]
)

print(f"\n📊 Taille train : {len(train_df)}, test : {len(test_df)}")


# ---- Partie 4 : Modèle logistique -------#

enc = OneHotEncoder(handle_unknown="ignore", dtype=float)

X_train = enc.fit_transform(train_df[["userId", "movieId"]])
y_train = train_df["label"].to_numpy()

X_test = enc.transform(test_df[["userId", "movieId"]])
y_test = test_df["label"].to_numpy()

logit = LogisticRegression(
    solver="liblinear",
    max_iter=200,
    class_weight="balanced"
)
logit.fit(X_train, y_train)

probs_logit = logit.predict_proba(X_test)[:, 1]

print("\n🔹 Modèle logistique entraîné.")


# ---- Partie 5 : Modèle SVD (Surprise) -------#

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(
    dr[["userId", "movieId", "rating"]],
    reader
)

# On entraîne SVD sur TOUT le dataset (full trainset),
# car en reco on veut utiliser toutes les interactions connues.
trainset_svd = data.build_full_trainset()

svd = SVD(
    n_factors=50,
    n_epochs=20,
    random_state=42
)

print("\n🚂 Entraînement du modèle SVD (full trainset)...")
svd.fit(trainset_svd)
print("✅ Modèle SVD entraîné.")


# ---- Partie 6 : Évaluation hybride sur le set de test -------#

def svd_score_normalized(user_id, movie_id, svd_model):
    """
    Retourne la note SVD normalisée entre 0 et 1.
    SVD prédit une note entre 0.5 et 5.0 → on normalise.
    """
    est = svd_model.predict(int(user_id), int(movie_id)).est
    # normalisation simple sur [0,1]
    score = (est - 0.5) / (5.0 - 0.5)
    # clamp au cas où
    return float(np.clip(score, 0.0, 1.0))


# On calcule les scores SVD normalisés pour chaque interaction du test set
svd_scores_norm = []
for _, row in test_df.iterrows():
    uid = row["userId"]
    mid = row["movieId"]
    svd_scores_norm.append(svd_score_normalized(uid, mid, svd))

svd_scores_norm = np.array(svd_scores_norm)

# Vérif alignement des tailles
assert len(svd_scores_norm) == len(probs_logit) == len(y_test)

# Score hybride
hybrid_scores = ALPHA * probs_logit + (1.0 - ALPHA) * svd_scores_norm
hybrid_preds = (hybrid_scores >= 0.5).astype(int)

acc_h = accuracy_score(y_test, hybrid_preds)
prec_h, rec_h, f1_h, _ = precision_recall_fscore_support(
    y_test, hybrid_preds, average="binary", zero_division=0
)
auc_h = roc_auc_score(y_test, hybrid_scores)

print("\n🎯 Résultats HYBRIDE :")
print(f"Accuracy  : {acc_h:.4f}")
print(f"Precision : {prec_h:.4f}")
print(f"Recall    : {rec_h:.4f}")
print(f"F1-score  : {f1_h:.4f}")
print(f"AUC       : {auc_h:.4f}")


# ---- Partie 7 : Log MLflow -------#

mlflow.set_experiment("MovieMood-Reco")

with mlflow.start_run(run_name="hybrid_v1"):
    mlflow.log_param("model_type", "hybrid_logit_svd")
    mlflow.log_param("alpha_logit", ALPHA)
    mlflow.log_param("n_factors_svd", 50)
    mlflow.log_param("n_epochs_svd", 20)
    mlflow.log_param("threshold_label", THRESHOLD)

    mlflow.log_metric("accuracy", acc_h)
    mlflow.log_metric("precision", prec_h)
    mlflow.log_metric("recall", rec_h)
    mlflow.log_metric("f1", f1_h)
    mlflow.log_metric("auc", auc_h)

print("\n✅ Modèle hybride loggué dans MLflow.")


# ---- Partie 8 : Utilitaires de reco hybride -------#

def movies_seen(df_ratings):
    """
    userId -> set(movieId déjà notés)
    """
    seen = {}
    for _, row in df_ratings.iterrows():
        user = int(row["userId"])
        movie = int(row["movieId"])
        if user not in seen:
            seen[user] = set()
        seen[user].add(movie)
    return seen


seen_dict = movies_seen(dr)
print("\n👀 Nombre d'utilisateurs :", len(seen_dict))


def recommend_hybrid_for_user(
    user_id,
    logit_model,
    encoder,
    svd_model,
    dr,
    movies,
    seen_dict,
    topk=10,
    min_score=0.6,
    alpha=ALPHA
):
    """
    Recommandations hybrides pour un utilisateur :
    - p_logit = proba d'aimer (modèle logistique)
    - p_svd   = score SVD normalisé [0,1]
    - score_hybride = alpha * p_logit + (1-alpha) * p_svd
    """

    user_id = int(user_id)

    if user_id not in dr["userId"].unique():
        print("❌ Utilisateur non trouvé dans ratings.")
        return None

    seen_movies = seen_dict.get(user_id, set())
    all_movies = movies["movieId"].unique()

    candidates = [m for m in all_movies if m not in seen_movies]
    if not candidates:
        print("😅 Aucun film candidat (il a déjà tout vu).")
        return None

    candidates = np.array(candidates, dtype=int)

    # Features pour le logit
    cand_df = pd.DataFrame({
        "userId": [user_id] * len(candidates),
        "movieId": candidates
    })
    X_cand = encoder.transform(cand_df[["userId", "movieId"]])
    probs_log = logit_model.predict_proba(X_cand)[:, 1]

    # Scores SVD normalisés
    svd_scores = np.array([
        svd_score_normalized(user_id, mid, svd_model) for mid in candidates
    ])

    # Score hybride
    hybrid_scores = alpha * probs_log + (1.0 - alpha) * svd_scores

    # Filtre min_score
    mask = hybrid_scores >= float(min_score)
    if not np.any(mask):
        print("Aucun film avec score hybride >= min_score")
        return None

    candidates = candidates[mask]
    hybrid_scores = hybrid_scores[mask]

    # Top-k
    order = np.argsort(-hybrid_scores)[:topk]
    top_ids = candidates[order]
    top_scores = hybrid_scores[order]

    recs = pd.DataFrame({
        "movieId": top_ids,
        "hybrid_score": top_scores
    }).merge(movies, on="movieId", how="left")

    return recs


# ---- Partie 9 : Test simple -------#

if __name__ == "__main__":
    user_test = 2
    recs_h = recommend_hybrid_for_user(
        user_test,
        logit,
        enc,
        svd,
        dr,
        movies,
        seen_dict,
        topk=10,
        min_score=0.6,
        alpha=ALPHA
    )

    if recs_h is not None:
        print(f"\n🎬 Top 10 recommandations HYBRIDES pour l'utilisateur {user_test} :")
        print(recs_h[["title", "hybrid_score"]])
