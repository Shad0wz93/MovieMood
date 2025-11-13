#---- partie 1 :chargement && nettoyage  -------#

import numpy as np 
import pandas as pd


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

# seuil pour dire "ça lui plaît"
THRESHOLD = 4.0

#--chargement --#
df= pd.read_csv("movies_metadata.csv", low_memory=False)
dr=pd.read_csv("ratings_small.csv", low_memory=False)
# movies_metadata: garder seulment le colomnes id et title #
if 'id'not in df.columns or "title" not in df.columns :
    print("colonnes id ou title absent ")
    
else:
 movies = df[["id", "title"]].copy()

#nettoyage movies_metadata
 movies["movieId"] = pd.to_numeric(movies["id"], errors="coerce")
 movies = movies.dropna(subset=["movieId"])
#garder ce quil faut de movies_metadata
 movies = movies [["movieId", "title"]]
 print (movies.head(5))


 #verifier lexistance dabord 
 required_colomns = ["userId", "movieId","rating", "timestamp" ]
 for c in required_colomns:
    if c not in dr.columns:
       raise ValueError("colomn not existe")
    
#rating : garder les colomns userId moviesId rating #
dr =dr[["userId", "movieId","rating", "timestamp"]].copy()

#rating :nettoyage types 
dr["userId"]= pd.to_numeric(dr["userId"], errors="coerce")
dr["movieId"]=pd.to_numeric(dr["movieId"], errors="coerce")
dr["rating"]=pd.to_numeric(dr["rating"],errors="coerce")
dr[ "timestamp"]= pd.to_numeric(dr["timestamp"], errors="coerce")
dr = dr.dropna(subset=["userId", "movieId", "rating", "timestamp"])

#garder la note la plus recente pour chaque (user, movie)
dr = dr.sort_values("timestamp")
dr = dr.drop_duplicates(subset=["userId", "movieId"], keep="last")  # garder la dernière
#creer le label binaire 
dr["label"]=(dr["rating"]>=THRESHOLD).astype(int)
print("evaluation nettoyer (user/film ,dernier note garde )")
print(dr.head(5))


#partie 2 entrainement de model il va lui plaire ou pas 

#encodage
enc = OneHotEncoder(handle_unknown="ignore",dtype=float)
x= enc.fit_transform(dr[["userId","movieId"]])
y=dr["label"].to_numpy()
#separation train/test 
x_train , x_test , y_train , y_test=train_test_split(x, y,test_size=0.1, random_state=42, stratify=y) 

#entainement 
clf = LogisticRegression(solver="liblinear", max_iter=200, class_weight="balanced")
clf.fit(x_train, y_train)

#evaluation
probs = clf.predict_proba(x_test)[:, 1]
preds = (probs >= 0.5).astype(int)
acc = accuracy_score(y_test, preds)
prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
auc = roc_auc_score(y_test, probs)

print("\n🎯 Résultats du modèle :")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"AUC       : {auc:.4f}")



#partie recommendation 
#film deja vu par chaque utilisateur 
def movies_seen (dr):
   """
   userId -> set(movieId déjà notés)
   """
   seen={}
   for _,row in dr.iterrows():
      user =int(row["userId"])
      movie=int(row["movieId"])
      if user not in seen :
         seen[user]=set()
      seen[user].add(movie)
   return seen  

seen_dict = movies_seen(dr)
print("Nombre d'utilisateurs :", len(seen_dict))
print("Exemple de contenu :")

# afficher les 3 premiers utilisateurs et quelques films vus
for user_id, movies_seen in list(seen_dict.items())[:3]:
    print(f"Utilisateur {user_id} a noté {len(movies_seen)} films : {list(movies_seen)[:5]} ...")


def recommend_for_user(user_id, clf,enc, dr, df, seen_dict,topk=10,min_prob=0.6):
   """
   Recomande les films les plus probables a aimer pour un utilisateur.
   """
   user_id=int(user_id)

   #verifier que l'utilisateur existe
   if user_id not in dr["userId"].unique():
      print ("utilisateur not found")
      return None
   #récupérer les films deja notés
   seen_movies = seen_dict.get(user_id, set())

   #recuperer tous les films disponibles
   all_movies = movies["movieId"].dropna().astype(int).unique()
   #candidats = films non vus 
   candidates = [m for m in all_movies if m not in seen_movies ]
   if not candidates :
      print("oh my god il a tout vus ")
      return None
   

 # IMPORTANT : convertir en array NumPy
   candidates = np.array(candidates, dtype=int)
   #dataframe de prediction (userId movieId)
   cand_df =pd.DataFrame({
       "userId":[user_id]  *len(candidates),
        "movieId":candidates 
   })

   #transformer via onehotEncoder 
   x_cand = enc.transform(cand_df[["userId","movieId"]])
   #probabilites d'aimer 
   probs = clf.predict_proba(x_cand)[:,1]
   #filtrer par probabilite minimale   
   mask =probs >=float(min_prob) 
   if not np.any(mask):
      print("aucun film avec probs>= min prob" )
      return None
   
   probs = probs[mask]
   canditates = np.array(candidates)[mask]

  #tri et slection de top k
   order=np.argsort(-probs)[:topk]
   top_ids=candidates[order]
   top_probs=probs[order]
 #joindre les titre 
   recs=pd.DataFrame({
      "movieId":top_ids,
      "probability": top_probs
   }).merge(movies, on ="movieId",how="left")
   return recs
 # ===== TEST =====
user_test = 2  # ou un autre userId présent dans ratings_small.csv
recs = recommend_for_user(user_test, clf, enc, dr, movies, seen_dict, topk=10, min_prob=0.6)

if recs is not None:
    print(f"\n🎬 Top 10 recommandations pour l'utilisateur {user_test} :")
    print(recs[["title", "probability"]])

    