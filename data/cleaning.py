import pandas as pd
from sklearn.preprocessing import StandardScaler

# Charger le dataset
df = pd.read_csv('mon_fichier.csv')

# 1. Afficher les premières lignes et infos du dataset
print(df.head())
print(df.info())

# 2. Supprimer les lignes avec des valeurs manquantes
df.dropna(inplace=True)

# 3. Remplacer des placeholders spécifiques (ex: "9999-99-99")
df.replace("9999-99-99", pd.NA, inplace=True)

# 4. Convertir les colonnes en types appropriés
df['date'] = pd.to_datetime(df['date'], errors='coerce')  # Convertir une colonne en format date
df['cases'] = pd.to_numeric(df['cases'], errors='coerce')  # Convertir une colonne en numérique

# 5. Supprimer les doublons
df.drop_duplicates(inplace=True)

# 6. Normaliser les données (si nécessaire)

scaler = StandardScaler()
df[['cases', 'deaths']] = scaler.fit_transform(df[['cases', 'deaths']])

# 7. Sauvegarder le dataset nettoyé
df.to_csv('mon_fichier_nettoye.csv', index=False)
