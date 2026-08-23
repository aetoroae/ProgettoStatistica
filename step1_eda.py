"""
Step 1: Analisi Preliminare (EDA) e Pre-processing
Dataset: Medical Cost Personal Datasets (insurance.csv)
Target: charges | Covariate: age, sex, bmi, children, smoker, region
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

OUT = "output"

#CARICAMENTO E ISPEZIONE DEI DATI
df = pd.read_csv("insurance.csv")

print("=" * 70)
print("1.1 STRUTTURA DEL DATASET")
print("=" * 70)
print(f"Dimensioni: {df.shape[0]} osservazioni x {df.shape[1]} variabili\n")
print(df.dtypes.to_string())
print("\nPrime 5 righe:")
print(df.head())

print("\n" + "=" * 70)
print("1.2 VALORI MANCANTI E DUPLICATI")
print("=" * 70)
print("Valori mancanti per colonna:")
print(df.isnull().sum().to_string())
n_dup = df.duplicated().sum()
print(f"\nRighe duplicate: {n_dup}")
if n_dup > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"-> Rimosse. Nuove dimensioni: {df.shape[0]} osservazioni")

print("\n" + "=" * 70)
print("1.3 STATISTICHE DESCRITTIVE - VARIABILI NUMERICHE")
print("=" * 70)
print(df.describe().T)

print("\n" + "=" * 70)
print("1.4 STATISTICHE DESCRITTIVE - VARIABILI CATEGORICHE")
print("=" * 70)
print(df.describe(include="object").T)
for col in ["sex", "smoker", "region"]:
    print(f"\nFrequenze '{col}':")
    print(df[col].value_counts())
    print(df[col].value_counts(normalize=True).round(3) * 100, "%")

print("\n" + "=" * 70)
print("1.5 ASIMMETRIA (SKEWNESS) E CURTOSI DI 'charges'")
print("=" * 70)
skew = df["charges"].skew()
kurt = df["charges"].kurt()
print(f"Skewness : {skew:.3f}  (>0 => coda a destra / asimmetria positiva)")
print(f"Kurtosis : {kurt:.3f}  (in eccesso rispetto alla normale, che ha 0)")

#DISTRIBUZIONE DELLA VARIABILE TARGET: charges
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.histplot(df["charges"], bins=40, kde=True, ax=axes[0], color="#4C72B0")
axes[0].axvline(df["charges"].mean(), color="red", linestyle="--", label=f"media={df['charges'].mean():,.0f}")
axes[0].axvline(df["charges"].median(), color="green", linestyle="--", label=f"mediana={df['charges'].median():,.0f}")
axes[0].set_title("Istogramma e densità di 'charges'")
axes[0].set_xlabel("charges ($)")
axes[0].legend()

sns.histplot(np.log(df["charges"]), bins=40, kde=True, ax=axes[1], color="#55A868")
axes[1].set_title("Istogramma e densità di 'log(charges)'")
axes[1].set_xlabel("log(charges)")

plt.tight_layout()
plt.savefig(f"{OUT}/01_charges_distribution.png", dpi=150)
plt.close()
print(f"\n[Figura salvata] {OUT}/01_charges_distribution.png")

#Distribuzione di charges per fattori chiave (utile a orientare lo Step 2)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
sns.boxplot(data=df, x="smoker", y="charges", hue="smoker", ax=axes[0], palette="Set2", legend=False)
axes[0].set_title("charges per stato di fumatore")
sns.scatterplot(data=df, x="age", y="charges", hue="smoker", ax=axes[1], alpha=0.6)
axes[1].set_title("charges vs age (colore = smoker)")
sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", ax=axes[2], alpha=0.6)
axes[2].set_title("charges vs bmi (colore = smoker)")
plt.tight_layout()
plt.savefig(f"{OUT}/02_charges_by_factors.png", dpi=150)
plt.close()
print(f"[Figura salvata] {OUT}/02_charges_by_factors.png")

#CODIFICA DELLE VARIABILI CATEGORICHE (dummy, evitando dummy trap)
#sex: 2 categorie -> 1 dummy (baseline = female)
#smoker: 2 categorie -> 1 dummy (baseline = no)
#region: 4 categorie -> 3 dummy (baseline = northeast, prima in ordine alfabetico)
df_enc = pd.get_dummies(
    df,
    columns=["sex", "smoker", "region"],
    drop_first=True,   # <-- evita la dummy variable trap (collinearità perfetta)
    dtype=int,
)

print("\n" + "=" * 70)
print("1.6 CODIFICA DUMMY (drop_first=True)")
print("=" * 70)
print("Colonne dopo l'encoding:")
print(list(df_enc.columns))
print("\nBaseline (categorie di riferimento, coeff. = 0 per costruzione):")
print(" - sex_male=0    -> baseline 'female'")
print(" - smoker_yes=0  -> baseline 'non fumatore'")
print(" - region_*=0    -> baseline 'northeast'")
print("\nAnteprima:")
print(df_enc.head())

df_enc.to_csv(f"{OUT}/insurance_encoded.csv", index=False)
print(f"\n[Dataset codificato salvato] {OUT}/insurance_encoded.csv")

#MATRICE DI CORRELAZIONE
corr = df_enc.corr(numeric_only=True)

print("\n" + "=" * 70)
print("1.7 MATRICE DI CORRELAZIONE (Pearson)")
print("=" * 70)
print(corr.round(3))

print("\nCorrelazioni con 'charges' ordinate per intensità:")
print(corr["charges"].drop("charges").sort_values(key=np.abs, ascending=False).round(3))

plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0,
    square=True, cbar_kws={"shrink": 0.8}, linewidths=0.5,
)
plt.title("Matrice di correlazione (Pearson)")
plt.tight_layout()
plt.savefig(f"{OUT}/03_correlation_heatmap.png", dpi=150)
plt.close()
print(f"\n[Figura salvata] {OUT}/03_correlation_heatmap.png")

#SCATTERPLOT MATRIX
num_vars = ["age", "bmi", "children", "charges"]
g = sns.pairplot(
    df[num_vars],
    kind="reg",
    diag_kind="kde",
    plot_kws={"line_kws": {"color": "red"}, "scatter_kws": {"alpha": 0.4, "s": 15}},
    corner=False,
)
g.fig.suptitle("Scatterplot matrix (variabili numeriche) con rette di regressione", y=1.02)
g.savefig(f"{OUT}/04_scatterplot_matrix.png", dpi=150)
plt.close()
print(f"[Figura salvata] {OUT}/04_scatterplot_matrix.png")

#Versione arricchita: colorata per smoker, per evidenziare l'interazione
g2 = sns.pairplot(
    df[num_vars + ["smoker"]],
    hue="smoker",
    diag_kind="kde",
    plot_kws={"alpha": 0.5, "s": 15},
    corner=False,
)
g2.fig.suptitle("Scatterplot matrix colorata per 'smoker'", y=1.02)
g2.savefig(f"{OUT}/05_scatterplot_matrix_smoker.png", dpi=150)
plt.close()
print(f"[Figura salvata] {OUT}/05_scatterplot_matrix_smoker.png")

print("\n" + "=" * 70)
print("STEP 1 COMPLETATO")
print("=" * 70)
