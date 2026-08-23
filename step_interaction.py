"""
Estensione: modello con interazione smoker_yes * bmi
Rifà le stime del modello finale aggiungendo il termine di interazione
per verificare se spiega parte dell'eteroschedasticità osservata.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 140)
OUT = "output"

df = pd.read_csv("insurance.csv").drop_duplicates().reset_index(drop=True)
df_enc = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True, dtype=int)

FINAL_VARS = ["age", "bmi", "children", "smoker_yes"]
y = df_enc["charges"]

#Modello SENZA interazione (richiamo, per confronto)
X0 = sm.add_constant(df_enc[FINAL_VARS])
model0 = sm.OLS(y, X0).fit()

#Modello CON interazione smoker_yes * bmi
df_enc["smoker_bmi"] = df_enc["smoker_yes"] * df_enc["bmi"]
INT_VARS = ["age", "bmi", "children", "smoker_yes", "smoker_bmi"]
X1 = sm.add_constant(df_enc[INT_VARS])
model1 = sm.OLS(y, X1).fit()

print("=" * 78)
print("MODELLO SENZA INTERAZIONE (richiamo)")
print("=" * 78)
print(model0.summary())

print("\n" + "=" * 78)
print("MODELLO CON INTERAZIONE smoker_yes * bmi")
print("=" * 78)
print(model1.summary())

print("\nConfronto R2:")
print(f"  Senza interazione: R2={model0.rsquared:.4f}  R2_adj={model0.rsquared_adj:.4f}")
print(f"  Con interazione:   R2={model1.rsquared:.4f}  R2_adj={model1.rsquared_adj:.4f}")

#Test F incrementale (nested F-test) per la significativita' congiunta
#dell'aggiunta del termine di interazione (qui e' un solo regressore,
#equivalente al suo t-test, riportato comunque per completezza)
from statsmodels.stats.anova import anova_lm
anova_res = anova_lm(model0, model1)
print("\nF-test incrementale (modello ridotto vs esteso):")
print(anova_res)

# VIF sul modello con interazione
vif_data = pd.DataFrame()
vif_data["regressore"] = X1.columns
vif_data["VIF"] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
print("\nVIF modello con interazione:")
print(vif_data.round(3).to_string(index=False))

# Test di eteroschedasticita' sul modello con interazione
def run_bp_white(resid_, exog_, label):
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(resid_, exog_)
    wh_lm, wh_lm_p, wh_f, wh_f_p = het_white(resid_, exog_)
    print(f"\n--- {label} ---")
    print(f"Breusch-Pagan : LM={bp_lm:9.3f}  p-value={bp_lm_p:.4e}   F={bp_f:8.3f}  p-value(F)={bp_f_p:.4e}")
    print(f"White         : LM={wh_lm:9.3f}  p-value={wh_lm_p:.4e}   F={wh_f:8.3f}  p-value(F)={wh_f_p:.4e}")
    return dict(label=label, bp_lm=bp_lm, bp_p=bp_lm_p, white_lm=wh_lm, white_p=wh_lm_p)

print("\n" + "=" * 78)
print("TEST DI ETEROSCHEDASTICITA': CONFRONTO SENZA/CON INTERAZIONE")
print("=" * 78)
res0 = run_bp_white(model0.resid, X0, "Senza interazione")
res1 = run_bp_white(model1.resid, X1, "Con interazione smoker_yes*bmi")

pct_change_bp = (res1["bp_lm"] - res0["bp_lm"]) / res0["bp_lm"] * 100
pct_change_white = (res1["white_lm"] - res0["white_lm"]) / res0["white_lm"] * 100
print(f"\nVariazione statistica LM Breusch-Pagan: {pct_change_bp:+.1f}%")
print(f"Variazione statistica LM White:         {pct_change_white:+.1f}%")

# Skew/kurtosi residui, per confronto con Step 2/3
from statsmodels.stats.stattools import jarque_bera
jb0 = jarque_bera(model0.resid)
jb1 = jarque_bera(model1.resid)
print(f"\nSkew/Kurt residui SENZA interazione: skew={jb0[2]:.3f} kurt={jb0[3]:.3f}")
print(f"Skew/Kurt residui CON interazione:   skew={jb1[2]:.3f} kurt={jb1[3]:.3f}")

# Grafici: Residui vs Fitted, confronto senza/con interazione
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, model, title in [(axes[0], model0, "Senza interazione"),
                          (axes[1], model1, "Con interazione smoker_yes$\\times$bmi")]:
    sc = ax.scatter(model.fittedvalues, model.resid, c=df_enc["smoker_yes"],
                     cmap="coolwarm", alpha=0.6, s=20, edgecolor="none")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Valori stimati (fitted values)")
    ax.set_ylabel("Residui")
    ax.set_title(title)
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8)
           for c in ["#4C72B0", "#C44E52"]]
axes[0].legend(handles, ["non fumatore", "fumatore"], title="smoker", loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/14_residuals_interaction_comparison.png", dpi=150)
plt.close()
print(f"\n[Figura salvata] {OUT}/14_residuals_interaction_comparison.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, model, title in [(axes[0], model0, "Senza interazione"),
                          (axes[1], model1, "Con interazione")]:
    mu, sigma = model.resid.mean(), model.resid.std()
    sns.histplot(model.resid, bins=40, stat="density", ax=ax, color="#4C72B0", alpha=0.6)
    xs = np.linspace(model.resid.min(), model.resid.max(), 300)
    ax.plot(xs, stats.norm.pdf(xs, mu, sigma), color="red", linewidth=2)
    ax.set_title(f"Istogramma residui - {title}")
    ax.set_xlabel("Residui")
plt.tight_layout()
plt.savefig(f"{OUT}/15_residuals_hist_interaction_comparison.png", dpi=150)
plt.close()
print(f"[Figura salvata] {OUT}/15_residuals_hist_interaction_comparison.png")

#Interpretazione dell'effetto marginale di bmi per fumatori/non fumatori
b_bmi = model1.params["bmi"]
b_int = model1.params["smoker_bmi"]
print(f"\nEffetto marginale di bmi:")
print(f"  Non fumatori: {b_bmi:.2f} $/punto di BMI")
print(f"  Fumatori:     {b_bmi + b_int:.2f} $/punto di BMI (={b_bmi:.2f} + {b_int:.2f})")

import pickle
with open(f"{OUT}/step_interaction_models.pkl", "wb") as f:
    pickle.dump({"model0": model0, "model1": model1, "res0": res0, "res1": res1}, f)

print("\nDONE")
