"""
Step 3: Eteroschedasticita' e Analisi dei Residui
Dataset: Medical Cost Personal Datasets (insurance.csv)
Modello di partenza: modello finale dello Step 2 (lin-lin):
    charges = b0 + b1*age + b2*bmi + b3*children + b4*smoker_yes
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 140)
OUT = "output"

# -----------------------------------------------------------------------
# 0. CARICAMENTO E MODELLO FINALE (dallo Step 2)
# -----------------------------------------------------------------------
df = pd.read_csv("insurance.csv").drop_duplicates().reset_index(drop=True)
df_enc = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True, dtype=int)

FINAL_VARS = ["age", "bmi", "children", "smoker_yes"]
y = df_enc["charges"]
X = sm.add_constant(df_enc[FINAL_VARS])

model = sm.OLS(y, X).fit()
print("=" * 78)
print("3.0 MODELLO FINALE (Step 2) - richiamo")
print("=" * 78)
print(model.summary())

fitted = model.fittedvalues
resid = model.resid
resid_std = model.get_influence().resid_studentized_internal  # residui studentizzati

# -----------------------------------------------------------------------
# 1. GRAFICI: RESIDUI vs FITTED, ISTOGRAMMA RESIDUI + NORMALE
# -----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sc = axes[0].scatter(fitted, resid, c=df_enc["smoker_yes"], cmap="coolwarm",
                      alpha=0.6, s=20, edgecolor="none")
axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
axes[0].set_xlabel("Valori stimati (fitted values)")
axes[0].set_ylabel("Residui")
axes[0].set_title("Residui vs Fitted values")
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8)
           for c in ["#4C72B0", "#C44E52"]]
axes[0].legend(handles, ["non fumatore", "fumatore"], title="smoker", loc="upper left")

mu, sigma = resid.mean(), resid.std()
sns.histplot(resid, bins=40, stat="density", ax=axes[1], color="#4C72B0", alpha=0.6, label="Residui")
xs = np.linspace(resid.min(), resid.max(), 300)
axes[1].plot(xs, stats.norm.pdf(xs, mu, sigma), color="red", linewidth=2,
             label=f"Normale(mu={mu:,.0f}, sigma={sigma:,.0f})")
axes[1].set_xlabel("Residui")
axes[1].set_title("Istogramma dei residui vs densita' normale teorica")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUT}/07_residuals_diagnostics.png", dpi=150)
plt.close()
print(f"\n[Figura salvata] {OUT}/07_residuals_diagnostics.png")

# Q-Q plot (complemento standard alla diagnosi di normalita')
fig = sm.qqplot(resid_std, line="45", fit=True)
fig.set_size_inches(5.5, 5.5)
plt.title("Q-Q plot dei residui studentizzati")
plt.tight_layout()
plt.savefig(f"{OUT}/08_qqplot_residuals.png", dpi=150)
plt.close()
print(f"[Figura salvata] {OUT}/08_qqplot_residuals.png")

# -----------------------------------------------------------------------
# 2. TEST DI ETEROSCHEDASTICITA': BREUSCH-PAGAN e WHITE
# -----------------------------------------------------------------------
def run_bp_white(resid_, exog_, label):
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(resid_, exog_)
    wh_lm, wh_lm_p, wh_f, wh_f_p = het_white(resid_, exog_)
    print(f"\n--- {label} ---")
    print(f"Breusch-Pagan : LM={bp_lm:9.3f}  p-value={bp_lm_p:.4e}   "
          f"F={bp_f:8.3f}  p-value(F)={bp_f_p:.4e}")
    print(f"White         : LM={wh_lm:9.3f}  p-value={wh_lm_p:.4e}   "
          f"F={wh_f:8.3f}  p-value(F)={wh_f_p:.4e}")
    concl = "ETEROSCHEDASTICITA' rilevata (rigetto H0 di omoschedasticita')" \
            if min(bp_lm_p, wh_lm_p) < 0.05 else "Omoschedasticita' non rigettata"
    print(f"-> {concl} (alpha=0.05)")
    return dict(label=label, bp_lm=bp_lm, bp_p=bp_lm_p, white_lm=wh_lm, white_p=wh_lm_p)

print("\n" + "=" * 78)
print("3.1 TEST DI ETEROSCHEDASTICITA' SUL MODELLO LIN-LIN (Step 2)")
print("=" * 78)
print("H0: Var(u_i | X_i) = sigma^2 costante (omoschedasticita')")
print("H1: la varianza dell'errore dipende da X (eteroschedasticita')")
res_linlin = run_bp_white(resid, X, "Modello lin-lin (charges ~ age+bmi+children+smoker_yes)")

# -----------------------------------------------------------------------
# 3. RIMEDIO 1: TRASFORMAZIONE LOGARITMICA DI Y + RIPETIZIONE DEL TEST
# -----------------------------------------------------------------------
print("\n" + "=" * 78)
print("3.2 RIMEDIO 1: TRASFORMAZIONE LOGARITMICA log(charges) E RIPETIZIONE TEST")
print("=" * 78)
y_log = np.log(df_enc["charges"])
model_log = sm.OLS(y_log, X).fit()
print(model_log.summary())
res_loglin = run_bp_white(model_log.resid, X, "Modello log-lin (log(charges) ~ age+bmi+children+smoker_yes)")

# Grafico residui vs fitted anche per il modello log-lin (per confronto visivo)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(fitted, resid, alpha=0.5, s=18, color="#4C72B0")
axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
axes[0].set_title("lin-lin: Residui vs Fitted (charges, $)")
axes[0].set_xlabel("Fitted (charges $)")
axes[0].set_ylabel("Residui ($)")

axes[1].scatter(model_log.fittedvalues, model_log.resid, alpha=0.5, s=18, color="#55A868")
axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
axes[1].set_title("log-lin: Residui vs Fitted (log charges)")
axes[1].set_xlabel("Fitted (log charges)")
axes[1].set_ylabel("Residui (log scale)")
plt.tight_layout()
plt.savefig(f"{OUT}/09_residuals_loglin_vs_linlin.png", dpi=150)
plt.close()
print(f"\n[Figura salvata] {OUT}/09_residuals_loglin_vs_linlin.png")

# -----------------------------------------------------------------------
# 4. RIMEDIO 2: ERRORI STANDARD ROBUSTI (HC0, HC1, HC3)
# -----------------------------------------------------------------------
print("\n" + "=" * 78)
print("3.3 RIMEDIO 2: STIMA CON ERRORI STANDARD ROBUSTI (HC) - modello lin-lin")
print("=" * 78)
model_hc0 = sm.OLS(y, X).fit(cov_type="HC0")
model_hc1 = sm.OLS(y, X).fit(cov_type="HC1")
model_hc3 = sm.OLS(y, X).fit(cov_type="HC3")

se_compare = pd.DataFrame({
    "coef": model.params,
    "SE nonrobust (OLS classico)": model.bse,
    "SE HC0": model_hc0.bse,
    "SE HC1": model_hc1.bse,
    "SE HC3": model_hc3.bse,
})
se_compare["Delta% HC3 vs OLS"] = (se_compare["SE HC3"] / se_compare["SE nonrobust (OLS classico)"] - 1) * 100
print(se_compare.round(3).to_string())

print("\nConfronto p-value (t-test) OLS classico vs HC3:")
pval_compare = pd.DataFrame({
    "p-value OLS classico": model.pvalues,
    "p-value HC3": model_hc3.pvalues,
})
print(pval_compare.round(5).to_string())
print("""
Nota: i COEFFICIENTI stimati restano identici (l'OLS resta BLU anche in
presenza di eteroschedasticita', che invalida pero' la stima "classica"
della varianza). Cambiano SOLO gli standard error (e quindi t, p-value,
IC) usati per l'inferenza: HC3 e' generalmente il piu' conservativo ed
e' raccomandato in campioni medio-piccoli con possibili osservazioni
ad alta leva.
""")

print(model_hc3.summary())

# -----------------------------------------------------------------------
# 5. RIMEDIO 3: MODELLO LINEARE ROBUSTO (RLM)
# -----------------------------------------------------------------------
print("\n" + "=" * 78)
print("3.4 RIMEDIO 3: MODELLO LINEARE ROBUSTO (RLM, stimatore-M di Huber)")
print("=" * 78)
model_rlm = sm.RLM(y, X, M=sm.robust.norms.HuberT()).fit()
print(model_rlm.summary())

coef_compare = pd.DataFrame({
    "OLS (classico)": model.params,
    "OLS (SE HC3)": model_hc3.params,
    "RLM (Huber-T)": model_rlm.params,
})
print("\nConfronto coefficienti OLS vs RLM:")
print(coef_compare.round(3).to_string())

# Pesi assegnati dall'RLM: osservazioni con residuo grande vengono
# "downweighted" (peso < 1). Individuiamo le osservazioni piu' penalizzate.
weights = pd.Series(model_rlm.weights, index=df_enc.index, name="peso_RLM")
low_weight = weights.sort_values().head(10)
print("\n10 osservazioni con il peso RLM piu' basso (piu' penalizzate come outlier):")
print(pd.concat([df.loc[low_weight.index, ["age", "bmi", "children", "smoker", "charges"]],
                 low_weight], axis=1).to_string())

fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(weights, bins=40, ax=ax, color="#8172B2")
ax.axvline(1.0, color="red", linestyle="--", label="peso = 1 (nessuna penalizzazione)")
ax.set_title("Distribuzione dei pesi assegnati dal Modello Lineare Robusto (RLM)")
ax.set_xlabel("peso")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/10_rlm_weights.png", dpi=150)
plt.close()
print(f"\n[Figura salvata] {OUT}/10_rlm_weights.png")

# -----------------------------------------------------------------------
# 6. SINTESI COMPARATIVA DEI TEST
# -----------------------------------------------------------------------
print("\n" + "=" * 78)
print("3.5 SINTESI: TEST DI ETEROSCHEDASTICITA' PRIMA/DOPO IL RIMEDIO LOG")
print("=" * 78)
summary_tests = pd.DataFrame([res_linlin, res_loglin]).set_index("label")
print(summary_tests.round(4).to_string())

import pickle
with open(f"{OUT}/step3_models.pkl", "wb") as f:
    pickle.dump({
        "model_linlin": model,
        "model_loglin": model_log,
        "model_hc3": model_hc3,
        "model_rlm": model_rlm,
        "final_vars": FINAL_VARS,
    }, f)
print(f"\n[Oggetti salvati] {OUT}/step3_models.pkl")

print("\n" + "=" * 78)
print("STEP 3 COMPLETATO")
print("=" * 78)
