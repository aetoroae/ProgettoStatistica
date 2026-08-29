"""
Ricalcolo completo assumendo LOG-LIN come modello finale (invece di lin-lin):
- Best Subset Selection su log(charges) con tutti gli 8 regressori
- Modello finale log-lin (age, bmi, children, smoker_yes)
- VIF sul modello log-lin (dipende solo da X, invariato rispetto a lin-lin)
- Test BP/White sul modello log-lin finale
- Modello log-lin CON interazione smoker_yes*bmi (nuovo)
- Test BP/White sul modello log-lin+interazione
- RLM sul modello log-lin finale
- Grafici residui per il modello log-lin finale e per log-lin+interazione
"""

import itertools
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 140)
OUT = "output"

df = pd.read_csv("insurance.csv").drop_duplicates().reset_index(drop=True)
df_enc = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True, dtype=int)
df_enc["smoker_bmi"] = df_enc["smoker_yes"] * df_enc["bmi"]

ALL_VARS = ["age", "bmi", "children", "sex_male", "smoker_yes",
            "region_northwest", "region_southeast", "region_southwest"]
FINAL_VARS = ["age", "bmi", "children", "smoker_yes"]
y_log = np.log(df_enc["charges"])

#MODELLO LOG-LIN COMPLETO (8 regressori)
X_full = sm.add_constant(df_enc[ALL_VARS])
model_full = sm.OLS(y_log, X_full).fit()
print("=" * 78)
print("MODELLO LOG-LIN COMPLETO (8 regressori)")
print("=" * 78)
print(model_full.summary())

#BEST SUBSET SELECTION (BIC) su log(charges)
print("\n" + "=" * 78)
print("BEST SUBSET SELECTION SU LOG(CHARGES)")
print("=" * 78)
subset_results = []
for k in range(1, len(ALL_VARS) + 1):
    for combo in itertools.combinations(ALL_VARS, k):
        Xd = sm.add_constant(df_enc[list(combo)])
        res = sm.OLS(y_log, Xd).fit()
        subset_results.append({"n": k, "vars": combo, "R2_adj": res.rsquared_adj,
                                "AIC": res.aic, "BIC": res.bic})
subset_df = pd.DataFrame(subset_results)
best_bic = subset_df.loc[subset_df["BIC"].idxmin()]
print(f"Miglior modello per BIC: {best_bic['vars']}  BIC={best_bic['BIC']:.2f}  R2_adj={best_bic['R2_adj']:.4f}")
print(f"Coincide con {tuple(FINAL_VARS)}: {set(best_bic['vars']) == set(FINAL_VARS)}")

#MODELLO LOG-LIN FINALE (age, bmi, children, smoker_yes)
X0 = sm.add_constant(df_enc[FINAL_VARS])
model0 = sm.OLS(y_log, X0).fit()
print("\n" + "=" * 78)
print("MODELLO LOG-LIN FINALE (age, bmi, children, smoker_yes)")
print("=" * 78)
print(model0.summary())

vif0 = pd.DataFrame()
vif0["regressore"] = X0.columns
vif0["VIF"] = [variance_inflation_factor(X0.values, i) for i in range(X0.shape[1])]
print("\nVIF modello log-lin finale:")
print(vif0.round(3).to_string(index=False))

def run_bp_white(resid_, exog_, label):
    bp_lm, bp_lm_p, bp_f, bp_f_p = het_breuschpagan(resid_, exog_)
    wh_lm, wh_lm_p, wh_f, wh_f_p = het_white(resid_, exog_)
    print(f"\n--- {label} ---")
    print(f"Breusch-Pagan : LM={bp_lm:9.3f}  p-value={bp_lm_p:.4e}   F={bp_f:8.3f}  p-value(F)={bp_f_p:.4e}")
    print(f"White         : LM={wh_lm:9.3f}  p-value={wh_lm_p:.4e}   F={wh_f:8.3f}  p-value(F)={wh_f_p:.4e}")
    return dict(bp_lm=bp_lm, bp_p=bp_lm_p, white_lm=wh_lm, white_p=wh_lm_p)

res0 = run_bp_white(model0.resid, X0, "log-lin finale")
jb0 = jarque_bera(model0.resid)
print(f"Skew={jb0[2]:.3f}  Kurt={jb0[3]:.3f}")

# Semi-elasticita'
print("\nSemi-elasticita' (exp(beta)-1)*100:")
for v in FINAL_VARS:
    b = model0.params[v]
    print(f"  {v:12s}: beta={b:.4f}  semi-elasticita'={ (np.exp(b)-1)*100:+.2f}%")
#MODELLO LOG-LIN CON INTERAZIONE smoker_yes*bmi
INT_VARS = ["age", "bmi", "children", "smoker_yes", "smoker_bmi"]
X1 = sm.add_constant(df_enc[INT_VARS])
model1 = sm.OLS(y_log, X1).fit()
print("\n" + "=" * 78)
print("MODELLO LOG-LIN CON INTERAZIONE smoker_yes*bmi")
print("=" * 78)
print(model1.summary())

vif1 = pd.DataFrame()
vif1["regressore"] = X1.columns
vif1["VIF"] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
print("\nVIF modello log-lin con interazione:")
print(vif1.round(3).to_string(index=False))

res1 = run_bp_white(model1.resid, X1, "log-lin con interazione")
jb1 = jarque_bera(model1.resid)
print(f"Skew={jb1[2]:.3f}  Kurt={jb1[3]:.3f}")

pct_bp = (res1["bp_lm"] - res0["bp_lm"]) / res0["bp_lm"] * 100
pct_white = (res1["white_lm"] - res0["white_lm"]) / res0["white_lm"] * 100
print(f"\nVariazione LM Breusch-Pagan: {pct_bp:+.1f}%")
print(f"Variazione LM White:         {pct_white:+.1f}%")

from statsmodels.stats.anova import anova_lm
anova_res = anova_lm(model0, model1)
print("\nF-test incrementale (log-lin ristretto vs esteso):")
print(anova_res)

#Semi-elasticita' modello con interazione: effetto di bmi per fumatori/non fumatori
b_bmi = model1.params["bmi"]
b_int = model1.params["smoker_bmi"]
b_smoker = model1.params["smoker_yes"]
print(f"\nSemi-elasticita' di bmi:")
print(f"  Non fumatori: {(np.exp(b_bmi)-1)*100:+.3f}% per punto di BMI (beta={b_bmi:.5f}, p={model1.pvalues['bmi']:.4f})")
print(f"  Fumatori:     {(np.exp(b_bmi+b_int)-1)*100:+.2f}% per punto di BMI (beta={b_bmi+b_int:.5f})")
print(f"Semi-elasticita' smoker_yes a bmi=0: {(np.exp(b_smoker)-1)*100:+.2f}%")
mean_bmi = df_enc["bmi"].mean()
print(f"Effetto smoker a bmi medio ({mean_bmi:.2f}): {(np.exp(b_smoker + b_int*mean_bmi)-1)*100:+.2f}%")

#RLM sul modello log-lin finale
model_rlm = sm.RLM(y_log, X0, M=sm.robust.norms.HuberT()).fit()
print("\n" + "=" * 78)
print("RLM SUL MODELLO LOG-LIN FINALE")
print("=" * 78)
print(model_rlm.summary())
coef_compare = pd.DataFrame({"OLS log-lin": model0.params, "RLM log-lin": model_rlm.params})
print(coef_compare.round(4).to_string())

weights = pd.Series(model_rlm.weights, index=df_enc.index)
low_weight = weights.sort_values().head(10)
print("\n10 osservazioni con peso RLM piu' basso:")
print(pd.concat([df.loc[low_weight.index, ["age", "bmi", "children", "smoker", "charges"]],
                 low_weight.rename("peso")], axis=1).to_string())

#GRAFICI
fitted0 = model0.fittedvalues
resid0 = model0.resid
resid0_std = model0.get_influence().resid_studentized_internal

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(fitted0, resid0, c=df_enc["smoker_yes"], cmap="coolwarm", alpha=0.6, s=20, edgecolor="none")
axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
axes[0].set_xlabel("Valori stimati (fitted, log-scala)")
axes[0].set_ylabel("Residui (log-scala)")
axes[0].set_title("Residui vs Fitted (modello log-lin finale)")
handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=8)
           for c in ["#4C72B0", "#C44E52"]]
axes[0].legend(handles, ["non fumatore", "fumatore"], title="smoker", loc="upper left")

mu, sigma = resid0.mean(), resid0.std()
sns.histplot(resid0, bins=40, stat="density", ax=axes[1], color="#4C72B0", alpha=0.6)
xs = np.linspace(resid0.min(), resid0.max(), 300)
axes[1].plot(xs, stats.norm.pdf(xs, mu, sigma), color="red", linewidth=2)
axes[1].set_xlabel("Residui")
axes[1].set_title("Istogramma residui (log-lin finale)")
plt.tight_layout()
plt.savefig(f"{OUT}/16_residuals_loglin_final.png", dpi=150)
plt.close()
print(f"\n[Figura salvata] {OUT}/16_residuals_loglin_final.png")

fig = sm.qqplot(resid0_std, line="45", fit=True)
fig.set_size_inches(5.5, 5.5)
plt.title("Q-Q plot residui (log-lin finale)")
plt.tight_layout()
plt.savefig(f"{OUT}/17_qqplot_loglin_final.png", dpi=150)
plt.close()
print(f"[Figura salvata] {OUT}/17_qqplot_loglin_final.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, model, title in [(axes[0], model0, "log-lin (senza interazione)"),
                          (axes[1], model1, "log-lin con interazione smoker_yes$\\times$bmi")]:
    ax.scatter(model.fittedvalues, model.resid, c=df_enc["smoker_yes"], cmap="coolwarm",
               alpha=0.6, s=20, edgecolor="none")
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Valori stimati (log-scala)")
    ax.set_ylabel("Residui (log-scala)")
    ax.set_title(title)
axes[0].legend(handles, ["non fumatore", "fumatore"], title="smoker", loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/18_residuals_loglin_interaction_comparison.png", dpi=150)
plt.close()
print(f"[Figura salvata] {OUT}/18_residuals_loglin_interaction_comparison.png")

fig, ax = plt.subplots(figsize=(7, 5))
sns.histplot(weights, bins=40, ax=ax, color="#8172B2")
ax.axvline(1.0, color="red", linestyle="--", label="peso = 1")
ax.set_title("Pesi RLM (modello log-lin finale)")
ax.set_xlabel("peso")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/19_rlm_weights_loglin.png", dpi=150)
plt.close()
print(f"[Figura salvata] {OUT}/19_rlm_weights_loglin.png")

print("\nDONE")
