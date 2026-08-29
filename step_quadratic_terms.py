"""
Esplorazione: effetto dell'aggiunta di bmi^2 e age^2 al modello log-lin
finale (con e senza interazione smoker_yes*bmi), per valutare se
catturano la non linearita' residua osservata nei grafici dei residui.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.anova import anova_lm

pd.set_option("display.width", 140)

df = pd.read_csv("insurance.csv").drop_duplicates().reset_index(drop=True)
df_enc = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True, dtype=int)
df_enc["smoker_bmi"] = df_enc["smoker_yes"] * df_enc["bmi"]
df_enc["bmi2"] = df_enc["bmi"] ** 2
df_enc["age2"] = df_enc["age"] ** 2

y_log = np.log(df_enc["charges"])

def run_bp_white(res, label):
    bp_lm, bp_p, bp_f, bp_fp = het_breuschpagan(res.resid, res.model.exog)
    wh_lm, wh_p, wh_f, wh_fp = het_white(res.resid, res.model.exog)
    jb = jarque_bera(res.resid)
    print(f"  BP:  LM={bp_lm:8.2f}  p={bp_p:.3e}   White: LM={wh_lm:8.2f}  p={wh_p:.3e}   "
          f"skew={jb[2]:.3f} kurt={jb[3]:.3f}")
    return dict(label=label, bp_lm=bp_lm, bp_p=bp_p, wh_lm=wh_lm, wh_p=wh_p, skew=jb[2], kurt=jb[3])

specs = {
    "A: baseline (age,bmi,children,smoker)":
        ["age", "bmi", "children", "smoker_yes"],
    "B: + smoker*bmi (interazione)":
        ["age", "bmi", "children", "smoker_yes", "smoker_bmi"],
    "C: baseline + bmi^2":
        ["age", "bmi", "bmi2", "children", "smoker_yes"],
    "D: baseline + age^2":
        ["age", "age2", "bmi", "children", "smoker_yes"],
    "E: baseline + bmi^2 + age^2":
        ["age", "age2", "bmi", "bmi2", "children", "smoker_yes"],
    "F: interazione + bmi^2 + age^2":
        ["age", "age2", "bmi", "bmi2", "children", "smoker_yes", "smoker_bmi"],
}

results = {}
rows = []
print("=" * 100)
for name, vars_ in specs.items():
    X = sm.add_constant(df_enc[vars_])
    res = sm.OLS(y_log, X).fit()
    results[name] = res
    print(f"\n{name}")
    print(f"  R2={res.rsquared:.4f}  R2adj={res.rsquared_adj:.4f}  AIC={res.aic:.1f}  BIC={res.bic:.1f}")
    for v in vars_:
        if v in ("bmi2", "age2"):
            print(f"  {v:12s} coef={res.params[v]: .6f}  t={res.tvalues[v]: .2f}  p={res.pvalues[v]:.4f}")
    diag = run_bp_white(res, name)
    rows.append({"modello": name, "R2": res.rsquared, "R2adj": res.rsquared_adj,
                 "AIC": res.aic, "BIC": res.bic, **{k: v for k, v in diag.items() if k != "label"}})

print("\n" + "=" * 100)
print("TABELLA RIASSUNTIVA")
print("=" * 100)
summary = pd.DataFrame(rows).set_index("modello")
print(summary.round(4).to_string())

#Incremental F-tests vs baseline (A) and vs interaction (B)
print("\n" + "=" * 100)
print("F-TEST INCREMENTALI")
print("=" * 100)
print("\nC vs A (aggiunta bmi^2):")
print(anova_lm(results["A: baseline (age,bmi,children,smoker)"], results["C: baseline + bmi^2"]))
print("\nD vs A (aggiunta age^2):")
print(anova_lm(results["A: baseline (age,bmi,children,smoker)"], results["D: baseline + age^2"]))
print("\nE vs A (aggiunta bmi^2 + age^2):")
print(anova_lm(results["A: baseline (age,bmi,children,smoker)"], results["E: baseline + bmi^2 + age^2"]))
print("\nF vs B (aggiunta bmi^2 + age^2 al modello con interazione):")
print(anova_lm(results["B: + smoker*bmi (interazione)"], results["F: interazione + bmi^2 + age^2"]))

#VIF for model E and F (raw, uncentered) to show the expected inflation
print("\n" + "=" * 100)
print("VIF (non centrato) - modello E")
print("=" * 100)
X = sm.add_constant(df_enc[specs["E: baseline + bmi^2 + age^2"]])
vif = pd.DataFrame({"var": X.columns,
                     "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
print(vif.round(2).to_string(index=False))

print("\n" + "=" * 100)
print("VIF (non centrato) - modello F")
print("=" * 100)
X = sm.add_constant(df_enc[specs["F: interazione + bmi^2 + age^2"]])
vif = pd.DataFrame({"var": X.columns,
                     "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]})
print(vif.round(2).to_string(index=False))

#Centered version to show VIF drops (diagnostic-only, not for final model)
print("\n" + "=" * 100)
print("VIF (centrato su media) - modello F, solo per confronto diagnostico")
print("=" * 100)
df_c = df_enc.copy()
df_c["age_c"] = df_c["age"] - df_c["age"].mean()
df_c["bmi_c"] = df_c["bmi"] - df_c["bmi"].mean()
df_c["age_c2"] = df_c["age_c"] ** 2
df_c["bmi_c2"] = df_c["bmi_c"] ** 2
df_c["smoker_bmic"] = df_c["smoker_yes"] * df_c["bmi_c"]
Xc = sm.add_constant(df_c[["age_c", "age_c2", "bmi_c", "bmi_c2", "children", "smoker_yes", "smoker_bmic"]])
resc = sm.OLS(y_log, Xc).fit()
vif = pd.DataFrame({"var": Xc.columns,
                     "VIF": [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]})
print(vif.round(2).to_string(index=False))
print(f"\nR2 centrato={resc.rsquared:.4f} (deve coincidere con F: {results['F: interazione + bmi^2 + age^2'].rsquared:.4f})")
for v in ["age_c2", "bmi_c2"]:
    print(f"  {v:12s} coef={resc.params[v]: .6f}  t={resc.tvalues[v]: .2f}  p={resc.pvalues[v]:.4f}")

print("\nDONE")
