"""
Step 2: Modelli Inferenziali, Forme Funzionali e Multicollinearita'
Dataset: Medical Cost Personal Datasets (insurance.csv)
"""

import itertools
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)
OUT = "output"

#CARICAMENTO E PRE-PROCESSING
df = pd.read_csv("insurance.csv").drop_duplicates().reset_index(drop=True)
df_enc = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True, dtype=int)

REGRESSORS = ["age", "bmi", "children", "sex_male", "smoker_yes",
              "region_northwest", "region_southeast", "region_southwest"]
y_level = df_enc["charges"]
X_level = df_enc[REGRESSORS]

print("=" * 78)
print("2.1 MODELLO OLS COMPLETO (lin-lin) - 'modello completo'")
print("=" * 78)
X_full = sm.add_constant(X_level)
model_full = sm.OLS(y_level, X_full).fit()
print(model_full.summary())


#CONFRONTO FORME FUNZIONALI: lin-lin, log-log, log-lin, lin-log
#NOTA METODOLOGICA:
#  -'children' e' un conteggio con valori pari a 0 e le dummy sono 0/1:
#   il logaritmo non e' definito/e' privo di senso per queste
#   variabili, quindi in tutte le specificazioni "log" restano in livelli.
#  -Solo age e bmi (sempre > 0) vengono log-trasformate quando richiesto
#   dalla forma funzionale.
df_log = df_enc.copy()
df_log["log_charges"] = np.log(df_enc["charges"])
df_log["log_age"] = np.log(df_enc["age"])
df_log["log_bmi"] = np.log(df_enc["bmi"])

OTHER = ["children", "sex_male", "smoker_yes",
         "region_northwest", "region_southeast", "region_southwest"]

specs = {
    "lin-lin": dict(y="charges", X=["age", "bmi"] + OTHER),
    "log-log": dict(y="log_charges", X=["log_age", "log_bmi"] + OTHER),
    "log-lin": dict(y="log_charges", X=["age", "bmi"] + OTHER),
    "lin-log": dict(y="charges", X=["log_age", "log_bmi"] + OTHER),
}

def duan_smearing_predict(res, X_design):
    """Ritrasforma previsioni in log-scala alla scala originale
    con lo smearing estimator di Duan (1983), non distorto anche
    se i residui non sono normali (correzione della disuguaglianza
    di Jensen su E[exp(u)] != exp(E[u]))."""
    resid = res.resid
    smear = np.mean(np.exp(resid))
    return np.exp(res.predict(X_design)) * smear

from statsmodels.stats.stattools import jarque_bera

results_forms = {}
rows = []
for name, spec in specs.items():
    Xd = sm.add_constant(df_log[spec["X"]])
    yd = df_log[spec["y"]]
    res = sm.OLS(yd, Xd).fit()
    results_forms[name] = res

    is_log_y = spec["y"] == "log_charges"
    if is_log_y:
        charges_hat = duan_smearing_predict(res, Xd)
    else:
        charges_hat = res.fittedvalues

    resid_orig = df_enc["charges"] - charges_hat
    sse_orig = np.sum(resid_orig ** 2)
    sst_orig = np.sum((df_enc["charges"] - df_enc["charges"].mean()) ** 2)
    r2_orig = 1 - sse_orig / sst_orig
    rmse_orig = np.sqrt(np.mean(resid_orig ** 2))

    jb_stat, jb_p, skew_r, kurt_r = jarque_bera(res.resid)

    rows.append({
        "Forma": name,
        "Y": spec["y"],
        "R2 (scala modello)": res.rsquared,
        "R2 adj (scala modello)": res.rsquared_adj,
        "AIC": res.aic,
        "BIC": res.bic,
        "Skew residui": skew_r,
        "Kurt residui": kurt_r,
        "JB p-value": jb_p,
        "R2 (scala $, ritrasformato)": r2_orig,
        "RMSE (scala $, ritrasformato)": rmse_orig,
    })

comparison = pd.DataFrame(rows).set_index("Forma")
print("\n" + "=" * 78)
print("2.2 CONFRONTO FORME FUNZIONALI")
print("=" * 78)
print(comparison.round(4).to_string())
print("""
Note di lettura:
 - R2/R2-adj/AIC/BIC "in scala modello" NON sono confrontabili tra modelli
   con Y diversa (charges vs log(charges)) perche' la devianza totale da
   spiegare cambia; sono confrontabili solo entro la stessa Y (lin-lin vs
   lin-log; log-log vs log-lin).
 - Skew/Kurt/JB dei residui indicano quanto la forma funzionale "mitiga le
   asimmetrie": piu' skew e kurtosi si avvicinano a (0, 3) e piu' il
   JB p-value e' alto, piu' i residui sono vicini alla normalita'
   (assunzione chiave per la validita' dei t-test/F-test in campioni finiti).
 - R2/RMSE "ritrasformati in $" sono riportati per trasparenza predittiva,
   ma NON vanno usati per scegliere la forma funzionale: l'OLS lin-lin
   minimizza esattamente la SSE in dollari per costruzione, quindi
   vincerebbe quasi sempre questo confronto anche se i suoi residui sono
   fortemente asimmetrici ed eteroschedastici (vedi Step 3).
""")

# Criterio di scelta gerarchico (in due fasi, perche' i modelli con Y
# diversa non sono confrontabili con un unico numero):
#
# Fase 1 - confronto ENTRO lo stesso scale di Y (valido statisticamente):
#   - gruppo Y=charges:      lin-lin domina lin-log su R2_adj, AIC, BIC
#                             E su skew/kurtosi dei residui -> vince lin-lin.
#   - gruppo Y=log(charges): log-log vince per R2_adj/AIC/BIC (di poco),
#                             ma log-lin ha residui meno asimmetrici
#                             -> confronto misto, nessun dominio netto.
#
# Fase 2 - confronto tra i vincitori dei due gruppi (lin-lin vs il gruppo
# log): qui R2/AIC/BIC non sono comparabili (Y su scale diverse), quindi
# la scelta si basa su:
#   a) lin-lin ha la skewness e la kurtosi dei residui PIU' BASSE tra
#      tutte e 4 le forme (Step 1 aveva gia' mostrato che nemmeno
#      log(charges) elimina del tutto la bimodalita' generata da smoker);
#   b) il grafico age vs charges nello Step 1 mostra una relazione bene
#      approssimata da rette parallele (non da curve concave), a sostegno
#      di una specificazione in livelli per age piuttosto che in log;
#   c) l'evidenza decisiva sulla necessita' della trasformazione log(Y)
#      e' l'eteroschedasticita' dei residui, che viene testata
#      FORMALMENTE (Breusch-Pagan/White) nello Step 3: se il modello
#      lin-lin risultera' eteroschedastico, lo Step 3 applichera' proprio
#      la trasformazione logaritmica come rimedio, come previsto dalla
#      traccia stessa ("se non gia' fatto nello step 2").
#
# Per queste ragioni la forma scelta in questo step e' 'lin-lin': e' la
# piu' parsimoniosa, ha il miglior fit entro il proprio gruppo e i residui
# meno asimmetrici; l'eventuale passaggio a log(charges) viene demandato,
# come da consegna, alla diagnosi formale di eteroschedasticita' (Step 3).
print("Fase 1 - vincitori entro lo stesso scale di Y:")
print("  Y=charges      -> lin-lin (domina lin-log su R2_adj, AIC, BIC, skew)")
print("  Y=log(charges) -> log-log per R2_adj/AIC/BIC; log-lin per skew/kurtosi (esito misto)")
print("\nFase 2 - confronto cross-scale (lin-lin vs gruppo log):")
print(comparison[["R2 adj (scala modello)", "Skew residui", "Kurt residui"]].round(4).to_string())
best_form = "lin-lin"
print(f"\n-> Forma funzionale scelta per lo Step 2: '{best_form}' "
      "(parsimonia + miglior fit nel proprio gruppo + residui meno asimmetrici; "
      "l'eventuale log(Y) come rimedio e' demandata al test formale di eteroschedasticita' dello Step 3)")

#Grafico comparativo
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
comparison["R2 (scala $, ritrasformato)"].plot(kind="bar", ax=axes[0], color="#4C72B0")
axes[0].set_title("R2 ritrasformato in scala $ per forma funzionale")
axes[0].set_ylabel("R2")
axes[0].tick_params(axis="x", rotation=0)
comparison["RMSE (scala $, ritrasformato)"].plot(kind="bar", ax=axes[1], color="#C44E52")
axes[1].set_title("RMSE ritrasformato in scala $ per forma funzionale")
axes[1].set_ylabel("RMSE ($)")
axes[1].tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT}/06_functional_forms_comparison.png", dpi=150)
plt.close()
print(f"[Figura salvata] {OUT}/06_functional_forms_comparison.png")

#SELEZIONE DEL MODELLO MIGLIORE SULLA FORMA FUNZIONALE VINCENTE
best_spec = specs[best_form]
y_best_name, X_best_full = best_spec["y"], best_spec["X"]
y_best = df_log[y_best_name]

print("\n" + "=" * 78)
print(f"2.3 MODELLO COMPLETO SULLA FORMA '{best_form}' (base di partenza per la selezione)")
print("=" * 78)
res_start = sm.OLS(y_best, sm.add_constant(df_log[X_best_full])).fit()
print(res_start.summary())

#Backward elimination per significativita' (p-value > 0.05)
print("\n" + "-" * 78)
print("2.3a BACKWARD ELIMINATION (soglia p-value = 0.05)")
print("-" * 78)
cur_vars = X_best_full.copy()
step = 0
while True:
    Xd = sm.add_constant(df_log[cur_vars])
    res = sm.OLS(y_best, Xd).fit()
    pvals = res.pvalues.drop("const")
    worst_p = pvals.max()
    if worst_p <= 0.05 or len(cur_vars) == 1:
        break
    worst_var = pvals.idxmax()
    step += 1
    print(f"  step {step}: rimuovo '{worst_var}' (p-value={worst_p:.4f})")
    cur_vars.remove(worst_var)
model_backward = res
vars_backward = cur_vars
print(f"\nRegressori finali (backward elimination): {vars_backward}")

# Best Subset Selection (criterio BIC, ricerca esaustiva)
print("\n" + "-" * 78)
print("2.3b BEST SUBSET SELECTION (ricerca esaustiva, criterio BIC)")
print("-" * 78)
subset_results = []
for k in range(1, len(X_best_full) + 1):
    for combo in itertools.combinations(X_best_full, k):
        Xd = sm.add_constant(df_log[list(combo)])
        res = sm.OLS(y_best, Xd).fit()
        subset_results.append({
            "n_regressori": k,
            "regressori": combo,
            "R2_adj": res.rsquared_adj,
            "AIC": res.aic,
            "BIC": res.bic,
        })
subset_df = pd.DataFrame(subset_results)
print(f"Modelli valutati: {len(subset_df)}  (2^{len(X_best_full)} - 1 combinazioni non vuote)")

best_by_bic = subset_df.loc[subset_df["BIC"].idxmin()]
best_by_aic = subset_df.loc[subset_df["AIC"].idxmin()]
best_by_r2adj = subset_df.loc[subset_df["R2_adj"].idxmax()]

print("\nMiglior modello per BIC (criterio piu' parsimonioso):")
print(f"  regressori: {best_by_bic['regressori']}")
print(f"  BIC={best_by_bic['BIC']:.2f}  R2_adj={best_by_bic['R2_adj']:.4f}")

print("\nMiglior modello per AIC:")
print(f"  regressori: {best_by_aic['regressori']}")
print(f"  AIC={best_by_aic['AIC']:.2f}  R2_adj={best_by_aic['R2_adj']:.4f}")

print("\nMiglior modello per R2 corretto:")
print(f"  regressori: {best_by_r2adj['regressori']}")
print(f"  R2_adj={best_by_r2adj['R2_adj']:.4f}")

vars_subset_bic = list(best_by_bic["regressori"])

print(f"\nBackward elimination -> {sorted(vars_backward)}")
print(f"Best subset (BIC)    -> {sorted(vars_subset_bic)}")
agree = set(vars_backward) == set(vars_subset_bic)
print(f"I due criteri concordano: {agree}")

#Il modello finale e' quello indicato dal criterio BIC (piu' rigoroso,
#penalizza maggiormente la complessita' e riduce il rischio di overfitting)
FINAL_VARS = vars_subset_bic if not agree else vars_backward

#MODELLO FINALE: ANALISI DELL'OUTPUT
print("\n" + "=" * 78)
print(f"2.4 MODELLO FINALE  ({best_form}, regressori selezionati)")
print("=" * 78)
X_final = sm.add_constant(df_log[FINAL_VARS])
model_final = sm.OLS(y_best, X_final).fit()
print(model_final.summary())

print("\nInterpretazione sintetica:")
print(f" - R2            = {model_final.rsquared:.4f}  "
      f"-> il modello spiega il {model_final.rsquared*100:.1f}% della varianza di {y_best_name}")
print(f" - R2 corretto   = {model_final.rsquared_adj:.4f}  "
      f"(penalizza il numero di regressori: {len(FINAL_VARS)})")
print(f" - F-statistic   = {model_final.fvalue:.2f}  (p-value = {model_final.f_pvalue:.2e})")
print("   -> H0: tutti i coefficienti (tranne l'intercetta) sono nulli. "
      "Il p-value <<0.05 porta a rigettare H0: il modello nel complesso e' "
      "significativo.")
print(" - t-test sui singoli regressori (H0: beta_j = 0):")
tt = pd.DataFrame({
    "coef": model_final.params,
    "std err": model_final.bse,
    "t": model_final.tvalues,
    "P>|t|": model_final.pvalues,
    "IC 2.5%": model_final.conf_int()[0],
    "IC 97.5%": model_final.conf_int()[1],
})
print(tt.round(4).to_string())

if y_best_name == "log_charges":
    print("\nInterpretazione dei coefficienti (dipendente in log -> semi-elasticita'):")
    for v in FINAL_VARS:
        b = model_final.params[v]
        if v in ("age", "bmi", "children"):
            pct = (np.exp(b) - 1) * 100
            print(f"  {v:20s}: +1 unita' -> charges varia del {pct:+.2f}% (a parita' di altre covariate)")
        else:
            pct = (np.exp(b) - 1) * 100
            print(f"  {v:20s}: passare da 0 a 1 -> charges varia del {pct:+.2f}%")

#MULTICOLLINEARITA': VIF E CONDITION NUMBER
print("\n" + "=" * 78)
print("2.5 MULTICOLLINEARITA': VIF e CONDITION NUMBER")
print("=" * 78)

vif_data = pd.DataFrame()
vif_data["regressore"] = X_final.columns
vif_data["VIF"] = [variance_inflation_factor(X_final.values, i) for i in range(X_final.shape[1])]
print("\nVariance Inflation Factor (per il modello finale):")
print(vif_data.round(3).to_string(index=False))
print("""
Regola pratica: VIF < 5 assenza di problemi; 5 <= VIF < 10 attenzione;
VIF >= 10 forte multicollinearita'. Il VIF della costante non e' interpretabile.
""")

#Condition number "k" alla Belsley-Kuh-Welsch: le colonne di X (inclusa
#la costante) sono scalate a norma unitaria (NON centrate, per non
#mascherare eventuale collinearita' che coinvolge l'intercetta), poi
#si calcola il rapporto tra massimo e minimo valore singolare.
X_arr = X_final.values.astype(float)
col_norms = np.linalg.norm(X_arr, axis=0)
X_scaled = X_arr / col_norms
sing_vals = np.linalg.svd(X_scaled, compute_uv=False)
k_condition = sing_vals.max() / sing_vals.min()

print(f"Condition Number k (Belsley-Kuh-Welsch) = {k_condition:.2f}")
print(f"Condition Number (statsmodels, cross-check) = {model_final.condition_number:.2f}")
print("""
Regola pratica: k < 10  -> nessun problema
                10 <= k < 30 -> multicollinearita' moderata
                k >= 30 -> multicollinearita' severa
""")

#SALVATAGGIO OGGETTI PER GLI STEP SUCCESSIVI
import pickle
with open(f"{OUT}/step2_final_model.pkl", "wb") as f:
    pickle.dump({
        "best_form": best_form,
        "final_vars": FINAL_VARS,
        "y_name": y_best_name,
        "model_full_linlin": model_full,
        "model_final": model_final,
    }, f)
print(f"\n[Oggetti salvati] {OUT}/step2_final_model.pkl")

print("\n" + "=" * 78)
print("STEP 2 COMPLETATO")
print("=" * 78)
