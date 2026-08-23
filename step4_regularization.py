"""
Step 4: Modelli Predittivi (Regolarizzazione)
Dataset: Medical Cost Personal Datasets (insurance.csv)
Target: charges (livelli) - Regressori: tutte le 8 variabili codificate
(a differenza dello Step 2/3, qui NON si parte dal sottoinsieme
gia' selezionato: si lascia che Ridge/LASSO/Elastic Net operino la
selezione/riduzione in modo automatico sull'intero set di covariate).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 140)
OUT = "output"
RNG = 42

# -----------------------------------------------------------------------
# 0. CARICAMENTO, ENCODING, TRAIN/TEST SPLIT
# -----------------------------------------------------------------------
df = pd.read_csv("insurance.csv").drop_duplicates().reset_index(drop=True)
df_enc = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True, dtype=int)

ALL_VARS = ["age", "bmi", "children", "sex_male", "smoker_yes",
            "region_northwest", "region_southeast", "region_southwest"]
CONTINUOUS = ["age", "bmi", "children"]

X = df_enc[ALL_VARS].copy()
y = df_enc["charges"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RNG)
print("=" * 78)
print("4.0 TRAIN/TEST SPLIT E STANDARDIZZAZIONE")
print("=" * 78)
print(f"Train: {X_train.shape[0]} oss. | Test: {X_test.shape[0]} oss.")

# Standardizzazione SOLO dei regressori continui (age, bmi, children).
# Le dummy (0/1) restano invariate: sono gia' su scala comparabile e
# la loro varianza (p*(1-p)) ha un significato diretto come "quota di
# popolazione nella categoria", che si perderebbe standardizzandole.
scaler = StandardScaler()
X_train_s = X_train.copy()
X_test_s = X_test.copy()
X_train_s[CONTINUOUS] = scaler.fit_transform(X_train[CONTINUOUS])  # fit SOLO su train (no leakage)
X_test_s[CONTINUOUS] = scaler.transform(X_test[CONTINUOUS])

print("\nMedia/Std dei regressori continui dopo standardizzazione (train):")
print(X_train_s[CONTINUOUS].agg(["mean", "std"]).round(3))

# -----------------------------------------------------------------------
# 1. GRIGLIA LOGARITMICA DECRESCENTE PER LAMBDA
# -----------------------------------------------------------------------
alphas = np.logspace(4, -4, 100)  # da 10^4 a 10^-4, decrescente
print(f"\nGriglia lambda: {len(alphas)} valori, da {alphas[0]:.1e} a {alphas[-1]:.1e}")

kf = KFold(n_splits=10, shuffle=True, random_state=RNG)

def cv_mse_path(estimator_cls, alphas_grid, **kwargs):
    """CV MSE (10-fold) per ciascun valore di alpha nella griglia."""
    means, ses = [], []
    for a in alphas_grid:
        est = estimator_cls(alpha=a, **kwargs)
        scores = cross_val_score(est, X_train_s, y_train, cv=kf,
                                  scoring="neg_mean_squared_error", n_jobs=-1)
        mse_folds = -scores
        means.append(mse_folds.mean())
        ses.append(mse_folds.std(ddof=1) / np.sqrt(len(mse_folds)))
    return np.array(means), np.array(ses)

def coef_path(estimator_cls, alphas_grid, **kwargs):
    coefs = []
    for a in alphas_grid:
        est = estimator_cls(alpha=a, **kwargs)
        est.fit(X_train_s, y_train)
        coefs.append(est.coef_)
    return np.array(coefs)

def lambda_1se(alphas_grid, means, ses):
    """Regola dell'errore standard (1-SE rule, alla glmnet): il piu'
    grande lambda (= modello piu' parsimonioso) il cui CV-MSE resta
    entro 1 SE dal minimo."""
    i_min = np.argmin(means)
    threshold = means[i_min] + ses[i_min]
    # alphas_grid e' decrescente: cerchiamo tra gli alpha >= alpha_min
    # (piu' regolarizzazione = piu' a sinistra nella griglia) quello piu'
    # grande che soddisfa la soglia
    candidates = [i for i in range(i_min + 1) if means[i] <= threshold]
    i_1se = min(candidates) if candidates else i_min
    return i_1se, i_min

MODELS = {
    "Ridge":          dict(cls=Ridge,       kwargs=dict()),
    "LASSO":          dict(cls=Lasso,       kwargs=dict(max_iter=50000)),
    "ElasticNet a=0.1": dict(cls=ElasticNet, kwargs=dict(l1_ratio=0.1, max_iter=50000)),
    "ElasticNet a=0.9": dict(cls=ElasticNet, kwargs=dict(l1_ratio=0.9, max_iter=50000)),
}

results = {}
print("\n" + "=" * 78)
print("4.1 10-FOLD CROSS-VALIDATION SULLA GRIGLIA DI LAMBDA")
print("=" * 78)
for name, spec in MODELS.items():
    means, ses = cv_mse_path(spec["cls"], alphas, **spec["kwargs"])
    coefs = coef_path(spec["cls"], alphas, **spec["kwargs"])
    i_1se, i_min = lambda_1se(alphas, means, ses)

    lam_min, lam_1se = alphas[i_min], alphas[i_1se]
    mse_min, mse_1se = means[i_min], means[i_1se]

    # Refit sul training set completo con lambda ottimale (lambda.min)
    best_est = spec["cls"](alpha=lam_min, **spec["kwargs"])
    best_est.fit(X_train_s, y_train)
    y_pred_test = best_est.predict(X_test_s)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    n_nonzero = np.sum(np.abs(best_est.coef_) > 1e-8)

    results[name] = dict(
        alphas=alphas, cv_means=means, cv_ses=ses, coefs=coefs,
        lam_min=lam_min, lam_1se=lam_1se, i_min=i_min, i_1se=i_1se,
        mse_min=mse_min, mse_1se=mse_1se,
        best_est=best_est, test_mse=test_mse, test_r2=test_r2, n_nonzero=n_nonzero,
    )
    print(f"\n--- {name} ---")
    print(f"  lambda.min = {lam_min:10.4f}   CV-MSE = {mse_min:12,.1f}  (RMSE={np.sqrt(mse_min):,.0f})")
    print(f"  lambda.1se = {lam_1se:10.4f}   CV-MSE = {mse_1se:12,.1f}  (modello piu' parsimonioso entro 1 SE dal minimo)")
    print(f"  Test MSE (lambda.min) = {test_mse:,.1f}   Test RMSE = {np.sqrt(test_mse):,.0f}   Test R2 = {test_r2:.4f}")
    print(f"  Regressori non nulli (|coef|>1e-8) a lambda.min: {n_nonzero}/{len(ALL_VARS)}")
    coef_series = pd.Series(best_est.coef_, index=ALL_VARS)
    print("  Coefficienti (lambda.min):")
    print("   " + coef_series.round(2).to_string().replace("\n", "\n   "))

# -----------------------------------------------------------------------
# 2. CONFRONTO CON OLS (baseline, lambda->0)
# -----------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
ols = LinearRegression().fit(X_train_s, y_train)
ols_test_mse = mean_squared_error(y_test, ols.predict(X_test_s))
ols_test_r2 = r2_score(y_test, ols.predict(X_test_s))
print("\n" + "=" * 78)
print("4.2 BASELINE OLS (senza regolarizzazione) sullo stesso train/test split")
print("=" * 78)
print(f"Test MSE = {ols_test_mse:,.1f}   Test RMSE = {np.sqrt(ols_test_mse):,.0f}   Test R2 = {ols_test_r2:.4f}")
print("Coefficienti OLS:")
print(pd.Series(ols.coef_, index=ALL_VARS).round(2).to_string())

# -----------------------------------------------------------------------
# 3. TRACE PLOT DEI COEFFICIENTI
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
palette = sns.color_palette("tab10", n_colors=len(ALL_VARS))

for ax, (name, res) in zip(axes.flat, results.items()):
    neg_log_lambda = -np.log10(res["alphas"])
    for j, var in enumerate(ALL_VARS):
        ax.plot(neg_log_lambda, res["coefs"][:, j], color=palette[j], label=var)
    ax.axvline(-np.log10(res["lam_min"]), color="red", linestyle="--", linewidth=1,
               label="lambda.min" if name == "Ridge" else None)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title(f"Trace plot - {name}")
    ax.set_xlabel(r"$-\log_{10}(\lambda)$")
    ax.set_ylabel("Coefficiente")

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f"{OUT}/11_trace_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[Figura salvata] {OUT}/11_trace_plots.png")

# -----------------------------------------------------------------------
# 4. CV MSE vs -log(lambda)
# -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, (name, res) in zip(axes.flat, results.items()):
    neg_log_lambda = -np.log10(res["alphas"])
    ax.errorbar(neg_log_lambda, res["cv_means"], yerr=res["cv_ses"],
                fmt="o", ms=3, elinewidth=0.5, capsize=0, alpha=0.5, color="#4C72B0")
    ax.axvline(-np.log10(res["lam_min"]), color="red", linestyle="--",
               label=f"lambda.min={res['lam_min']:.3g}")
    ax.axvline(-np.log10(res["lam_1se"]), color="green", linestyle="--",
               label=f"lambda.1se={res['lam_1se']:.3g}")
    ax.set_title(f"10-fold CV MSE - {name}")
    ax.set_xlabel(r"$-\log_{10}(\lambda)$")
    ax.set_ylabel("CV MSE")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/12_cv_mse_vs_loglambda.png", dpi=150)
plt.close()
print(f"[Figura salvata] {OUT}/12_cv_mse_vs_loglambda.png")

# -----------------------------------------------------------------------
# 5. TABELLA RIASSUNTIVA E CONFRONTO COEFFICIENTI
# -----------------------------------------------------------------------
print("\n" + "=" * 78)
print("4.3 TABELLA RIASSUNTIVA DEI MODELLI")
print("=" * 78)
summary_rows = []
for name, res in results.items():
    summary_rows.append({
        "Modello": name,
        "lambda.min": res["lam_min"],
        "CV-MSE (min)": res["mse_min"],
        "lambda.1se": res["lam_1se"],
        "Test MSE": res["test_mse"],
        "Test RMSE": np.sqrt(res["test_mse"]),
        "Test R2": res["test_r2"],
        "N. regressori non nulli": res["n_nonzero"],
    })
summary_rows.append({
    "Modello": "OLS (no regolarizzazione)",
    "lambda.min": 0.0, "CV-MSE (min)": np.nan, "lambda.1se": np.nan,
    "Test MSE": ols_test_mse, "Test RMSE": np.sqrt(ols_test_mse),
    "Test R2": ols_test_r2, "N. regressori non nulli": len(ALL_VARS),
})
summary_df = pd.DataFrame(summary_rows).set_index("Modello")
print(summary_df.round(4).to_string())

best_model_name = summary_df["Test MSE"].idxmin()
print(f"\n-> Modello con Test MSE piu' basso: '{best_model_name}'")

print("\nConfronto coefficienti a lambda.min (tutti i modelli):")
coef_table = pd.DataFrame({name: pd.Series(res["best_est"].coef_, index=ALL_VARS)
                            for name, res in results.items()})
coef_table["OLS"] = pd.Series(ols.coef_, index=ALL_VARS)
print(coef_table.round(2).to_string())

# Variabili azzerate (o quasi) da LASSO / Elastic Net a lambda.min
print("\nVariabili azzerate a lambda.min (|coef| < 1):")
for name in ["LASSO", "ElasticNet a=0.1", "ElasticNet a=0.9"]:
    zeroed = coef_table.index[coef_table[name].abs() < 1].tolist()
    print(f"  {name}: {zeroed if zeroed else 'nessuna'}")

# -----------------------------------------------------------------------
# 6. GRAFICO RIASSUNTIVO: TEST MSE PER MODELLO
# -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
summary_df["Test RMSE"].plot(kind="bar", ax=ax, color="#4C72B0")
ax.set_ylabel("Test RMSE ($)")
ax.set_title("Confronto Test RMSE tra modelli (train/test split 80/20)")
ax.set_ylim(summary_df["Test RMSE"].min() * 0.97, summary_df["Test RMSE"].max() * 1.02)
for i, v in enumerate(summary_df["Test RMSE"]):
    ax.text(i, v + 5, f"{v:,.0f}", ha="center", va="bottom", fontsize=9)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(f"{OUT}/13_test_rmse_comparison.png", dpi=150)
plt.close()
print(f"\n[Figura salvata] {OUT}/13_test_rmse_comparison.png")

import pickle
with open(f"{OUT}/step4_models.pkl", "wb") as f:
    pickle.dump({"results": results, "ols": ols, "summary_df": summary_df,
                 "scaler": scaler, "ALL_VARS": ALL_VARS}, f)
print(f"[Oggetti salvati] {OUT}/step4_models.pkl")

print("\n" + "=" * 78)
print("STEP 4 COMPLETATO")
print("=" * 78)
