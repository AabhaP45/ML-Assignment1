import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler
from sklearn.linear_model import RidgeCV, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
ROLL = "BT2024130"

# Max polynomial degree to test for each problem variant
CONFIG = {1: 10, 2: 20}

PHASE_TITLES = {
    1: "Phase 1: Power Plant Steam Turbine Optimization (var1)",
    2: "Phase 2: Subterranean Thermal Reservoir Mapping (var2)",
}

# Ridge alphas
RIDGE_ALPHAS = np.logspace(-8, 2, 21)

# Lasso settings (var1)
LASSO_KW = dict(alpha=0.01, max_iter=20000, tol=1e-4)


def build_ridge(degree):
    # Scale inputs -> polynomial expansion -> rescale features -> Ridge
    return make_pipeline(
        MinMaxScaler(feature_range=(-1, 1)),
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        RidgeCV(alphas=RIDGE_ALPHAS),
    )


def build_lasso(degree):
    # Polynomial expansion -> standardize -> Lasso (L1)
    return make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        Lasso(**LASSO_KW),
    )


def build_model(prb, degree):
    return build_lasso(degree) if prb == 1 else build_ridge(degree)


def get_features(df):
    # Any column starting with 'x' is an input feature
    return [c for c in df.columns if c.lower().startswith("x")]


def plot_results(res, chosen, prb, reg):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.errorbar(res.degree, res.mse, yerr=res.mse_std, marker="o", capsize=3)
    ax.axvline(chosen, color="red", linestyle="--", label=f"chosen degree = {chosen}")
    ax.set_yscale("log")
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("CV MSE (log scale)")
    ax.set_title(f"var{prb} [{reg}]: Degree vs MSE")
    ax.set_xticks(res.degree)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(res.degree, res.r2, marker="o", color="tab:green")
    ax.axvline(chosen, color="red", linestyle="--", label=f"chosen degree = {chosen}")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("CV R$^2$ (symlog scale)")
    ax.set_title(f"var{prb} [{reg}]: Degree vs R$^2$")
    ax.set_xticks(res.degree)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fname = f"plots_var{prb}.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved {fname}")


def run(prb):
    max_deg = CONFIG[prb]
    reg = "L1 (Lasso)" if prb == 1 else "L2 (Ridge)"
    train = pd.read_csv(f"{ROLL}_train_var{prb}.csv")
    test = pd.read_csv(f"{ROLL}_test_var{prb}.csv")
    feats = get_features(train)
    print(f"\n{PHASE_TITLES[prb]}")

    X, y = train[feats].values, train["y"].values
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # Step 1: choose the best polynomial degree with 5-fold CV
    rows = []
    for d in range(1, max_deg + 1):
        model = build_model(prb, d)
        mse = -cross_val_score(model, X, y, cv=kf,
                               scoring="neg_mean_squared_error",
                               n_jobs=-1, error_score=np.nan)
        r2 = cross_val_score(model, X, y, cv=kf, scoring="r2",
                             n_jobs=-1, error_score=np.nan)
        rows.append((d, mse.mean(), mse.std(), r2.mean()))
        print(f"degree {d:2d} | MSE {mse.mean():.6f} (+-{mse.std():.4f}) | R2 {r2.mean():.5f}")
    res = pd.DataFrame(rows, columns=["degree", "mse", "mse_std", "r2"])
    res.to_csv(f"results_var{prb}.csv", index=False)

    chosen = int(res.loc[res.mse.idxmin(), "degree"])
    print(f"Chosen degree (min MSE): {chosen}")
    plot_results(res, chosen, prb, reg)

    # Step 2: sanity check on a 20% hold-out set
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=SEED)
    m = build_model(prb, chosen).fit(Xtr, ytr)
    p = m.predict(Xva)
    print(f"Hold-out MSE {mean_squared_error(yva, p):.6f}, R2 {r2_score(yva, p):.5f}")

    # Step 3: fit on full training set, predict on the test set
    final = build_model(prb, chosen).fit(X, y)
    if prb == 1:
        coef = final[-1].coef_
        nz = int(np.sum(np.abs(coef) > 1e-6))
        print(f"Lasso params: {LASSO_KW} | non-zero coefs: {nz}/{coef.size}")

    preds = final.predict(test[feats].values)
    out = test.copy()
    out["y"] = preds
    out.to_csv(f"{ROLL}_pred_var{prb}.csv", index=False)
    print(f"Saved {ROLL}_pred_var{prb}.csv")


if __name__ == "__main__":
    for prb in (1, 2):
        run(prb)