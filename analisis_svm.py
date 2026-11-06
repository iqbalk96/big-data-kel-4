import json, warnings

warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import wilcoxon
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_validate,
    GridSearchCV,
    learning_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
)

plt.rcParams.update(
    {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
)
SVMC, GREY = "#C8553D", "#9DB4C0"
d = load_breast_cancer()
X = pd.DataFrame(d.data, columns=d.feature_names)
y = pd.Series(np.where(d.target == 0, 1, 0))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
out = {}

# ---- 1. Tuning SVM ----
pipe = Pipeline(
    [("sc", StandardScaler()), ("svm", SVC(probability=True, random_state=42))]
)
grid = {
    "svm__kernel": ["linear", "rbf", "poly", "sigmoid"],
    "svm__C": [0.01, 0.1, 1, 10, 100],
    "svm__gamma": ["scale", 0.001, 0.01, 0.1],
}
gs = GridSearchCV(
    pipe,
    grid,
    cv=StratifiedKFold(10, shuffle=True, random_state=42),
    scoring="f1",
    n_jobs=-1,
)
gs.fit(Xtr, ytr)
out["best_params"] = gs.best_params_
out["best_cv_f1"] = gs.best_score_
cvr = pd.DataFrame(gs.cv_results_)
kern = (
    cvr.groupby("param_svm__kernel")["mean_test_score"]
    .max()
    .sort_values(ascending=False)
)
out["kernel"] = [[k, round(v, 4)] for k, v in kern.items()]
print("best", gs.best_params_, gs.best_score_)
print(kern)

fig, ax = plt.subplots(figsize=(6, 3.2))
ax.bar(
    kern.index,
    kern.values,
    color=[SVMC if k == gs.best_params_["svm__kernel"] else GREY for k in kern.index],
)
for i, v in enumerate(kern.values):
    ax.text(i, v + 0.003, f"{v:.4f}", ha="center")
ax.set_ylim(min(kern.values) - 0.05, 1)
ax.set_ylabel("F1 (10-fold CV)")
ax.set_title("F1 Terbaik per Kernel SVM")
fig.tight_layout()
fig.savefig("fig/s01_kernel.png", dpi=200)
plt.close()

rbf = cvr[(cvr.param_svm__kernel == "rbf")].copy()
hm = rbf.pivot_table(
    index="param_svm__C",
    columns=rbf.param_svm__gamma.astype(str),
    values="mean_test_score",
)
fig, ax = plt.subplots(figsize=(6, 3.6))
sns.heatmap(hm, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax)
ax.set_xlabel("gamma")
ax.set_ylabel("C")
ax.set_title("Grid Search SVM RBF: F1 (10-fold CV)")
fig.tight_layout()
fig.savefig("fig/s02_grid.png", dpi=200)
plt.close()

svm = gs.best_estimator_
# ---- 2. Evaluasi data uji ----
models = {
    "SVM": svm,
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000)
    ),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
    "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
}
rows, probs, cms = [], {}, {}
for n, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    pb = m.predict_proba(Xte)[:, 1]
    probs[n] = pb
    cms[n] = confusion_matrix(yte, p).tolist()
    tn, fp, fn, tp = confusion_matrix(yte, p).ravel()
    rows.append(
        [
            n,
            accuracy_score(yte, p),
            precision_score(yte, p),
            recall_score(yte, p),
            tn / (tn + fp),
            f1_score(yte, p),
            roc_auc_score(yte, pb),
            matthews_corrcoef(yte, p),
            int(fp),
            int(fn),
        ]
    )
test = pd.DataFrame(
    rows,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1",
        "AUC",
        "MCC",
        "FP",
        "FN",
    ],
)
out["test"] = test.round(4).values.tolist()
out["cms"] = cms
ps = svm.predict(Xte)
out["report"] = classification_report(
    yte, ps, target_names=["Jinak", "Ganas"], output_dict=True
)
out["kappa"] = cohen_kappa_score(yte, ps)
out["n_sv"] = svm.named_steps["svm"].n_support_.tolist()
out["ap"] = average_precision_score(yte, probs["SVM"])
print(test.round(4).to_string())

pal = {
    "SVM": SVMC,
    "Logistic Regression": "#2E86AB",
    "Random Forest": "#6A994E",
    "k-NN": "#F2A541",
    "Naive Bayes": "#7B5EA7",
    "Decision Tree": "#5C6B73",
}
fig, ax = plt.subplots(figsize=(10, 4))
mets = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "AUC", "MCC"]
w = 0.13
xs = np.arange(len(mets))
for i, r in test.iterrows():
    ax.bar(xs + (i - 2.5) * w, r[mets].values, w, label=r.Model, color=pal[r.Model])
ax.set_xticks(xs)
ax.set_xticklabels(mets)
ax.set_ylim(0.75, 1.01)
ax.set_title("Perbandingan Kinerja SVM dan 5 Model Lain (Data Uji)")
ax.legend(
    ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False
)
fig.tight_layout()
fig.savefig("fig/s03_banding.png", dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(4.2, 3.6))
sns.heatmap(
    cms["SVM"],
    annot=True,
    fmt="d",
    cmap="Reds",
    cbar=False,
    ax=ax,
    xticklabels=["Jinak", "Ganas"],
    yticklabels=["Jinak", "Ganas"],
    annot_kws={"size": 16},
)
ax.set_xlabel("Prediksi")
ax.set_ylabel("Aktual")
ax.set_title("Confusion Matrix SVM (n=114)")
fig.tight_layout()
fig.savefig("fig/s04_cm_svm.png", dpi=200)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for n, pb in probs.items():
    f, t, _ = roc_curve(yte, pb)
    lw = 2.6 if n == "SVM" else 1.2
    axes[0].plot(f, t, color=pal[n], lw=lw, label=f"{n} ({roc_auc_score(yte, pb):.3f})")
    pr, rc, _ = precision_recall_curve(yte, pb)
    axes[1].plot(
        rc,
        pr,
        color=pal[n],
        lw=lw,
        label=f"{n} ({average_precision_score(yte, pb):.3f})",
    )
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set(
    xlabel="False Positive Rate", ylabel="True Positive Rate", title="Kurva ROC (AUC)"
)
axes[1].set(xlabel="Recall", ylabel="Precision", title="Kurva Precision-Recall (AP)")
for a in axes:
    a.legend(fontsize=7.5, loc="lower right" if a is axes[0] else "lower left")
fig.tight_layout()
fig.savefig("fig/s05_roc_pr.png", dpi=200)
plt.close()

# threshold analysis SVM
th = np.arange(0.1, 0.91, 0.05)
tr = []
for t in th:
    p = (probs["SVM"] >= t).astype(int)
    tr.append(
        [
            t,
            recall_score(yte, p),
            precision_score(yte, p, zero_division=0),
            f1_score(yte, p),
            accuracy_score(yte, p),
        ]
    )
tr = pd.DataFrame(tr, columns=["t", "Recall", "Precision", "F1", "Accuracy"])
out["thr"] = tr.round(4).values.tolist()
fig, ax = plt.subplots(figsize=(7, 3.6))
for c, col in zip(
    ["Recall", "Precision", "F1", "Accuracy"], [SVMC, "#2E86AB", "#6A994E", "#5C6B73"]
):
    ax.plot(tr.t, tr[c], marker="o", ms=3, label=c, color=col)
ax.axvline(0.5, ls="--", c="grey", lw=1)
ax.set_xlabel("Ambang batas probabilitas")
ax.set_title("Pengaruh Threshold terhadap Kinerja SVM")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("fig/s06_threshold.png", dpi=200)
plt.close()

# learning curve
sz, trs, vs = learning_curve(
    svm,
    Xtr,
    ytr,
    cv=StratifiedKFold(10, shuffle=True, random_state=42),
    train_sizes=np.linspace(0.1, 1, 8),
    scoring="accuracy",
    n_jobs=-1,
)
fig, ax = plt.subplots(figsize=(7, 3.6))
for s_, c, l in [(trs, "#2E86AB", "Skor latih"), (vs, SVMC, "Skor validasi")]:
    ax.plot(sz, s_.mean(1), marker="o", color=c, label=l)
    ax.fill_between(
        sz, s_.mean(1) - s_.std(1), s_.mean(1) + s_.std(1), color=c, alpha=0.15
    )
ax.set_xlabel("Jumlah data latih")
ax.set_ylabel("Akurasi")
ax.set_title("Learning Curve SVM")
ax.legend()
fig.tight_layout()
fig.savefig("fig/s07_learning.png", dpi=200)
plt.close()
out["lc"] = [round(trs.mean(1)[-1], 4), round(vs.mean(1)[-1], 4)]

# PCA decision boundary
Z = StandardScaler().fit_transform(X)
pca = PCA(2).fit(Z)
Z2 = pca.transform(Z)
s2 = SVC(
    kernel=gs.best_params_["svm__kernel"],
    C=gs.best_params_["svm__C"],
    gamma=gs.best_params_["svm__gamma"],
).fit(Z2, y)
xx, yy = np.meshgrid(
    np.linspace(Z2[:, 0].min() - 1, Z2[:, 0].max() + 1, 300),
    np.linspace(Z2[:, 1].min() - 1, Z2[:, 1].max() + 1, 300),
)
zz = s2.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.contourf(xx, yy, zz, levels=[-50, 0, 50], colors=["#D6E8F3", "#F6D5CE"], alpha=0.8)
ax.contour(
    xx,
    yy,
    zz,
    levels=[-1, 0, 1],
    colors="k",
    linestyles=["--", "-", "--"],
    linewidths=[1, 1.8, 1],
)
ax.scatter(Z2[y == 0, 0], Z2[y == 0, 1], s=10, c="#2E86AB", label="Jinak")
ax.scatter(Z2[y == 1, 0], Z2[y == 1, 1], s=10, c=SVMC, label="Ganas")
sv = s2.support_vectors_
ax.scatter(
    sv[:, 0],
    sv[:, 1],
    s=40,
    facecolors="none",
    edgecolors="k",
    lw=0.7,
    label="Support vector",
)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("Hyperplane & Margin SVM (proyeksi 2D PCA)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("fig/s08_boundary.png", dpi=200)
plt.close()
out["pca_var"] = pca.explained_variance_ratio_.sum()

# ---- 3. Pengujian: Repeated 10-fold CV (10x) pada seluruh data ----
rcv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
from sklearn.base import clone

allm = dict(models)
allm["Baseline (Dummy)"] = DummyClassifier(strategy="most_frequent")
sc = {}
for n, m in allm.items():
    r = cross_validate(
        clone(m),
        X,
        y,
        cv=rcv,
        scoring=["accuracy", "f1", "roc_auc", "recall", "precision"],
        n_jobs=-1,
    )
    sc[n] = {k.replace("test_", ""): v for k, v in r.items() if k.startswith("test_")}
summ = []
for n in allm:
    summ.append(
        [n]
        + [
            f"{sc[n][k].mean():.4f} ± {sc[n][k].std():.4f}"
            for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]
        ]
    )
out["rcv"] = summ
print(pd.DataFrame(summ).to_string())
out["rcv_mean"] = {n: {k: float(v.mean()) for k, v in sc[n].items()} for n in allm}

# Wilcoxon signed-rank
wil = []
for metric in ["accuracy", "f1", "roc_auc"]:
    for n in allm:
        if n == "SVM":
            continue
        a, b = sc["SVM"][metric], sc[n][metric]
        diff = a - b
        try:
            st, p = wilcoxon(a, b, zero_method="wilcox", alternative="greater")
        except ValueError:
            st, p = np.nan, 1.0
        try:
            _, p2 = wilcoxon(a, b, zero_method="wilcox")
        except ValueError:
            p2 = 1.0
        wins = int((diff > 0).sum())
        ties = int((diff == 0).sum())
        loss = int((diff < 0).sum())
        wil.append(
            [
                metric,
                n,
                float(a.mean()),
                float(b.mean()),
                float(diff.mean()),
                float(st),
                float(p2),
                float(p),
                wins,
                ties,
                loss,
            ]
        )
W = pd.DataFrame(
    wil,
    columns=[
        "metric",
        "lawan",
        "svm",
        "lawan_mean",
        "selisih",
        "W",
        "p_two",
        "p_greater",
        "menang",
        "seri",
        "kalah",
    ],
)


# Koreksi Holm per metrik (6 perbandingan)
def holm(p):
    p = np.array(p)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0
    for i, k in enumerate(o):
        run = max(run, min(1, (m - i) * p[k]))
        adj[k] = run
    return adj


W["p_holm"] = np.concatenate(
    [holm(g.p_greater.values) for _, g in W.groupby("metric", sort=False)]
)
out["wil"] = W.round(6).values.tolist()
print(W.round(5).to_string())

# grafik boxplot repeated CV
fig, ax = plt.subplots(figsize=(10, 4))
order = list(models)
ax.boxplot(
    [sc[n]["accuracy"] for n in order],
    tick_labels=order,
    patch_artist=True,
    boxprops=dict(facecolor=GREY),
    medianprops=dict(color="k"),
)
ax.patches[0].set_facecolor(SVMC)
ax.set_ylabel("Akurasi")
ax.set_title("Sebaran Akurasi Repeated 10-Fold CV (10 x 10 = 100 fold)")
fig.tight_layout()
fig.savefig("fig/s09_rcv.png", dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(10, 3.8))
sub = W[W.metric == "accuracy"]
pp = np.maximum(sub.p_greater.to_numpy(dtype=float), 1e-20)
lg = -np.log10(pp)
names = list(sub.lawan)
ax.barh(range(len(names)), lg, color=[SVMC if p < 0.05 else GREY for p in pp])
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names)
ax.axvline(-np.log10(0.05), ls="--", c="k", lw=1)
ax.text(-np.log10(0.05) + 0.15, 5.45, "α = 0,05", fontsize=8)
for i, (p, l) in enumerate(zip(pp, lg)):
    ax.text(
        max(l, 1.35) + 0.2,
        i,
        ("p < 1e-20" if p <= 1e-20 else f"p = {p:.2e}")
        + ("" if p < 0.05 else "  (tidak signifikan)"),
        va="center",
        fontsize=8,
    )
ax.set_xlim(0, lg.max() * 1.25)
ax.invert_yaxis()
ax.set_xlabel("-log10(p-value)  (semakin panjang = semakin signifikan)")
ax.set_title("Uji Wilcoxon: SVM vs Model Lain (Akurasi, H1: SVM > lawan)")
fig.tight_layout()
fig.savefig("fig/s10_wilcoxon.png", dpi=200)
plt.close()

fig, ax = plt.subplots(figsize=(10, 3.6))
rng = np.random.default_rng(0)
for i, n in enumerate(order[1:]):
    dd = sc["SVM"]["accuracy"] - sc[n]["accuracy"]
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(dd)), dd, alpha=0.3, color=pal[n], s=14)
    ax.scatter(i, dd.mean(), color="k", marker="D", s=40, zorder=3)
ax.set_xticks(range(5))
ax.set_xticklabels(order[1:])
ax.axhline(0, c="k", lw=1)
ax.set_ylabel("Akurasi SVM - Akurasi lawan")
ax.set_title(
    "Selisih Akurasi per Fold (titik di atas 0 = SVM lebih baik; ◆ = rata-rata)"
)
fig.tight_layout()
fig.savefig("fig/s11_selisih.png", dpi=200)
plt.close()

json.dump(out, open("info_svm.json", "w"), indent=1, default=str)
