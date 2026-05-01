# =========================================================================
#  CODE REKOR: 0.99439 (IMPROVED REGAN HYBRID - TUNED VERSION)
#  Strategi: Hybrid TF-IDF + Stacking + Pseudo Labeling + Rank-Blend Tuning
#  Fix: Anti-OOM (Float32) & Path Auto-Detect
# =========================================================================

import os
import sys
import gc
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack, vstack, csr_matrix
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

# ================== 1. PATH CONFIGURATION ==================
if "__file__" in globals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    BASE_DIR = os.getcwd()


if os.path.exists(os.path.join(BASE_DIR, "data", "train.csv")):
    WORK_DIR = os.path.join(BASE_DIR, "data")
else:
    WORK_DIR = BASE_DIR

TRAIN_PATH = os.path.join(WORK_DIR, "train.csv")
TEST_PATH = os.path.join(WORK_DIR, "test.csv")
SAMPLE_PATH = os.path.join(WORK_DIR, "sample_submission.csv")

print("ð¥ STARTING REKOR ENGINE (TUNED)...")
print(f"ð Reading Data from: {WORK_DIR}")

if not os.path.exists(TRAIN_PATH):
    sys.exit("â ERROR: File train.csv tidak ditemukan!")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

# ================== 2. IMPROVED FEATURE ENGINEERING ==================
def improved_fe(df: pd.DataFrame) -> pd.DataFrame:
    url = df["URL"].astype(str).str.lower()

    # Numeric Basics
    df["len"] = url.str.len()
    df["digits"] = url.str.count(r"\d")
    df["dots"] = url.str.count(r"\.")
    df["slash"] = url.str.count("/")
    df["ques"] = url.str.count(r"\?")
    df["eq"] = url.str.count("=")
    df["dash"] = url.str.count("-")
    df["at"] = url.str.contains("@", regex=False).astype(int)

    # Ratios
    df["digit_ratio"] = df["digits"] / (df["len"] + 1)
    df["dot_ratio"] = df["dots"] / (df["len"] + 1)

    # Killer Keywords
    sus = (
        "login|secure|update|verify|bank|paypal|ebay|amazon|webscr|signin|password|"
        "admin|cmd|root|free|gift|win|claim|prize|urgent|confirm|account|mozi"
    )
    df["sus_count"] = url.str.count(sus)
    df["has_sus"] = (df["sus_count"] > 0).astype(int)

    # Interactions
    if "https_flag" in df.columns:
        df["sus_nohttps"] = df["has_sus"] * (df["https_flag"] == 0)
    else:
        df["sus_nohttps"] = 0

    if "tld_popularity" in df.columns:
        df["bad_tld"] = (df["tld_popularity"] <= 100).astype(int)
    else:
        df["bad_tld"] = 0

    if "url_entropy" in df.columns:
        df["high_entropy"] = (df["url_entropy"] > 4.4).astype(int)
    else:
        df["high_entropy"] = 0

    # Extra Features
    if "path_length" in df.columns:
        df["long_path"] = (df["path_length"] > 80).astype(int)
    if "query_param_count" in df.columns:
        df["many_params"] = (df["query_param_count"] >= 5).astype(int)

    return df.fillna(0)


print("â³ Engineering Features...")
train = improved_fe(train)
test = improved_fe(test)

# ================== 3. HYBRID TF-IDF (CHAR + WORD) ==================
print("â³ Vectorizing (Hybrid Mode - RAM Safe)...")

# Char N-Gram (10k)
tf_char = TfidfVectorizer(
    analyzer="char", ngram_range=(2, 5), max_features=10000, min_df=2, sublinear_tf=True
)

# Word N-Gram (4k)
tf_word = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=4000,
    min_df=2,
    sublinear_tf=True,
    token_pattern=r"\w+",
)

all_url = pd.concat([train["URL"], test["URL"]])
tf_char.fit(all_url)
tf_word.fit(all_url)

train_char = tf_char.transform(train["URL"])
test_char = tf_char.transform(test["URL"])
train_word = tf_word.transform(train["URL"])
test_word = tf_word.transform(test["URL"])

X_text = hstack([train_char, train_word])
X_test_text = hstack([test_char, test_word])

# Numeric
num_cols = [c for c in train.columns if c not in ["URL", "ClassLabel", "ID", "class_lable"]]
X_num = csr_matrix(train[num_cols].values)
X_test_num = csr_matrix(test[num_cols].values)

# Final Matrix (Float32 agar RAM Aman)
X = hstack([X_text, X_num]).tocsr().astype(np.float32)
X_test = hstack([X_test_text, X_test_num]).tocsr().astype(np.float32)
y = train["ClassLabel"].values

print(f"   Final Shape: {X.shape}")

del tf_char, tf_word, train_char, test_char, train_word, test_word, all_url
gc.collect()

# ================== 4. TRAINING DUAL ENGINE ==================
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

lgb_oof = np.zeros(len(train))
cat_oof = np.zeros(len(train))
lgb_pred = np.zeros(len(test))
cat_pred = np.zeros(len(test))

print("\nð Training LightGBM DART...")
for fold, (tr, val) in enumerate(skf.split(X, y), 1):
    m = LGBMClassifier(
        boosting_type="dart",
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=128,
        max_depth=13,
        colsample_bytree=0.68,
        subsample=0.82,
        reg_alpha=0.09,
        reg_lambda=0.11,
        random_state=42 + fold,  # sedikit diversifikasi seed per fold
        n_jobs=-1,
        verbose=-1,
    )
    m.fit(
        X[tr],
        y[tr],
        eval_set=[(X[val], y[val])],
        callbacks=[lgb.early_stopping(200, False), lgb.log_evaluation(0)],
    )

    lgb_oof[val] = m.predict_proba(X[val])[:, 1]
    lgb_pred += m.predict_proba(X_test)[:, 1] / skf.n_splits

    del m
    gc.collect()
    print(f"   Fold {fold} LGBM â")

print(f"   >>> LGBM AUC: {roc_auc_score(y, lgb_oof):.6f}")

print("\nð Training CatBoost (CPU - Safe Mode)...")

# Sanitize Data (PENTING BIAR GAK CRASH)
X.data = np.nan_to_num(X.data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
X_test.data = np.nan_to_num(X_test.data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

for fold, (tr, val) in enumerate(skf.split(X, y), 1):
    m = CatBoostClassifier(
        iterations=4000,
        learning_rate=0.025,
        depth=9,
        task_type="CPU",
        thread_count=-1,
        verbose=0,
        random_seed=42 + fold,
        early_stopping_rounds=200,
        allow_writing_files=False,
        max_bin=128,
    )
    m.fit(X[tr], y[tr], eval_set=(X[val], y[val]))

    cat_oof[val] = m.predict_proba(X[val])[:, 1]
    cat_pred += m.predict_proba(X_test)[:, 1] / skf.n_splits

    del m
    gc.collect()
    print(f"   Fold {fold} CatBoost â")

print(f"   >>> CatBoost AUC: {roc_auc_score(y, cat_oof):.6f}")

# ================== 5. STACKING META LOGISTIC ==================
print("\nâ  Stacking Meta-Model...")
stack_train = np.column_stack((lgb_oof, cat_oof))
stack_test = np.column_stack((lgb_pred, cat_pred))

meta = LogisticRegression(max_iter=500, C=0.5, random_state=42)
meta.fit(stack_train, y)

meta_oof = meta.predict_proba(stack_train)[:, 1]
meta_pred = meta.predict_proba(stack_test)[:, 1]

print(f"   >>> META AUC (LGB+CAT -> LR): {roc_auc_score(y, meta_oof):.6f}")

# ================== 6. PSEUDO LABELING (ADAPTIVE) ==================
print("\nð® Pseudo Labeling (adaptive thresholds)...")
blend = meta_pred

# Adaptive thresholds with percentiles
high_thr = np.percentile(blend, 99.5)
low_thr = np.percentile(blend, 0.5)
mask_raw = (blend > high_thr) | (blend < low_thr)
idx = np.where(mask_raw)[0]

max_pseudo = int(0.4 * len(train))  # cap 40% of train size
if len(idx) > max_pseudo:
    # keep most confident (farthest from 0.5)
    conf = np.abs(blend[idx] - 0.5)
    top_idx = np.argsort(-conf)[:max_pseudo]
    idx = idx[top_idx]

if len(idx) > 300:
    print(f"   Adding {len(idx)} high-confidence pseudo samples...")
    X_big = vstack([X, X_test[idx]])
    y_pseudo = (blend[idx] > 0.5).astype(int)
    y_big = np.concatenate([y, y_pseudo])

    print("   Training Final CatBoost (student)...")
    final_cat = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.02,
        depth=8,
        random_seed=2025,
        task_type="CPU",
        thread_count=-1,
        verbose=500,
        allow_writing_files=False,
        max_bin=128,
    )
    final_cat.fit(X_big, y_big)
    final_pred_pseudo = final_cat.predict_proba(X_test)[:, 1]
else:
    print("   Not enough high-confidence pseudo samples, using meta only.")
    final_pred_pseudo = blend

# ================== 7. FINAL RANK BLEND (TUNED BY OOF) ==================
print("\nð Final Rank Blending (tuned by OOF)...")

# Use OOF to search best alpha between LGBM and META
r_lgb_oof = rankdata(lgb_oof)
r_meta_oof = rankdata(meta_oof)

best_auc = 0.0
best_alpha = 0.6

for a in np.linspace(0.0, 1.0, 21):  # step 0.05
    blend_oof_rank = a * r_lgb_oof + (1 - a) * r_meta_oof
    auc = roc_auc_score(y, blend_oof_rank)
    if auc > best_auc:
        best_auc = auc
        best_alpha = a

print(f"   Best alpha (LGB vs META OOF): {best_alpha:.2f} | OOF AUC={best_auc:.6f}")

# Apply tuned alpha on TEST between LGBM and final_pred_pseudo
r_lgb_test = rankdata(lgb_pred)
r_final_test = rankdata(final_pred_pseudo)

final = best_alpha * r_lgb_test + (1 - best_alpha) * r_final_test
final = final / final.max()

# ================== SAVE ==================
target_col = [c for c in sample.columns if c.lower() != "id"][0]
sample[target_col] = final

output_file = os.path.join(BASE_DIR, "REKOR_TUNED2_SUBMISSION.csv")
sample.to_csv(output_file, index=False)

print(f"\nâ DONE! File saved: {output_file}")
