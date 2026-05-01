# ============================================================
#  ULTIMATE DSCO PHISHING URL ENGINE (CPU, AGGRESSIVE LB)
#  - Hybrid TF-IDF (char + word) + rich phishing features
#  - LGBM DART + CatBoost (10-fold) as teacher
#  - OOF-tuned blend (LGB vs Cat) for teacher
#  - High-confidence pseudo-labeling + CatBoost student
#  - Final rank blend (LGB vs Student)
#  - Runtime timer + GLOBAL ETA (time remaining)
# ============================================================

import os
import sys
import gc
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

from urllib.parse import urlparse
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, vstack, csr_matrix
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

# ================== 1. PATH, TIMER & PROGRESS SETUP ==================

overall_start = time.time()

# total "unit kerja" kira-kira:
# 1 = feature eng
# 1 = TF-IDF
# 10 = LGB folds
# 10 = Cat folds
# 1 = pseudo/student
TOTAL_UNITS = 1 + 1 + 10 + 10 + 1
units_done = 0

def log(msg: str):
    """Just print elapsed time + message (tanpa ETA global)."""
    elapsed = (time.time() - overall_start) / 60.0
    print(f"[+{elapsed:6.2f} min] {msg}")

def log_step(msg: str, add_units: int = 1):
    """
    Log progress step + update global ETA.
    add_units = berapa unit kerja yang dianggap selesai di step ini.
    """
    global units_done
    units_done += add_units
    now = time.time()
    elapsed = now - overall_start
    avg_per_unit = elapsed / max(units_done, 1)
    remaining_units = max(TOTAL_UNITS - units_done, 0)
    remaining_time = remaining_units * avg_per_unit
    print(
        f"[+{elapsed/60.0:6.2f} min] {msg} | "
        f"Est. total remaining: {remaining_time/60.0:6.2f} min"
    )

# Resolve base dir (works both from script & notebook)
if "__file__" in globals():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    BASE_DIR = os.getcwd()

# Support either ./data or current folder
if os.path.exists(os.path.join(BASE_DIR, "data", "train.csv")):
    WORK_DIR = os.path.join(BASE_DIR, "data")
else:
    WORK_DIR = BASE_DIR

TRAIN_PATH = os.path.join(WORK_DIR, "train.csv")
TEST_PATH = os.path.join(WORK_DIR, "test.csv")
SAMPLE_PATH = os.path.join(WORK_DIR, "sample_submission.csv")

log("Starting ULTIMATE DSCO PHISH ENGINE (CPU Aggressive)")
log(f"Using data from: {WORK_DIR}")

if not os.path.exists(TRAIN_PATH):
    sys.exit("â ERROR: train.csv not found!")

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_PATH)

log(f"Loaded train: {train.shape}, test: {test.shape}")


# ================== 2. FEATURE ENGINEERING ==================

def phishing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rich lexical + meta URL features for phishing detection.
    """
    url = df["URL"].astype(str).str.strip()
    url_lower = url.str.lower()

    # --- basic counts (lexical) ---
    df["len"] = url_lower.str.len()
    df["digits"] = url_lower.str.count(r"\d")
    df["dots"] = url_lower.str.count(r"\.")
    df["slash"] = url_lower.str.count(r"/")
    df["ques"] = url_lower.str.count(r"\?")
    df["eq"] = url_lower.str.count(r"=")
    df["dash"] = url_lower.str.count(r"-")
    df["at"] = url_lower.str.contains("@", regex=False).astype(int)
    df["hash"] = url_lower.str.count(r"#")
    df["amp"] = url_lower.str.count(r"&")
    df["percent"] = url_lower.str.count(r"%")

    # ratios
    df["digit_ratio"] = df["digits"] / (df["len"] + 1.0)
    df["dot_ratio"] = df["dots"] / (df["len"] + 1.0)
    df["slash_ratio"] = df["slash"] / (df["len"] + 1.0)
    df["special_count"] = url_lower.str.count(r"[@\-\._=%\?&#]")
    df["special_ratio"] = df["special_count"] / (df["len"] + 1.0)

    # --- suspicious keywords (common in phishing) ---
    sus_pattern = (
        "login|logon|signin|verify|update|secure|account|bank|paypal|ebay|amazon|"
        "webscr|password|reset|confirm|billing|invoice|limited|urgent|alert|"
        "support|security|unlock|win|reward|gift|prize|free|bonus|claim|"
        "auth|authenticate|credential|wallet|crypto"
    )
    df["sus_count"] = url_lower.str.count(sus_pattern)
    df["has_sus"] = (df["sus_count"] > 0).astype(int)

    # --- advanced lexical indicators ---
    df["has_redirect"] = url_lower.str.contains(r"(redirect|url|goto)=", regex=True).astype(int)
    df["hex_encoded"] = url_lower.str.contains(r"%[0-9a-fA-F]{2}", regex=True).astype(int)
    df["double_slash"] = (url_lower.str.count("//") > 1).astype(int)
    df["has_non_ascii"] = url.str.contains(r"[^\x00-\x7F]", regex=True).astype(int)  # IDN/homograph risk

    # --- meta columns from dataset (if exist) ---
    if "https_flag" in df.columns:
        df["https_flag"] = df["https_flag"].fillna(0)
        df["sus_nohttps"] = df["has_sus"] * (df["https_flag"] == 0)
        df["starts_http"] = ((df["https_flag"] == 0) & url_lower.str.startswith("http://")).astype(int)
    else:
        df["sus_nohttps"] = 0
        df["starts_http"] = url_lower.str.startswith("http://").astype(int)

    if "tld_popularity" in df.columns:
        df["bad_tld"] = (df["tld_popularity"] <= 100).astype(int)
    else:
        df["bad_tld"] = 0

    if "url_entropy" in df.columns:
        df["high_entropy"] = (df["url_entropy"] > 4.4).astype(int)
    else:
        df["high_entropy"] = 0

    if "path_length" in df.columns:
        df["long_path"] = (df["path_length"] > 80).astype(int)
    else:
        df["long_path"] = 0

    if "query_param_count" in df.columns:
        df["many_params"] = (df["query_param_count"] >= 5).astype(int)
    else:
        df["many_params"] = 0

    if "url_length" in df.columns:
        df["long_url"] = (df["url_length"] > 130).astype(int)
    else:
        df["long_url"] = (df["len"] > 130).astype(int)

    if "subdomain_count" in df.columns:
        df["many_sub"] = (df["subdomain_count"] >= 3).astype(int)
    else:
        df["many_sub"] = 0

    if "has_ip_address" in df.columns:
        df["sus_ip"] = df["has_sus"] * df["has_ip_address"]
    else:
        df["sus_ip"] = 0

    if "domain_name_length" in df.columns:
        df["long_domain"] = (df["domain_name_length"] > 30).astype(int)
    else:
        df["long_domain"] = 0

    # --- interactions ---
    df["ent_sus"] = df["high_entropy"] * df["has_sus"]
    df["long_ent"] = df["high_entropy"] * df["long_url"]

    # --- host/path level features via urlparse (vectorized-ish) ---
    parsed = url.apply(lambda u: pd.Series(urlparse(u)[:3], index=["scheme", "netloc", "path"]))
    host = parsed["netloc"].fillna("")
    path = parsed["path"].fillna("")

    df["host_len"] = host.str.len()
    df["host_dots"] = host.str.count(r"\.")
    df["host_digits"] = host.str.count(r"\d")
    df["host_dash"] = host.str.count(r"-")
    df["host_digit_ratio"] = df["host_digits"] / (df["host_len"] + 1.0)

    df["path_len"] = path.str.len()
    df["path_slash"] = path.str.count(r"/")
    df["path_depth"] = df["path_slash"]
    df["path_digits"] = path.str.count(r"\d")

    del parsed, host, path

    return df.fillna(0)


log("Engineering phishing features...")
train = phishing_features(train)
test = phishing_features(test)
log_step(f"Feature engineering done. Train cols: {len(train.columns)}", add_units=1)


# ================== 3. HYBRID TF-IDF (CHAR + WORD) ==================

log("Vectorizing URL text (Hybrid TF-IDF)...")

tf_char = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    max_features=10000,
    min_df=2,
    sublinear_tf=True,
)

tf_word = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 3),
    max_features=4000,
    min_df=2,
    sublinear_tf=True,
    token_pattern=r"\w+",
)

all_url = pd.concat([train["URL"], test["URL"]], axis=0)
tf_char.fit(all_url)
tf_word.fit(all_url)

train_char = tf_char.transform(train["URL"])
test_char = tf_char.transform(test["URL"])
train_word = tf_word.transform(train["URL"])
test_word = tf_word.transform(test["URL"])

X_text = hstack([train_char, train_word])
X_test_text = hstack([test_char, test_word])

drop_cols = ["URL", "ClassLabel", "ID", "class_lable"]
num_cols = [c for c in train.columns if c not in drop_cols]
X_num = csr_matrix(train[num_cols].values)
X_test_num = csr_matrix(test[num_cols].values)

X = hstack([X_text, X_num]).tocsr().astype(np.float32)
X_test = hstack([X_test_text, X_test_num]).tocsr().astype(np.float32)
y = train["ClassLabel"].values

log_step(f"TF-IDF & numeric matrix built. Shape: {X.shape}", add_units=1)

del tf_char, tf_word, train_char, test_char, train_word, test_word, all_url, X_text, X_test_text
gc.collect()


# ================== 4. TRAIN LGBM + CATBOOST (TEACHER) ==================

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_oof = np.zeros(len(train), dtype=np.float32)
cat_oof = np.zeros(len(train), dtype=np.float32)
lgb_pred = np.zeros(len(test), dtype=np.float32)
cat_pred = np.zeros(len(test), dtype=np.float32)

# ---- LGBM DART ----
log("Training LightGBM DART (10-fold)...")
lgb_start = time.time()
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    fold_start = time.time()
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    lgb_model = LGBMClassifier(
        boosting_type="dart",
        n_estimators=3500,
        learning_rate=0.025,
        num_leaves=160,
        max_depth=13,
        colsample_bytree=0.68,
        subsample=0.82,
        reg_alpha=0.09,
        reg_lambda=0.11,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1,
    )

    lgb_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(200, False), lgb.log_evaluation(0)],
    )

    lgb_oof[val_idx] = lgb_model.predict_proba(X_val)[:, 1]
    lgb_pred += lgb_model.predict_proba(X_test)[:, 1] / n_splits

    fold_time = time.time() - fold_start
    elapsed_lgb = time.time() - lgb_start
    avg_per_fold = elapsed_lgb / fold
    remaining_lgb = avg_per_fold * (n_splits - fold)

    log_step(
        f"LGB Fold {fold}/{n_splits} done in {fold_time/60:.2f} min | "
        f"Est. remaining LGB only: {remaining_lgb/60:.2f} min",
        add_units=1,
    )

    del lgb_model, X_tr, X_val, y_tr, y_val
    gc.collect()

log(f"LGBM OOF AUC: {roc_auc_score(y, lgb_oof):.6f}")

# ---- CatBoost ----
log("Training CatBoost (CPU, 10-fold)...")

X.data = np.nan_to_num(X.data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
X_test.data = np.nan_to_num(X_test.data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

cat_start = time.time()
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
    fold_start = time.time()
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    cat_model = CatBoostClassifier(
        iterations=4000,
        learning_rate=0.022,
        depth=9,
        task_type="CPU",
        thread_count=-1,
        verbose=0,
        random_seed=42 + fold,
        early_stopping_rounds=200,
        allow_writing_files=False,
        max_bin=128,
    )

    cat_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))

    cat_oof[val_idx] = cat_model.predict_proba(X_val)[:, 1]
    cat_pred += cat_model.predict_proba(X_test)[:, 1] / n_splits

    fold_time = time.time() - fold_start
    elapsed_cat = time.time() - cat_start
    avg_per_fold = elapsed_cat / fold
    remaining_cat = avg_per_fold * (n_splits - fold)

    log_step(
        f"Cat Fold {fold}/{n_splits} done in {fold_time/60:.2f} min | "
        f"Est. remaining Cat only: {remaining_cat/60:.2f} min",
        add_units=1,
    )

    del cat_model, X_tr, X_val, y_tr, y_val
    gc.collect()

log(f"CatBoost OOF AUC: {roc_auc_score(y, cat_oof):.6f}")


# ================== 5. TEACHER OOF BLEND (LGB vs CAT) ==================

log("Searching best OOF blend between LGBM and CatBoost...")

r_lgb_oof = rankdata(lgb_oof)
r_cat_oof = rankdata(cat_oof)

best_auc = 0.0
best_w_lgb = 0.5

for w in np.linspace(0.0, 1.0, 21):  # step 0.05
    blend_rank = w * r_lgb_oof + (1.0 - w) * r_cat_oof
    auc = roc_auc_score(y, blend_rank)
    if auc > best_auc:
        best_auc = auc
        best_w_lgb = w

log(
    f"Best OOF teacher weight: LGB={best_w_lgb:.2f}, "
    f"CAT={1-best_w_lgb:.2f} | OOF AUC={best_auc:.6f}"
)

w_lgb = best_w_lgb
w_cat = 1.0 - best_w_lgb

teacher_test_prob = w_lgb * lgb_pred + w_cat * cat_pred


# ================== 6. PSEUDO-LABELING WITH TEACHER ==================

log("Pseudo-labeling high-confidence test samples...")

high_thr = max(0.99, np.percentile(teacher_test_prob, 99.4))
low_thr = min(0.01, np.percentile(teacher_test_prob, 0.6))

mask = (teacher_test_prob > high_thr) | (teacher_test_prob < low_thr)
idx = np.where(mask)[0]
n_raw = len(idx)
max_pseudo = int(0.4 * len(train))
min_pseudo = 500

if n_raw > max_pseudo:
    conf = np.abs(teacher_test_prob[idx] - 0.5)
    top_idx = np.argsort(-conf)[:max_pseudo]
    idx = idx[top_idx]

log(f"Candidate pseudo samples: raw={n_raw}, used={len(idx)}")

if len(idx) >= min_pseudo:
    log("Using pseudo-labels. Training CatBoost student on (train + pseudo)...")
    pl_start = time.time()

    X_big = vstack([X, X_test[idx]])
    y_pseudo = (teacher_test_prob[idx] > 0.5).astype(int)
    y_big = np.concatenate([y, y_pseudo])

    student = CatBoostClassifier(
        iterations=5000,
        learning_rate=0.018,
        depth=8,
        random_seed=2025,
        task_type="CPU",
        thread_count=-1,
        verbose=500,
        allow_writing_files=False,
        max_bin=128,
    )

    student.fit(X_big, y_big)
    student_test_prob = student.predict_proba(X_test)[:, 1]

    pl_time = (time.time() - pl_start) / 60.0
    log_step(f"Student CatBoost training finished in {pl_time:.2f} min", add_units=1)
else:
    log_step("Not enough high-confidence pseudo samples. Using teacher only.", add_units=1)
    student_test_prob = teacher_test_prob


# ================== 7. FINAL RANK BLENDING (LGB vs STUDENT) ==================

log("Final rank blending between LGBM teacher and CatBoost student...")

r_lgb_test = rankdata(lgb_pred)
r_student_test = rankdata(student_test_prob)

alpha_lgb = 0.36
final_rank = alpha_lgb * r_lgb_test + (1.0 - alpha_lgb) * r_student_test
final_rank = final_rank / final_rank.max()


# ================== 8. SAVE SUBMISSION ==================

target_cols = [c for c in sample.columns if c.lower() != "id"]
if len(target_cols) == 0:
    raise ValueError("No target column found in sample_submission!")

target_col = target_cols[0]
sample[target_col] = final_rank

out_file = os.path.join(BASE_DIR, "ULTIMATE_DSCO_PHISH_SUBMISSION.csv")
sample.to_csv(out_file, index=False)

total_minutes = (time.time() - overall_start) / 60.0
log(f"â DONE! Submission saved to: {out_file}")
log(f"Total runtime: {total_minutes:.2f} minutes")
