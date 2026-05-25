import numpy as np
from sklearn.metrics import auc

# =========================================================
# PEHE
# =========================================================

def pehe(tau_hat, tau_true):
    tau_hat = np.asarray(tau_hat).flatten()
    tau_true = np.asarray(tau_true).flatten()
    return np.sqrt(np.mean((tau_hat - tau_true) ** 2))

# =========================================================
# POLICY VALUE / RISK
# =========================================================

def policy_risk(y, t, pi, K = 2):
    y = np.asarray(y).flatten()
    t = np.asarray(t).flatten()
    pi = np.asarray(pi).flatten()
    policy_value = 0
    for i in range(1, K+1):
        Pi = (pi == i)
        T = (t == i)
        subset = Pi & T
        avg_outcome = y[subset].mean()
        p_i = Pi.sum()/len(y)
        policy_value += avg_outcome * p_i
    return 1 - policy_value
        
# =========================================================
# UPLIFT CURVE (GAIN CURVE)
# =========================================================

def uplift_curve(tau_hat, t, y):
    ## Gain(k) = n_k * (E[Y|T=1] - E[Y|T=0])
    tau_hat = np.asarray(tau_hat).flatten()
    t = np.asarray(t).flatten()
    y = np.asarray(y).flatten()
    
    order = np.argsort(-tau_hat)
    
    t_sorted = t[order]
    y_sorted = y[order]
    
    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1 - t_sorted)
    
    # avoid divide-by-zero
    cum_t = np.clip(cum_t, 1, None)
    cum_c = np.clip(cum_c, 1, None)

    cum_y_t = np.cumsum(y_sorted * t_sorted)
    cum_y_c = np.cumsum(y_sorted * (1 - t_sorted))
    
    uplift = cum_y_t / cum_t - cum_y_c / cum_c
    cumulative_gain = uplift * np.arange(1, len(y) + 1)
    population_share =  np.arange(1, len(y)+1) / len(y)
    return population_share, cumulative_gain

# =========================================================
# AUUC
# =========================================================

def uplift_auc_score(tau_hat, t, y, normalize=False, perfect_tau=None,):
    # 1. 模型曲线
    population_share, gain_curve =  uplift_curve(tau_hat, t, y)
    auc_model = auc(population_share, gain_curve)
    
    # 2. 随机曲线
    random_curve = gain_curve[-1] * population_share
    auc_random = auc(population_share, random_curve)
    
    auuc = auc_model - auc_random
    if not normalize:
        return auuc
    
    # 3. 完美曲线
    if perfect_tau is None:
        # pseudo-perfect heuristic
        perfect_tau = y * (2 * t - 1)

    population_share, perfect_curve =  uplift_curve(perfect_tau, t, y)
    auc_perfect = auc(population_share, perfect_curve)
    
    # 4. 标准化
    denom = auc_perfect - auc_random
    if np.abs(denom) < 1e-10:
            return 0.0
    return auuc / denom


def qini_curve_industry(tau_hat, t, y):
    tau_hat = np.asarray(tau_hat).flatten()
    t = np.asarray(t).flatten()
    y = np.asarray(y).flatten()
    
    order = np.argsort(-tau_hat)
    t_sorted = t[order]
    y_sorted = y[order]
    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1-t_sorted)
    
    cum_t = np.clip(cum_t, 1, None)
    cum_c = np.clip(cum_c, 1, None)
    
    cum_y_t = np.cumsum(y_sorted * t_sorted)
    cum_y_c = np.cumsum(y_sorted * (1 - t_sorted))
    
    qini = cum_y_t - cum_y_c * (cum_t / cum_c)
    population_share =  np.arange(1, len(y)+1) / len(y)
    return population_share, qini


def qini_auc_score_industry(tau_hat, t, y, normalize=False, perfect_tau=None):
    population_share, qini_curve = qini_curve_industry(tau_hat, t, y)
    auc_model = auc(population_share, qini_curve)
    random_curve = qini_curve[-1] * population_share
    auc_random = auc(population_share, random_curve)
    
    qini_auc = auc_model - auc_random
    if not normalize:
        return qini_auc
    
    if perfect_tau is None:
        perfect_tau = y * (2 * t - 1)
    population_share, perfect_qini =  qini_curve_industry(perfect_tau, t, y)
    auc_perfect = auc(population_share, perfect_qini)
    denom = auc_perfect - auc_random
    if np.abs(denom) < 1e-10:
        return 0.0
    return qini_auc / denom