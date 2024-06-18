import os, sys
# for accessing src, stan, etc.
sys.path.append(os.path.abspath(os.path.join("../..")))


from src.models.NormalHmm import NormalHmm


model = NormalHmm(n_cls = 2, n_obs=[100, 100], separation=3.0, alpha=0.5)
model.prior_means = [0.0, 0.0, -1.5, 1.00]
model.prior_sds = [3.12,  3.12, 1.0, 0.65]