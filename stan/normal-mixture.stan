data {
    int n_obs;
    int n_cls;
    int n_rep;
    array[n_obs] real y;
    real prior_scale;
    vector[n_cls] alpha;
}
parameters {
    simplex[n_cls] p;
    ordered[n_cls] mu;
}
transformed parameters {
    array[n_obs] vector[n_cls] log_probs;
    real cumulative_log_probs = 0.0;

    for (i in 1:n_obs) {
        log_probs[i] = log(p);
        for (k in 1:n_cls) {
            log_probs[i,k] += normal_lpdf(y[i] | mu[k], 1.0);
        }
        cumulative_log_probs += log_sum_exp(log_probs[i]);
    }
}
model {
    // priors
    target += dirichlet_lpdf(p | alpha);
    target += normal_lpdf(mu | 0.0, prior_scale);

    // likelihood
    target += cumulative_log_probs;
}
generated quantities {
    array[n_obs] simplex[n_cls] class_membership;

    for (i in 1:n_obs) {
        class_membership[i] = softmax(log_probs[i]);
    }
}