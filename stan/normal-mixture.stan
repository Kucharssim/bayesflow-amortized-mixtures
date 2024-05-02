data {
    int n_cls; // number of mixture components
    int n_obs; // number of observations to classify
    int n_rep; // number of 'replications' for each observation
    array[n_obs,n_rep] real y; // observations
    real separation; // prior separation between components
    vector[n_cls] alpha; // prior on mixture proportions
}
transformed data {
    vector[n_cls] hyper_mu;
    for (k in 1:n_cls) {
        hyper_mu[k] = separation * (k - n_cls / 2.0 - 0.5);
    }
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
            for (j in 1:n_rep) {
                log_probs[i,k] += normal_lpdf(y[i,j] | mu[k], 1.0);
            }
        }
        cumulative_log_probs += log_sum_exp(log_probs[i]);
    }
}
model {
    // priors
    target += dirichlet_lpdf(p | alpha);
    target += normal_lpdf(mu | hyper_mu, 1.0);

    // likelihood
    target += cumulative_log_probs;
}
generated quantities {
    array[n_obs] simplex[n_cls] class_membership;

    for (i in 1:n_obs) {
        class_membership[i] = softmax(log_probs[i]);
    }
}