functions {
#include hmm/forward.stan
#include hmm/backward.stan
}
data {
    int n_cls; // number of hmm components
    int n_obs; // number of observations to classify
    array[n_obs] real y; // observations
    real separation; // prior separation between components
    array[n_cls] vector[n_cls] alpha; // prior on transition probability matrix
}
transformed data {
    vector[n_cls] hyper_mu;
    for (k in 1:n_cls) {
        hyper_mu[k] = separation * (k - n_cls / 2.0 - 0.5);
    }
    vector[n_cls] log_init_prob = rep_vector(-log(n_cls), n_cls);
}
parameters {
    ordered[n_cls] mu;
    array[n_cls] simplex[n_cls] transition_matrix;
}
transformed parameters {
    matrix[n_cls, n_cls] log_transition_matrix;
    array[n_obs] vector[n_cls] emission_log_likelihoods;
    array[n_obs] vector[n_cls] log_alpha; // forward variable

    for (state in 1:n_cls) {
        log_transition_matrix[state,] = to_row_vector(log(transition_matrix[state]));
    }

    for (obs in 1:n_obs) {
        for (state in 1:n_cls) {
            emission_log_likelihoods[obs, state] = normal_lpdf(y[obs] | mu[state], 1.0);
        }
    }

    log_alpha = forward(n_cls, n_obs, log_init_prob, log_transition_matrix, emission_log_likelihoods);
}
model {
    target += log_sum_exp(log_alpha[n_obs]);

    for (state in 1:n_cls) {
        transition_matrix[state] ~ dirichlet(alpha[state]);
        mu[state] ~ normal(hyper_mu[state], 1);
    }
}
generated quantities {
    array[n_obs] vector[n_cls] log_beta; //backward variable
    array[n_obs] simplex[n_cls] smoothing; // p(state_t | y_{1:T})
    array[n_obs] simplex[n_cls] filtering; // p(state_t | y_{1:t})

    log_beta = backward(n_cls, n_obs, log_init_prob, log_transition_matrix, emission_log_likelihoods);

    for (obs in 1:n_obs) {
        smoothing[obs] = softmax(log_alpha[obs] + log_beta[obs]);
        filtering[obs] = softmax(log_alpha[obs]);
    }
}