functions {
#include hmm/forward.stan
#include hmm/backward.stan
}
data {
    int n_obs; // number of observations (timepoints) to classify
    int n_rep; // number of repetitions per timepoint
    array[n_obs, n_rep] real y; // observations
    array[n_obs, n_rep] int valid; // indicator whether the given value in `y` is valid or missing
}
transformed data {
    int n_cls = 2;
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
            emission_log_likelihoods[obs, state] = 0.0;
            for (rep in 1:n_rep) {
                if (valid[obs,rep]) {
                    emission_log_likelihoods[obs, state] += normal_lpdf(y[obs, rep] | mu[state], 1.0);
                }
            }
        }
    }

    log_alpha = forward(n_cls, n_obs, log_init_prob, log_transition_matrix, emission_log_likelihoods);
}
model {
    target += log_sum_exp(log_alpha[n_obs]);

    
    target += normal_lpdf(mu[1] | -1.0, 1);
    target += normal_lpdf(mu[2] |  1.0, 1);
    target += dirichlet_lpdf(transition_matrix[1] | [2, 2]);
    target += dirichlet_lpdf(transition_matrix[2] | [2, 2]);
}
generated quantities {
    array[n_obs] vector[n_cls] log_beta; //backward variable
    array[n_obs] simplex[n_cls] smoothing; // p(state_t | y_{1:T})
    array[n_obs] simplex[n_cls] filtering; // p(state_t | y_{1:t})
    array[n_obs] simplex[n_cls] backward_filtering; // p(state_t | y_{t+1:T})

    log_beta = backward(n_cls, n_obs, log_init_prob, log_transition_matrix, emission_log_likelihoods);

    for (obs in 1:n_obs) {
        smoothing[obs]          = softmax(log_alpha[obs] + log_beta[obs]);
        filtering[obs]          = softmax(log_alpha[obs]);
        backward_filtering[obs] = softmax(log_beta[obs]);
    }
}