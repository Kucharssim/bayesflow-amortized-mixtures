functions {
    // Forward algorithm: log(alpha[t, k]) = log p(state_t = k | x_{1:t})
    array[] vector forward(int K, int T, vector log_initial, matrix log_transition, array[] vector emission_log_likelihoods) {
        array[T] vector[K] log_alpha;

        log_alpha[1] = log_initial + emission_log_likelihoods[1];

        for (t in 2:T) {
            for (j in 1:K) {
                vector[K] log_current = log_alpha[t-1] + log_transition[:,j] + emission_log_likelihoods[t, j];
                log_alpha[t,j] = log_sum_exp(log_current);
            }
        }

        return log_alpha;
    }
    // Backward algorithm: log(beta[t, k]) = log p(state_t = k | x_{t:T})
    array[] vector backward(int K, int T, vector log_initial, matrix log_transition, array[] vector emission_log_likelihoods) {
        array[T] vector[K] log_beta;

        log_beta[T] = rep_vector(0.0, K);

        for(tback in 1:(T-1)){
            int t = T-tback;

            for(j in 1:K){
                vector[K] log_current = log_beta[t+1] + log_transition[:,j] + emission_log_likelihoods[t+1, j];
                log_beta[t, j] = log_sum_exp(log_current);
            }
        }

        return log_beta;
    }

}
data {
    int n_obs; // number of observations to classify
    array[n_obs] real y; // observations
    array[2] vector[2] alpha; // prior on transition probability matrix
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
            emission_log_likelihoods[obs, state] = normal_lpdf(y[obs] | mu[state], 1.0);
        }
    }

    log_alpha = forward(n_cls, n_obs, log_init_prob, log_transition_matrix, emission_log_likelihoods);
}
model {
    target += log_sum_exp(log_alpha[n_obs]);

    
    target += normal_lpdf(mu[1] | -1.5, 1);
    target += normal_lpdf(mu[2] |  1.5, 1);
    target += dirichlet_lpdf(transition_matrix[1] | [2, 2]);
    target += dirichlet_lpdf(transition_matrix[2] | [2, 2]);
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