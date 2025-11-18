# Assignment 4 - Tommy Stell

# Question 2_0

# the model: Y = beta0 + beta1*X1 + beta2*X^2 + epsilon
# epislon ~ N(0, sigma^2)
# x_i ~ Unif(-2,2)
# beta0 = 1, beta1 = 2, beta2 = -1

Q2_model <- function(n, beta0 = 1, beta1 = 2, beta2 = -1, sigma = 1) {
    X1 <- runif(n, min = -2, max = 2)
    epsilon <- rnorm(n, mean = 0, sd = sigma)
    Y <- beta0 + beta1 * X1 + beta2 * (X1^2) + epsilon
    data.frame(X1 = X1, Y = Y)
}

# question 2.a

Q2_observations <- Q2_model(60)

Q2_poly_degree <- c(1, 2, 3, 4, 5, 6)

Q2_CV <- function(data, degrees, k = 5) {
    n <- nrow(data)
    folds <- sample(rep(1:k, length.out = n))
    cv_errors <- numeric(length(degrees))

    for (d in seq_along(degrees)) {
        degree <- degrees[d]
        fold_errors <- numeric(k)

        for (fold in 1:k) {
            train_data <- data[folds != fold, ]
            test_data <- data[folds == fold, ]

            model <- lm(Y ~ poly(X1, degree), data = train_data)
            predictions <- predict(model, newdata = test_data)
            fold_errors[fold] <- mean((test_data$Y - predictions)^2)
        }

        cv_errors[d] <- mean(fold_errors)
    }

    names(cv_errors) <- paste0("Degree_", degrees)
    return(cv_errors)
}

# Question 2.b

Q2_plot_CV_errors <- function(cv_errors) {
    plot(as.numeric(sub("Degree_", "", names(cv_errors))), cv_errors,
        type = "b",
        xlab = "Polynomial Degree", ylab = "Cross-Validation MSE",
        main = "CV MSE vs Polynomial Degree"
    )
}

# Question 2.c

Q2_train_test_split <- function(data, train_frac = 0.7) {
    n <- nrow(data)
    train_indices <- sample(1:n, size = floor(train_frac * n))
    train_data <- data[train_indices, ]
    test_data <- data[-train_indices, ]
    list(train = train_data, test = test_data)
}

Q2_MSE <- function(train_data, test_data, degree) {
    model <- lm(Y ~ poly(X1, degree), data = train_data)
    predictions <- predict(model, newdata = test_data)
    mean((test_data$Y - predictions)^2)
}

Q2_main <- function() {
    data <- Q2_model(60)

    cv_errors <- Q2_CV(data, Q2_poly_degree)
    print(cv_errors)

    Q2_plot_CV_errors(cv_errors)

    splits <- Q2_train_test_split(data)
    train_data <- splits$train
    test_data <- splits$test

    test_MSEs <- sapply(Q2_poly_degree, function(d) {
        Q2_MSE(train_data, test_data, d)
    })

    names(test_MSEs) <- paste0("Degree_", Q2_poly_degree)
    print(test_MSEs)
}

Q2_main()

# question 3.b

Q3_n <- 40
Q3_p <- 3
Q3_sigma <- 1

Q3_X <- matrix(rnorm(Q3_n * Q3_p), nrow = Q3_n, ncol = Q3_p)
Q3_data <- data.frame(y = rnorm(Q3_n, sd = Q3_sigma), Q3_X)

Q3_fit_model <- function(data) {
    model <- lm(y ~ ., data = data)
    summary(model)
}

Q3_X_matrix <- model.matrix(lm(y ~ ., data = Q3_data))

Q3_H <- Q3_X_matrix %*% solve(t(Q3_X_matrix) %*% Q3_X_matrix) %*% t(Q3_X_matrix)

h <- diag(Q3_H)

Q3_shortcut_LOOCV <- function(data, h) {
    n <- nrow(data)
    model <- lm(y ~ ., data = data)
    residuals <- resid(model)
    loo_cv_errors <- (residuals / (1 - h))^2
    mean(loo_cv_errors)
}

Q3_refitting_LOOCV <- function(data) {
    n <- nrow(data)
    loo_cv_errors <- numeric(n)

    for (i in 1:n) {
        train_data <- data[-i, ]
        test_data <- data[i, , drop = FALSE]

        model <- lm(y ~ ., data = train_data)
        prediction <- predict(model, newdata = test_data)

        loo_cv_errors[i] <- (test_data$y - prediction)^2
    }

    mean(loo_cv_errors)
}

Q3_main <- function() {
    fit_summary <- Q3_fit_model(Q3_data)
    print(fit_summary)

    time_shortcut_start <- Sys.time()
    loocv_shortcut_mse <- Q3_shortcut_LOOCV(Q3_data, h)
    time_shortcut_end <- Sys.time()
    time_refit_start <- Sys.time()
    loocv_refit_mse <- Q3_refitting_LOOCV(Q3_data)
    time_refit_end <- Sys.time()

    absolute_difference <- abs(loocv_shortcut_mse - loocv_refit_mse)

    print(paste("LOOCV MSE (Shortcut):", loocv_shortcut_mse))
    print(paste("LOOCV MSE (Refitting):", loocv_refit_mse))
    print(paste("Absolute Difference:", absolute_difference))
    print(paste("Time taken (Shortcut):", time_shortcut_end - time_shortcut_start))
    print(paste("Time taken (Refitting):", time_refit_end - time_refit_start))
}

Q3_main()

# question 4.b

# X_1, ... X_n ~ N(0,1)
# estimate the error of the sample median

Q4_sample_median_error <- function(n = 50, B = 2000) {
    medians <- numeric(B)

    for (b in 1:B) {
        sample_data <- rnorm(n)
        medians[b] <- median(sample_data)
    }

    median_estimate <- mean(medians)
    temp <- mean((medians - median_estimate)^2)

    MSE_bootstrap <- ((1/(B-1)) * sum((medians - median_estimate)^2))^0.5

    list(median_estimate = median_estimate, mse = MSE_bootstrap)
}

Q4_compute_theoretical <- function(n = 50, mu = 0, sigma = 1) {
    f_mu <- 1 / (2 * pi)^0.5
    theoretical <- 1 / (2 * (n^0.5) * f_mu)
    return(theoretical)
}

Q4_compare_errors <- function() {
    n <- 50
    B <- 2000

    sim_results <- Q4_sample_median_error(n, B)
    theoretical_mse <- Q4_compute_theoretical(n)

    print(paste("Simulated Median Estimate:", sim_results$median_estimate))
    print(paste("Simulated MSE:", sim_results$mse))
    print(paste("Theoretical MSE:", theoretical_mse))
}

Q4_compare_errors()

# question 5

# theta_hat = max(X_1, ... X_n), X_i ~ Unif(0, 1)

# question 5.b

Q5_n <- 30

Q5_true_theta <- function(n) {
    n / (n + 1)
}

Q5_true_var <- function(n) {
    n / ((n + 1)^2 * (n + 2))
}

Q5_simulate_theta_hat <- function(n, B = 10000) {
    theta_hats <- numeric(B)

    for (b in 1:B) {
        sample_data <- runif(n, min = 0, max = 1)
        theta_hats[b] <- max(sample_data)
    }

    mean_theta_hat <- mean(theta_hats)
    var_theta_hat <- var(theta_hats)

    list(mean_theta_hat = mean_theta_hat, var_theta_hat = var_theta_hat)
}

Q5_main <- function() {
    n <- Q5_n
    B <- 10000

    sim_results <- Q5_simulate_theta_hat(n, B)
    true_theta <- Q5_true_theta(n)
    true_var <- Q5_true_var(n)

    print(paste("Simulated E[theta_hat]:", sim_results$mean_theta_hat))
    print(paste("True E[theta_hat]:", true_theta))
    print(paste("Simulated Var(theta_hat):", sim_results$var_theta_hat))
    print(paste("True Var(theta_hat):", true_var))
}

Q5_main()
