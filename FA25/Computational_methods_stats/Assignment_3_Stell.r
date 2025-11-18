# Question 1

# 1.c)

theta_true <- exp(-2) - exp(-4)

m_samples <- c(1e2, 1e3, 1e4, 1e5)

theta_sample <- function(m){
    
    running_sum <- 0

    for (i in 1:m){
        set.seed(i)
        x <- runif(1, 2, 4)
        running_sum <- running_sum + exp(-x)
    }

    return((4-2)*running_sum / m)
}

m_sampled_values <- numeric(length(m_samples))

for (i in seq_along(m_samples)){
    m <- m_samples[i]
    m_sampled_values[i] <- theta_sample(m)
}

m_sampled_errors <- abs(m_sampled_values - theta_true)
results <- data.frame(m = m_samples, theta_sampled = m_sampled_values, error = m_sampled_errors)
print(results)

plot(log10(results$m), log10(results$error), type='b', xlab='log10(m)', ylab='log10(error)', main='Log-Log plot of Error vs m')
abline(lm(log10(results$error) ~ log10(results$m)), col='red')

slope <- coef(lm(log10(results$error) ~ log10(results$m)))[2]
cat("Slope of the line in log-log plot:", slope, "\n")

# Question 3

# 3.c)

function_3 <- function(x){
    return(exp(-x))
}

true_3 <- exp(0) - exp(-1)

m_3 <- 1e3
R_3 <- 1e2

# simple MC

simple_MC_3 <- function(m){
    running_sum <- 0

    for (i in 1:m){
        x <- runif(1, 0, 1)
        running_sum <- running_sum + function_3(x)
    }

    return(running_sum / m)
}

simple_MC_values <- simple_MC_3(m_3)

# antithetic MC

antithetic_MC_3 <- function(m){
    running_sum <- 0

    for (i in 1:(m/2)){
        u <- runif(1, 0, 1)
        running_sum <- running_sum + (function_3(1-u) + function_3(u))/2
    }

    return(running_sum / m)
}

antithetic_MC_values <- antithetic_MC_3(m_3)

# generate sample variances

simple_MC_variances <- numeric(R_3)
antithetic_MC_variances <- numeric(R_3)

for (r in 1:R_3) {
    simple_MC_variances[r] <- var(replicate(100, simple_MC_3(m_3)))
    antithetic_MC_variances[r] <- var(replicate(100, antithetic_MC_3(m_3)))
}
cat("Simple MC estimate:", simple_MC_values, "with variance:", mean(simple_MC_variances), "\n")
cat("Antithetic MC estimate:", antithetic_MC_values, "with variance:", mean(antithetic_MC_variances), "\n")