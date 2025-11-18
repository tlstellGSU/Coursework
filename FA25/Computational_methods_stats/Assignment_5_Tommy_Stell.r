set.seed(8670)

# Question 1

generate_noisy_sin <- function(n=100){

    x <- vector("numeric", n)
    y <- vector("numeric", n)
    for (i in 1:n){
        eps_i <- rnorm(1, mean = 0, sd = 0.5**2)
        x[i] <- 10*(i-1)/(n-1)
        y[i] <- sin(x[i]) + eps_i
    }
    return(list(x = x, y = y))

}

average_moving_smoother <- function(x,y,k){
    n <- length(x)
    y_smooth <- vector("numeric", n)
    for (i in 1:n){
        if (i <= k){
            y_smooth[i] <- mean(y[1:(i+k)])
        } else if (i > n - k){
            y_smooth[i] <- mean(y[(i-k):n])
        } else {
            y_smooth[i] <- mean(y[(i-k):(i+k)])
        }
    }
    return(y_smooth)
}

gaussian_kernel_smoother <- function(x,y,h){
    n <- length(x)
    y_smooth <- vector("numeric", n)
    for (i in 1:n){
        weights <- dnorm((x - x[i]) / h)
        weights <- weights / sum(weights)
        y_smooth[i] <- sum(weights * y)
    }
    return(y_smooth)
}

question_1_plot <- function(x,y,y_avg_smooth,y_gauss_smooth, y_true){
    plot(x, y, main = "Smoothing Methods Comparison", xlab = "x", ylab = "y", pch = 16, col = "lightgray")
    lines(x, y_avg_smooth, col = "red", lwd = 2)
    lines(x, y_gauss_smooth, col = "green", lwd = 2)
    lines(x, y_true, col = "black", lwd = 2, lty = 2)
    legend("topright", legend = c("Noisy Data", "True Function", "Average Moving Smoother", "Gaussian Kernel Smoother"),
           col = c("lightgray", "black", "red", "green"), pch = c(16, NA, NA, NA), lty = c(NA, 2, 1, 1), lwd = c(NA, 2, 2, 2))
}

Q1_main_a <- function(){
    data <- generate_noisy_sin(n = 100)
    x <- data$x
    y <- data$y
    y_true <- sin(x)

    y_avg_smooth <- average_moving_smoother(x, y, k = 7)
    y_gauss_smooth <- gaussian_kernel_smoother(x, y, h = 0.4)


    question_1_plot(x, y, y_avg_smooth, y_gauss_smooth, y_true)
}

Q1_main_b <- function(){
    data <- generate_noisy_sin(n = 100)
    x <- data$x
    y <- data$y
    y_true <- sin(x)

    y_avg_smooth_3 <- average_moving_smoother(x, y, k = 3)
    y_avg_smooth_11 <- average_moving_smoother(x, y, k = 11)
    y_gauss_smooth_0.2 <- gaussian_kernel_smoother(x, y, h = 0.2)
    y_gauss_smooth_0.8 <- gaussian_kernel_smoother(x, y, h = 0.8)

    plot(x,y, main = "Smoothing Methods Comparison with Increased Smoothing Parameters", xlab = "x", ylab = "y", pch = 16, col = "lightgray")
    lines(x, y_true, col = "black", lwd = 2, lty = 2)
    lines(x, y_avg_smooth_3, col = "red", lwd = 2)
    lines(x, y_avg_smooth_11, col = "orange", lwd = 2)
    lines(x, y_gauss_smooth_0.2, col = "green", lwd = 2)
    lines(x, y_gauss_smooth_0.8, col = "purple", lwd = 2)
    legend("topright", legend = c("Noisy Data", "True Function", "Avg Moving Smoother k=3", "Avg Moving Smoother k=11", "Gauss Kernel h=0.2", "Gauss Kernel h=0.8"),
           col = c("lightgray", "black", "red", "orange", "green", "purple"), pch = c(16, NA, NA, NA, NA, NA), lty = c(NA, 2, 1, 1, 1, 1), lwd = c(NA, 2, 2, 2, 2, 2))
}

Q1_main_a()
Q1_main_b()

# Question 2

LOESS_smooting <- function(x,y,span){
    loess_model <- loess(y ~ x, span = span)
    y_smooth <- predict(loess_model, newdata = data.frame(x = x))
    return(y_smooth)
}

MSE_error <- funtion(y_true, y_pred){
    return(mean((y_true - y_pred)^2))
}

Q2_main_a <- function(){
    data <- generate_noisy_sin(n = 100)
    x <- data$x
    y <- data$y
    y_true <- sin(x)

    spans_to_try <- c(0.1, 0.3, 0.6)

    plot(x, y, main = "LOESS Smoothing with Different Spans", xlab = "x", ylab = "y", pch = 16, col = "lightgray")

    for (span in spans_to_try){
        y_loess_smooth <- LOESS_smooting(x, y, span)
        mse <- MSE_error(y_true, y_loess_smooth)
        cat("Span:", span, "MSE:", mse, "\n")
        lines(x, y_loess_smooth, lwd = 2, col = rainbow(length(spans_to_try))[which(spans_to_try == span)])
    }

    lines(x, y_true, col = "black", lwd = 2, lty = 2)
    legend("topright", legend = c("Noisy Data", "True Function", paste("LOESS Span =", spans_to_try)),
           col = c("lightgray", "black", rainbow(length(spans_to_try))), pch = c(16, NA, rep(NA, length(spans_to_try))), lty = c(NA, 2, rep(1, length(spans_to_try))), lwd = c(NA, 2, rep(2, length(spans_to_try))))
}

Q2_main_b <- function(){
    data <- generate_noisy_sin(n = 100)
    x <- data$x
    y <- data$y
    y_true <- sin(x)

    bandwidths_to_try <- c(0.2, 0.4, 0.8)

    plot(x, y, main = "Gaussian Kernel Smoothing with Different Bandwidths", xlab = "x", ylab = "y", pch = 16, col = "lightgray")

    for (h in bandwidths_to_try){
        y_kernel_smooth <- gaussian_kernel_smoother(x, y, h)
        mse <- MSE_error(y_true, y_kernel_smooth)
        cat("Bandwidth:", h, "MSE:", mse, "\n")
        lines(x, y_kernel_smooth, lwd = 2, col = rainbow(length(bandwidths_to_try))[which(bandwidths_to_try == h)])
    }

    lines(x, y_true, col = "black", lwd = 2, lty = 2)
    legend("topright", legend = c("Noisy Data", "True Function", paste("Gaussian Kernel h =", bandwidths_to_try)),
           col = c("lightgray", "black", rainbow(length(bandwidths_to_try))), pch = c(16, NA, rep(NA, length(bandwidths_to_try))), lty = c(NA, 2, rep(1, length(bandwidths_to_try))), lwd = c(NA, 2, rep(2, length(bandwidths_to_try))))
}

Q2_main_a()
Q2_main_b()

# Question 3

# Let X ~ 0.6 N(0,1^2) + 0.4 N(3, 0.5^2)

Q3_main_a <- function(){
    n <- 500

    x_sim <- 0.6 * rnorm(n,mean = 0, sd=1) + 0.4 * rnorm(n, mean = 3, sd = 0.5)

    hist(x_sim, breaks = 30, probability = TRUE, main = "Histogram of Simulated Data from Mixture Distribution", xlab = "x", col = "lightblue")

    kernels_to_try <- c("gaussian", "epanechnikov", "rectangular")

    bandwidth <- 0.4

    plot(density(x_sim, bw = bandwidth, kernel = "gaussian"), main = "Kernel Density Estimates with Different Kernels", xlab = "x", lwd = 2, col = rainbow(length(kernels_to_try))[1])

    for (kernel in kernels_to_try){
        density_estimate <- density(x_sim, bw = bandwidth, kernel = kernel)
        lines(density_estimate, lwd = 2, col = rainbow(length(kernels_to_try))[which(kernels_to_try == kernel)])
    }
    legend("topright", legend = kernels_to_try, col = rainbow(length(kernels_to_try)), lwd = 2)
}

Q3_main_b <- function(){
    n <- 500

    x_sim <- 0.6 * rnorm(n,mean = 0, sd=1) + 0.4 * rnorm(n, mean = 3, sd = 0.5)

    hist(x_sim, breaks = 30, probability = TRUE, main = "Histogram of Simulated Data from Mixture Distribution", xlab = "x", col = "lightblue")

    bandwidths_to_try <- c(0.2, 0.4, 0.8)

    kernel <- "gaussian"

    plot(density(x_sim, bw = bandwidths_to_try[1], kernel = kernel), main = "Kernel Density Estimates with Different Bandwidths", xlab = "x", lwd = 2, col = rainbow(length(bandwidths_to_try))[1])

    for (h in bandwidths_to_try){
        density_estimate <- density(x_sim, bw = h, kernel = kernel)
        lines(density_estimate, lwd = 2, col = rainbow(length(bandwidths_to_try))[which(bandwidths_to_try == h)])
    }
    legend("topright", legend = paste("Bandwidth =", bandwidths_to_try), col = rainbow(length(bandwidths_to_try)), lwd = 2)
}

Q3_main_a()
Q3_main_b()

