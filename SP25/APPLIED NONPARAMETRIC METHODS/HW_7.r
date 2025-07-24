#problem 11.7

tractors <- c(9.2, 30.9, 51.8, 72.7, 89.9, 88.7, 90.2)
horses <- c(91.8, 88, 80.6, 43.6, 16.7, 14.4, 10.5)

png("problem_11_7_plot.png", width=800, height=600)

plot(tractors, horses, main="Problem 11.7",
     xlab="Tractors", ylab="Horses",
     pch=19, col="blue", xlim=c(min(tractors)-1, max(tractors)+1), ylim=c(min(horses)-1, max(horses)+1))

pearson_corr <- cor(tractors, horses, method = "pearson")

kendall_corr <- cor(tractors, horses, method = "kendall")

cat("Pearson Correlation Coefficient:", pearson_corr, "\n")
cat("Kendall Correlation Coefficient:", kendall_corr, "\n")

tractors_neg = -tractors

iso_fit <- isoreg(tractors_neg, horses)

iso_fit$yf <- rev(iso_fit$yf)

png("problem_11_7_b_plot.png", width=800, height=600)
plot(tractors_neg, horses, main="Isotonic Regression", pch=16, col="blue")
smoothing_seq <- seq(min(tractors_neg), max(tractors_neg), length.out=100)
smoothing_values <- approx(tractors_neg, iso_fit$yf, xout = smoothing_seq)$y

lines(smoothing_seq, smoothing_values, col="red", lwd=2)  # Smoothed line
