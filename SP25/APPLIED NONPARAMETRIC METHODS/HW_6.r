# problem 11.3

cat("Problem 11.3\n")

x_1 <- c(3, 5, 8, 11, 15, 18, 20, 25, 27, 30)  
y_1 <- c(2, 4, 7, 10, 17, 23, 29, 45, 59, 73)

# this is a function to determine the thiel slope by making each slope pair-wise and then taking the median
theil_kendall_slope <- function(x, y) {
  n <- length(x)
  slopes <- numeric(0)  
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      slope <- (y[j] - y[i]) / (x[j] - x[i])
      slopes <- c(slopes, slope)
    }
  }
  median(slopes)  
}

slope_1 <- theil_kendall_slope(x_1, y_1)

# this is a function to determine the median intercept by making the intercept for each pair of points and then taking the median
median_intercept <- function(x, y, slope) {
  n <- length(x)
  intercepts <- numeric(0)
  
  for (i in 1:n) {
    intercept <- y[i] - slope * x[i]
    intercepts <- c(intercepts, intercept)
  }
  
  return(median(intercepts))
}

intercept_1 <- median_intercept(x_1, y_1, slope_1)

# this was a pain to figure out, but I finally got it to work
# it just makes a png file so I could see the plot
# There is probably a better way but I couldn't figure it out
png("problem_11_3_plot.png", width=800, height=600)

plot(x_1, y_1, main="Problem 11.3",
     xlab="Days in Storage (x)", ylab="Number of Rotten Oranges (y)",
     pch=19, col="blue", xlim=c(min(x_1)-1, max(x_1)+1), ylim=c(min(y_1)-1, max(y_1)+1))

abline(a=intercept_1, b=slope_1, col="red", lwd=2)

cat("Theil-Kendall Slope Estimate: ", slope_1, "\n")
cat("Intercept: ", intercept_1, "\n")


# problem 11.8

cat("\n\nProblem 11.8\n")

x_2 <- c(25, 50, 100, 150, 155, 187, 200, 237, 287, 290, 300)
y_2 <- c(6.13, 5.51, 6.18, 6.70, 7.22, 7.28, 7.22, 7.48, 7.38, 7.38, 7.64)

# this does the same thing as the previous problem but doesn't take the median of the slopes
slopes_2 <- numeric(0)
for (i in 1:(length(x_2)-1)) {
  for (j in (i+1):length(x_2)) {
    slope <- (y_2[j] - y_2[i]) / (x_2[j] - x_2[i])
    slopes_2 <- c(slopes_2, slope)
  }
}

slope_2 <- median(slopes_2)

# this uses the same intercept formula as before
intercept_2 <- median_intercept(x_2, y_2, slope_2)

png("problem_11_8_plot.png", width=800, height=600)

plot(x_2, y_2, main="Problem 11.3",
     xlab="Days in Storage (x)", ylab="Number of Rotten Oranges (y)",
     pch=19, col="blue", xlim=c(min(x_2)-1, max(x_2)+1), ylim=c(min(y_2)-1, max(y_2)+1))

abline(a=intercept_2, b=slope_2, col="red", lwd=2)

lower_bound <- quantile(slopes_2, 0.025)
upper_bound <- quantile(slopes_2, 0.975)

cat("Theil-Kendall Slope Estimate: ", slope_2, "\n")
cat("Intercept: ", intercept_2, "\n")
cat("95% Confidence Interval: [", lower_bound, ", ", upper_bound, "]\n")