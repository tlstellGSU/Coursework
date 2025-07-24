# FINAL EXAM TAKE HOME

# problem 1

# part a - KW test

print("Problem 1")

c_1_1 <- c(28, 23, 26, 31, 14, 29)
c_1_2 <- c(21, 31, 32, 19, 27, 16)
c_1_3 <- c(24, 17, 21, 31, 22, 18)

values <- c(c_1_1, c_1_2, c_1_3)

groups <- factor(c(rep("Group1", length(c_1_1)), 
                   rep("Group2", length(c_1_2)), 
                   rep("Group3", length(c_1_3))))

kruskal_result <- kruskal.test(values ~ groups)

print("Problem 1.a: Kruskal-Wallis Test Result")
print(kruskal_result)

# part b - ANOVA

anova_result <- aov(values ~ groups)

print("Problem 1.b: ANOVA Test Result")
print(summary(anova_result))

# problem 2

print("Problem 2")

age_2 <- c(6, 3, 5, 5, 1)
price_2 <- c(2.8, 1.9, 1.7, 2.4, 1.6)

result_2 = cor.test(age_2, price_2, method = "spearman", alternative = "greater")

print("Problem 2: Spearman Correlation Test Result")
print(result_2)

rank_age <- rank(age_2)
rank_price <- rank(price_2)
rho <- cor(rank_age, rank_price, method = "spearman")
slope <- rho * (sd(price_2) / sd(age_2))

intercept <- mean(price_2) - slope * mean(age_2)

plot(age_2, price_2, 
     main = "Scatterplot with Spearman Rank-Based Regression Line",
     xlab = "Age", ylab = "Price", pch = 19, col = "darkblue")
abline(a = intercept, b = slope, col = "red", lwd = 2)
legend("topright", legend = "Rank-Based Fit", col = "red", lwd = 2)

# problem 3

print("Problem 3")

data_3 <- c(56, 44, 79, 72, 70, 54, 94, 85, 33, 65)

function_3 <- function(t)
    {
     3*(1-(t^2)/5)/(4*5^0.5)* (abs(t) <= sqrt(5))
    }

h_3 <- 5

x_grid <- seq(min(data_3) - 10, max(data_3) + 10, length.out = length(data_3))

density_est <- sapply(x_grid, function(x) {
  mean(sapply(data_3, function(xi) function_3((x - xi) / h_3))) / h_3
})

dx <- diff(x_grid)[1]  
cdf <- cumsum(density_est) * dx

lower_index <- which.min(abs(cdf - 0.05))
upper_index <- which.min(abs(cdf - 0.95))

lower_bound <- x_grid[lower_index]
upper_bound <- x_grid[upper_index]

cat("90% CI from custom kernel:\n")
cat(sprintf("[%.2f, %.2f]\n", lower_bound, upper_bound))


plot(x_grid, density_est, type = "l", main = "Custom Kernel Density with 90% CI", xlab = "Value", ylab = "Density")
abline(v = c(lower_bound, upper_bound), col = "red", lty = 2)
polygon(x_grid[x_grid >= lower_bound & x_grid <= upper_bound],
        density_est[x_grid >= lower_bound & x_grid <= upper_bound],
        col = rgb(1, 0, 0, 0.3), border = NA)





print("In class problems")




# problem 1

data_1 <- c(2.8, 8.3, 4.3, 2.2, 1.8, 1.2, 5.5, 8.5, 9.1, 9.5)

F <-function(x) { x/10}

result_2_1 <- ks.test(data_1, F, alternative = "two.sided", exact = TRUE)
print(result_2_1)


result_1 <- ks.test(data_1, 0, alternative = "l", exact = TRUE)
result_1_1 <- ks.test(data_1, 10, alternative = "g", exact = FALSE)

print(result_1)
print(result_1_1)

# problem 2



# problem 3

data_3_a <- c(11.7, 10.0, 10.8, 11.1, 12.9)
data_3_b <- c(11.5, 12.8, 13.8, 13.6, 15.5, 12.4)

result_3_1 <- wilcox.test(data_3_a, data_3_b, ,alternative = "t", conf.int=TRUE, exact = FALSE)
print(result_3_1)

# problem 4


data_4_hours <- c(9,5,11,13,10,5,18,15,2,8)
data_4_grades <- c(56, 44, 79, 72, 70, 54, 94, 85, 33, 65)

result_4 <- cor.test(data_4_hours, data_4_grades, method = "spearman", alternative = "greater")
print(result_4)

