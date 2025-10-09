# Assignment 1 for STAT 8670 - due 9/12/2025
# all work by Tommy Stell

# loading packages/libraries
options(repos = c(CRAN = "https://cloud.r-project.org/"))

install.packages("pandoc")

library(microbenchmark)
library(tidyverse)
library(ggplot2)

# Question 1 - Data Analysis

# loading the iris dataset
Q1_data <- iris

# 1.a) five summary statistics using base R
summary(Q1_data)

# 1.b) five summary statistics using tidyverse
Q1_data %>%
  summarise(
    Sepal.Length = list(summary(Sepal.Length)),
    Sepal.Width = list(summary(Sepal.Width)),
    Petal.Length = list(summary(Petal.Length)),
    Petal.Width = list(summary(Petal.Width)),
    Species = list(summary(Species))
  )

# 1.c) use ggplot2 to create a scatter plot of Sepal.Length vs Sepal.Width

ggplot(Q1_data, aes(x = Sepal.Length, y = Sepal.Width)) +
  geom_point() +
  labs(title = "Sepal.Length vs Sepal.Width",
       x = "Sepal Length",
       y = "Sepal Width")

ggsave("Assignment_1_1c_scatter_plot.png")

# 1.d) fit a linear regression with Length as response and Width as predictor
lm_model <- lm(Sepal.Length ~ Sepal.Width, data = Q1_data)
summary(lm_model)

# 1.e) add fitted regression line to the plot from part c

ggplot(Q1_data, aes(x = Sepal.Length, y = Sepal.Width)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  labs(title = "Sepal.Length vs Sepal.Width",
       x = "Sepal Length",
       y = "Sepal Width")

ggsave("Assignment_1_1e_scatter_plot_with_regression_line.png")

# 1.f) use different colors to distinguish species in the plot
ggplot(Q1_data, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) +
  geom_point() +
  #geom_smooth(method = "lm", se = FALSE) +
  labs(title = "Sepal.Length vs Sepal.Width",
       x = "Sepal Length",
       y = "Sepal Width")

ggsave("Assignment_1_1f_scatter_plot_with_regression_line.png")


# Problem 2 - conceptual

# 2.c)

set.seed(777) # I personally always use 137 as my set seed for anything like this since its tattooed on my arm

X <- matrix(rnorm(1000*100), 1000, 100)
y <- rnorm(1000)

normal_eqn <- function(X, y) {
  solve(t(X) %*% X) %*% t(X) %*% y
}

g_elim <- function(X,y){
    XtX <- t(X) %*% X
    Xty <- t(X) %*% y
    solve(XtX, Xty)
}

qr_decomp <- function(X,y){
    qr_X <- qr(X)
    qr.coef(qr_X, y)
}

benchmark_results <- microbenchmark(
  normal_eqn(X, y),
  g_elim(X, y),
  qr_decomp(X, y),
  times = 100
)

print(benchmark_results)

A_1_2_c_plot <- autoplot(benchmark_results)

ggsave("Assignment_1_2c_benchmark_plot.png")