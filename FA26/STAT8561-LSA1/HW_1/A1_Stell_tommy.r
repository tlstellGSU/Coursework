# Question 4

# 4.a

data <- data(cars)

fit <- lm(dist ~ speed, data = cars)
summary(fit)

lm(formula = dist ~ speed, data = cars)

# 4.b

Y <- matrix(cars$dist, ncol = 1)
Y

X <- cbind(
  1,
  cars$speed
)
X

# X^T X
XtX <- t(X) %*% X
XtX

# X^T Y
XtY <- t(X) %*% Y
XtY

beta_hat <- solve(t(X) %*% X) %*% t(X) %*% Y

beta_hat
