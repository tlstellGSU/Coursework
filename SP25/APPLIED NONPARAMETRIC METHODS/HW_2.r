# problem 3.14

print("Problem 3.14")
data_1 <- c(1175, 1183, 1327, 1581, 1592, 1624, 1777, 1924, 2483, 2642, 2713, 3419, 5350, 7615)

median_2005 <- 1579

walsh <- function (x) 
{
    n <- length(x)
    w <- vector(n * (n + 1)/2, mode = "numeric")
    ind <- 0
    for (i in 1:n) {
        for (j in i:n) {
            ind <- ind + 1
            w[ind] <- 0.5 * (x[i] + x[j])
        }
    }
    return(w)
  }


results_1_1 <- wilcox.test(data_1, mu = median_2005, alternative = "two.sided", exact = FALSE)
results_1_2 <- wilcox.test(data_1, mu = median_2005, alternative = "greater", exact = FALSE)
wilcoxon_test_1 = wilcox.test(sort(walsh(data_1)), conf.int = TRUE)

print(results_1_1)
print(results_1_2)
print(wilcoxon_test_1)

# problem 4.3
print("Problem 4.3")
#install.packages("nortest", repos = "https://cloud.r-project.org/")
library(nortest)

lillie_result <- lillie.test(data_1)

print(lillie_result)
# p is low so reject at 95% conf.

SW_result <- shapiro.test(data_1)
print(SW_result)

# problem 4.2

print("Problem 4.2")

data_2 <-  c(10,42,29,11.00001,63,145,11,8,23,17,5,20,15.00001,36,32,15)
data_2_sort <- sort(data_2)
#print(data_2_sort)

F <-function(x) { 1-exp(-x/20)}

result_2_1 <- ks.test(data_2, F, alternative = "two.sided", exact = TRUE)
print(result_2_1)


# problem 4.7

print("Problem 4.7")

n_1 <- 18
n_2 <- 225

c_1 <- c(6,12)
c_2 <- c(75,150)

H_0 <- 0.5
p_hat <- 1/3

SE_1 <- sqrt(H_0*(1-H_0)/n_1)
SE_2 <- sqrt(H_0*(1-H_0)/n_2)

z_1 <- abs((p_hat - H_0)/(SE_1))
z_2 <- abs((p_hat - H_0)/(SE_2))

print(z_1)
print(z_2)

score_1 <- 2*pnorm(-abs(z_1))
score_2 <- 2*pnorm(-abs(z_2))

print(score_1)
print(score_2)

CI_1 <- c(p_hat - 1.96 * SE_1, p_hat + 1.96 * SE_1)
CI_2 <- c(p_hat - 1.96 * SE_2, p_hat + 1.96 * SE_2)

print(CI_1)
print(CI_2)