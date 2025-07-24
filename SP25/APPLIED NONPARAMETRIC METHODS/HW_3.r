# functions

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

# problem 6.4
cat("problem 6.4 \n")
data_1_1 <- c(15.7, 14.8, 14.2, 16.1, 15.3, 13.9, 17.2, 14.9)
data_1_2 <- c(13.7, 14.1, 14.7, 15.4, 15.6, 14.4, 12.9, 15.1, 14.0)


result_1_1 <-wilcox.test(data_1_1, data_1_2, exact = FALSE)
print(result_1_1)
print(wilcox.test(data_1_1, data_1_2,conf.int = TRUE))

result_1_2 <- wilcox.test(data_1_1, data_1_2, exact = TRUE)
print(result_1_2)

t_test_1 <- t.test(x=data_1_1, y=data_1_2, var.equal = FALSE)
print(t_test_1)
print(t.test(x=data_1_1, y=data_1_2, conf.int = TRUE, var.equal = FALSE))


# problem 6.4.2
cat(" \n \n \n")
cat("problem 6.4.2 \n")


result_2 <- ks.test(data_1_1,data_1_2)
print(result_2)



# problem 6.7
cat(" \n \n \n")
cat("problem 6.7 \n")

data_3_1 <- c(42, 44, 38, 52, 48, 46, 34, 44, 38)
data_3_2 <- c(34, 43, 35, 33, 34, 26, 30, 31, 27, 28, 27, 30, 37, 38, 32, 32, 36, 32, 32, 38, 42, 36, 44, 33, 38)

result_3_1 <- wilcox.test(data_3_1, data_3_2, conf.int=TRUE, exact = FALSE)
#print(result_3_1)

cat("Hodge Lehman Estimator \n")
print(result_3_1$estimate)
cat("Confidence Interval\n")
print(result_3_1$conf.int)

result_3_2 <- t.test(data_3_1, data_3_2, var.equal = FALSE)
#print(result_3_2)
mean_diff <- result_3_2$estimate[1] - result_3_2$estimate[2]
cat("Mean difference \n")
print(mean_diff)

cat("Confidence interval from t test \n")
print(result_3_2$conf.int)


# problem 6.8
cat(" \n \n \n")
cat("problem 6.8 \n")

data_4_1 <- c(0,1,2,3,4,5,6)
data_4_2 <- c(26,28,34,48,21,22,34)
data_4_3 <- c(28,27,42,44,17, 6,16)

result_4_1 <- wilcox.test(data_4_2, data_4_3, conf.int=TRUE, exact = FALSE)
print(result_4_1)
cat("Hodge Lehman Estimator \n")
print(result_4_1$estimate)
cat("Confidence Interval\n")
print(result_4_1$conf.int)

result_4_2 <- t.test(data_4_2, data_4_3, var.equal = FALSE)
print(result_4_2)
mean_diff_2 <- result_4_2$estimate[1] - result_4_2$estimate[2]
cat("Mean difference \n")
print(mean_diff_2)

cat("Confidence interval from t test \n")
print(result_4_2$conf.int)




