# problem 10.9

data_1 <- data.frame(
  Dog = c("MVO", "LVP"),
  A = c(78, 33),
  B = c(92, 33),
  C = c(116, 45),
  D = c(90, 30),
  E = c(106, 38),
  F = c(78, 24),
  G = c(89, 44)
)

mv_values <- as.numeric(data_1[1, -1])
lv_values <- as.numeric(data_1[2, -1])  

result_1_1 <- cor.test(mv_values, lv_values, method = "kendall", exact = FALSE)
result_1_2 <- cor.test(mv_values, lv_values, method = "spearman", exact = FALSE)

cat("Problem 10.9 \n")
cat("Kendall Correlation Coefficient: \n")
#print(result_1_1$estimate)  
print(result_1_1)
cat("Exact two-tail P-value (Kendall): \n")
#print(result_1_1$p.value)  
print(result_1_2)

cat("Spearman Correlation Coefficient: \n")
print(result_1_2$estimate)  

cat("Exact two-tail P-value (Spearman): \n")
print(result_1_2$p.value)  


# problem 10.10

data_2_1 <- c(41, 0, 42, 15, 47, 0, 0, 0, 56, 67, 707, 368, 231, 104, 132, 200, 172, 146, 0)
data_2_2 <- c(4716, 4605, 4951, 2745, 6592, 8934, 9165, 5917, 2618, 1105, 150, 2005, 3222, 7150, 8658, 6304, 1800, 5270, 1537)

result_2 <- cor.test(data_2_1, data_2_2, method = "pearson", exact = FALSE)
result_2_2 <- cor.test(data_2_1, data_2_2, method = "spearman", exact = FALSE)
result_2_3 <- cor.test(data_2_1, data_2_2, method = "kendall", exact = FALSE)
cat("\n\n\nProblem 10.10 \n")
cat("Pearson Correlation Coefficient: \n")
print(result_2$estimate)  
cat("Exact two-tail P-value (Pearson): \n")
print(result_2$p.value)  
print(result_2)
print(result_2_2)
print(result_2_3)