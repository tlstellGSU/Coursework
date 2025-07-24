# problem 7.13

data_1_1 <- c(53, 50, 52, 50, 49, 47, 54, 51, 52, 57)
data_1_2 <- c(49, 49, 47, 54, 43, 51, 49, 51, 50, 46, 49)
data_1_3 <- c(58, 51, 45, 53, 49, 51, 50, 51)

all_values_1 <- c(data_1_1, data_1_2, data_1_3)
group_1 <- factor(c(rep("species_A", length(data_1_1)),
                    rep("species_B", length(data_1_2)),
                    rep("species_C", length(data_1_3))
                    ))

N_1 <- length(data_1_1) + length(data_1_2) + length(data_1_3)

result_1 <- kruskal.test(all_values_1 ~ group_1)

cat("problem 7.13")
print(result_1)

# problem 7.15

data_2_1 <- c(156, 181, 220, 238, 295, 334, 342, 359, 365, 374, 391, 395, 481)
data_2_2 <- c(65, 105, 121, 150, 158, 170, 214, 235, 238, 255, 265, 390)
data_2_3 <- c(33, 70, 87, 109, 114, 132, 150, 179, 184, 241, 323)
data_2_4 <- c(79, 84, 94, 259)

all_values_2 <- c(data_2_1,data_2_2,data_2_3,data_2_4)
group_2 <- factor(c(rep("size_0", length(data_2_1)),
                    rep("size_1", length(data_2_2)),
                    rep("size_2", length(data_2_3)),
                    rep("size_3", length(data_2_4))),
                  levels = c("size_0", "size_1", "size_2", "size_3"),
                  ordered = TRUE)  



#install.packages('PMCMRplus', repos = "https://cloud.r-project.org/")
library(PMCMRplus)

result_2 <- jonckheereTest(all_values_2, group_2, "less")
cat("\n \nproblem 7.13")
print(result_2)