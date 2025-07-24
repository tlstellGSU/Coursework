data <- data.frame(
    ID = c(64,65,66,67,68,69,70,71,72,73,75,77,78,83),
    count = c(1,2,1,1,4,3,4,5,3,3,1,6,1,1)
)
print(data)

#result <- wilcox.test((data$ID, mu = 70.5, alternative = "two.sided", exact = True))

test_69_F <- wilcox.test(data$ID, mu = 69, 
                      alternative = "greater", # Test if the median is different from 70.5
                      exact = FALSE) # Use an approximate method for larger datasets

test_72_F <- wilcox.test(data$ID, mu = 72, 
                      alternative = "less", # Test if the median is different from 70.5
                      exact = FALSE) # Use an approximate method for larger datasets

test_69 <- wilcox.test(data$ID, mu = 69, 
                      alternative = "greater", # Test if the median is different from 70.5
                      exact = TRUE) # Use an approximate method for larger datasets

test_72 <- wilcox.test(data$ID, mu = 72, 
                      alternative = "less", # Test if the median is different from 70.5
                      exact = TRUE) # Use an approximate method for larger datasets

result_735 <- wilcox.test(data$ID, mu = 73.5, 
                      alternative = "two.sided", # Test if the median is different from 70.5
                      exact = TRUE) # Use an approximate method for larger datasets


#print(test_69$p.value)

#print(test_72$p.value)

#print(test_69_F$p.value)

#print(test_72_F$p.value)

#print(1- test_69$p.value - test_72$p.value)

print(result_735$p.value)


data_2 <- data.frame(
    count = c(10,42,29,11,63,145,11,8,23,17,5,20,15,36,32,15)
)

