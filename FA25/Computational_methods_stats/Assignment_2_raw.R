base_function <- function(x){
    return(exp(2*x) - 2)
}

base_function_deriv <- function(x){
    return(2 * exp(2*x))
}

exact_solution <- log(2) / 2

# 2.a Root-finding/bisection method

root_finding_method <- function(f, a, b, tol = 1e-5, max_iter = 10){
    if (f(a) * f(b) >= 0){
        stop("f(a) and f(b) must have different signs")
    }
    
    iter <- 0
    while ((b - a) / 2 > tol && iter < max_iter){
        c <- (a + b) / 2
        if (f(c) == 0){
            return(c)
        } else if (f(a) * f(c) < 0){
            b <- c
        } else {
            a <- c
        }
        iter <- iter + 1

        print(paste("Iteration:", iter)) 
        print(paste("a:", a, "b:", b, "c:", c))
        print(paste("f(a):", f(a), "f(b):", f(b)))
    }

    return((a + b) / 2)}

# 2.b Newton-Raphson method

NR_method <- function(f, f_deriv, x0, tol = 1e-5, max_iter = 10){
    iter <- 0
    x <- x0

    while (iter < max_iter && abs(f(x)) > tol){
        x_new <- x - f(x) / f_deriv(x)
                
        x <- x_new
        iter <- iter + 1

        print(paste("Iteration:", iter, "x:", x, "f(x):", f(x)))
    }
    
    return(x)
}

# 2.c Secant method

Secant_method <- function(f, x0, xn1, tol =1e-5, max_iter = 10){
    iter <- 0
    x_prev <- x0
    x_curr <- xn1

    while (iter < max_iter && abs(f(x_curr)) > tol){
        x_new <- x_curr - f(x_curr) * (x_curr - x_prev) / (f(x_curr) - f(x_prev))
        
        x_prev <- x_curr
        x_curr <- x_new
        iter <- iter + 1

        print(paste("Iteration:", iter, "x:", x_curr, "f(x):", f(x_curr)))
    }
    
    return(x_curr)
}

# running the code:

# Root-finding method
print("Root-finding method:")
answer_a <- root_finding_method(base_function, 0, 2)
cat("\n")
diff_a <- abs(answer_a - exact_solution)
print(paste("Absolute Difference:", diff_a))
cat("\n")
# Newton-Raphson method
print("Newton-Raphson method:")
answer_b <- NR_method(base_function, base_function_deriv, 0)
cat("\n")
diff_b <- abs(answer_b - exact_solution)
print(paste("Absolute Difference:", diff_b))
cat("\n")
# Secant method
print("Secant method:")
answer_c <- Secant_method(base_function, 1, 1.5)
cat("\n")
diff_c <- abs(answer_c - exact_solution)
print(paste("Absolute Difference:", diff_c))
