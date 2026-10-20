options(repos = c(CRAN = "https://cran.rstudio.com/"))

#install.packages("geometry")
#install.packages("dbscan")
#install.packages("alphahull")
#install.packages("igraph")

auto_epsilon <- function(points, min_points = 5, quantile_value = 0.90) {
    points <- as.matrix(points)
    k_distribution <- dbscan::kNNdist(points, k = min_points)
    k_distribution_vec <- as.numeric(k_distribution)
    epsilon_estimated <- as.numeric(quantile(k_distribution_vec, probs = quantile_value))

    return(epsilon_estimated)
}

dbscan_filter <- function(points, min_points = 5, epsilon = NULL,
                          quantile_epsilon = 0.95, B = 2000,
                          sample_fraction = 0.95, CL = 0.95) {
    points <- as.matrix(points)
    n <- nrow(points)
    d <- ncol(points)

    if (is.null(epsilon)) {
        epsilon <- auto_epsilon(points, min_points, quantile_epsilon)
    }

    # m-out-of-n sample size
    m <- max(min(n - 1, floor(n * sample_fraction)), min_points + 1)

    # m-out-of-n density scaling:
    #
    #   epsilon_m = epsilon * (n/m)^(1/d)
    #
    eps_m <- epsilon * (n / m)^(1 / d)

    inclusion_counts <- integer(n)
    sampled_counts <- integer(n)

    for (b in 1:B) {
        idx <- sample.int(n, size = m, replace = FALSE)
        sampled_counts[idx] <- sampled_counts[idx] + 1L

        db <- dbscan::dbscan(points[idx, , drop = FALSE],
            eps = eps_m,
            minPts = min_points
        )

        clustered_local <- which(db$cluster != 0)

        inclusion_counts[idx[clustered_local]] <-
            inclusion_counts[idx[clustered_local]] + 1L
    }

    # Estimate per-point inclusion probability
    inclusion_prob <- inclusion_counts / pmax(sampled_counts, 1L)

    # CL now means PB( point is “clustered” | sampled ) >= CL
    keep <- which(inclusion_prob >= 1 - CL & sampled_counts > 0)

    points[keep, , drop = FALSE]
}


# fast alpha shape (2D) implementation
alpha_shape_2D <- function(points, alpha, CL=0.95, simplify = TRUE) {
    if (!requireNamespace("alphahull", quietly = TRUE)) {
        stop("Install alphahull: install.packages('alphahull')")
    }

    pts <- as.matrix(points)

    # DBSCAN filtering
    if (!is.null(CL) && CL < 1.0){
    pts <- dbscan_filter(pts, min_points = 5, CL = CL)
    }

    ash <- alphahull::ashape(pts, alpha = alpha)

    edges <- ash$edges

    # Case 1: no edges at all → empty list
    if (length(edges) == 0) {
        return(list(polygons = list(), edges = NULL, ash = ash))
    }

    
    # Case 2: edges is a matrix (happens rarely)
    if (is.matrix(edges)) {
        if (ncol(edges) >= 2) {
            edges <- data.frame(ind1 = edges[,1], ind2 = edges[,2])
        } else {
            return(list(polygons = list(), edges = NULL, ash = ash))
        }
    }

    # Case 3: edges is not a data frame (atomic vector) → cannot build polygons
    if (!is.data.frame(edges)) {
        return(list(polygons = list(), edges = NULL, ash = ash))
    }

    # Ensure required columns exist
    if (!all(c("ind1", "ind2") %in% colnames(edges))) {
        return(list(polygons = list(), edges = NULL, ash = ash))
    }

    # ---- adjacency ----
    adj <- list()
    for (i in seq_len(nrow(edges))) {
        a <- edges$ind1[i]
        b <- edges$ind2[i]

        adj[[as.character(a)]] <- c(adj[[as.character(a)]], b)
        adj[[as.character(b)]] <- c(adj[[as.character(b)]], a)
    }

    visited <- new.env()

    mark_edge <- function(a, b) {
        k <- if (a < b) paste0(a, "-", b) else paste0(b, "-", a)
        visited[[k]] <- TRUE
    }

    is_marked <- function(a, b) {
        k <- if (a < b) paste0(a, "-", b) else paste0(b, "-", a)
        !is.null(visited[[k]])
    }

    polygons <- list()

    # ---- walk cycles ----
    for (i in seq_len(nrow(edges))) {
        a <- edges$ind1[i]
        b <- edges$ind2[i]

        if (is_marked(a, b)) next

        poly <- c(a, b)
        mark_edge(a, b)

        prev <- a
        cur  <- b

        repeat {
            neigh <- adj[[as.character(cur)]]
            nxt <- neigh[!neigh %in% prev]

            nxt <- nxt[!vapply(nxt, function(v) is_marked(cur, v), logical(1))]

            if (length(nxt) == 0) break

            nxt <- nxt[1]

            poly <- c(poly, nxt)
            mark_edge(cur, nxt)

            prev <- cur
            cur  <- nxt

            if (cur == a) break
        }

        if (length(poly) >= 3) {
            P <- pts[c(poly, poly[1]), , drop = FALSE]
            polygons[[length(polygons) + 1]] <- P
        }
    }

    return(list(polygons = polygons, edges = edges, ash = ash))
}

visualize_alpha_shape_2D <- function(alpha_shape_result, points, title = "Alpha Shape (2D)",
                                     r_outer = NULL, r_inner = NULL, add = FALSE, shape = "ring",
                                     num_rings = NULL, spacing = NULL, ring_width = NULL,
                                     save_path = NULL) {
    if (!is.null(save_path)) {
        png(filename = save_path, width = 800, height = 800)
        on.exit(dev.off())
    }

    plots <- alpha_shape_result
    loops <- plots$polygons

    # If not adding to an existing plot, initialize the plot window with no background
    if (!add) {
        plot(points,
            col = "black", pch = 19, main = title, asp = 1,
            xlab = "x", ylab = "y",
            bg = NA
        ) # transparent background
    } else {
        points(points, col = "black", pch = 19)
    }

    if (length(loops) == 0) {
        return(invisible(NULL))
    }

    # ---- Alpha-shape outlines (blue) ----
    for (i in seq_along(loops)) {
        loop <- loops[[i]]
        polygon(loop, border = "blue", lwd = 2, col = NA)
    }

    # ---- True shape outlines (red) ----
    if (shape == "ring" && !is.null(r_outer) && !is.null(r_inner)) {
        # outline between the two circles to represent the ring
        theta <- seq(0, 2 * pi, length.out = 400)
        lines(r_outer * cos(theta), r_outer * sin(theta),
            col = "red", lwd = 2
        )

        lines(r_inner * cos(theta), r_inner * sin(theta),
            col = "red", lwd = 2
        )
    } else if (shape == "concentric" && !is.null(num_rings) && !is.null(ring_width) && !is.null(spacing)) {
        # outline for multiple concentric rings
        r_inner <- spacing
        for (j in 1:num_rings) {
            r_outer <- r_inner + ring_width
            theta <- seq(0, 2 * pi, length.out = 400)
            lines(r_outer * cos(theta), r_outer * sin(theta), col = "red", lwd = 2)
            lines(r_inner * cos(theta), r_inner * sin(theta), col = "red", lwd = 2)
            r_inner <- r_outer + spacing
        }
    } else if (shape == "squares" && !is.null(r_outer) && !is.null(r_inner)) {
        # outline between the two squares to represent the square ring
        square_outer <- rbind(
            c(-r_outer, -r_outer),
            c(r_outer, -r_outer),
            c(r_outer, r_outer),
            c(-r_outer, r_outer),
            c(-r_outer, -r_outer)
        )
        lines(square_outer, col = "red", lwd = 2)

        square_inner <- rbind(
            c(-r_inner, -r_inner),
            c(r_inner, -r_inner),
            c(r_inner, r_inner),
            c(-r_inner, r_inner),
            c(-r_inner, -r_inner)
        )
        lines(square_inner, col = "red", lwd = 2)
    } else {
        invisible(NULL)
    }


    invisible(NULL)
}

concentric_rings_2D <- function(num_rings = 3, spacing = 1, ring_width = 1.0, total_points = 1000, noise = 0.01) {
    outer_edge <- num_rings * spacing + ring_width + spacing
    n_noise <- floor(noise * total_points)
    n_ring_points <- total_points - n_noise

    # equal ratio of points per ring

    points_per_ring_ratios <- seq(1, num_rings) / sum(seq(1, num_rings))

    points <- matrix(0, nrow = total_points, ncol = 2)
    current_idx <- 1

    r_inner <- spacing

    # Generate ring points first
    for (i in 1:num_rings) {
        r_outer <- r_inner + ring_width

        points_per_ring <- points_per_ring_ratios[i] * n_ring_points

        r <- runif(points_per_ring, r_inner, r_outer)
        theta <- runif(points_per_ring, 0, 2 * pi)
        x <- r * cos(theta)
        y <- r * sin(theta)

        end_idx <- current_idx + points_per_ring - 1
        points[current_idx:end_idx, ] <- cbind(x, y)
        current_idx <- end_idx + 1

        r_inner <- r_outer + spacing
    }

    # Generate noise points in remaining slots
    if (n_noise > 0) {
        x_noise <- runif(n_noise, -outer_edge, outer_edge)
        y_noise <- runif(n_noise, -outer_edge, outer_edge)
        points[current_idx:(current_idx + n_noise - 1), ] <- cbind(x_noise, y_noise)
    }

    return(points)
}


alternating_squares_pointcloud_2D <- function(total_points = 1000, num_squares = 5, square_width = 0.5, spacing = 1, noise = 0.05) {
    points <- matrix(0, nrow = 0, ncol = 2)

    total_points <- ceiling(total_points / 2)

    if (noise > 0) {
        noisy_points <- floor(noise * total_points)
        total_points <- total_points - noisy_points
    }

    points_per_square_ratio <- seq(1, num_squares) / sum(seq(1, num_squares))

    center_x <- 0
    center_y <- 0

    for (i in 1:num_squares) {
        inner_k <- spacing + 2 * (i - 1) * (square_width + spacing)
        outer_k <- spacing + 2 * (i - 1) * (square_width + spacing) + 2 * square_width

        num_square_points <- floor(points_per_square_ratio[i] * total_points)

        square_points <- matrix(0, nrow = 0, ncol = 2)

        # generate two numbers that are uniform in the hollow square and then generate a othrogonal point
        # in the actual bounds

        for (j in 1:num_square_points) {
            rand_x_1 <- runif(1, -outer_k, outer_k)

            y_bounds_1 <- runif(1, inner_k, outer_k)
            y_sign_1 <- sample(c(-1, 1), 1)
            rand_y_1 <- y_sign_1 * y_bounds_1

            rand_y_2 <- runif(1, -outer_k, outer_k)

            x_bounds_2 <- runif(1, inner_k, outer_k)
            x_sign_2 <- sample(c(-1, 1), 1)
            rand_x_2 <- x_sign_2 * x_bounds_2

            square_points <- rbind(square_points, cbind(rand_x_1, rand_y_1))
            square_points <- rbind(square_points, cbind(rand_x_2, rand_y_2))
        }

        points <- rbind(points, square_points)
    }

    # Add noise points
    if (noise > 0) {
        x_noise <- runif(noisy_points, min(points[, 1]) - spacing, max(points[, 1]) + spacing)
        y_noise <- runif(noisy_points, min(points[, 2]) - spacing, max(points[, 2]) + spacing)
        noise_points <- cbind(x_noise, y_noise)
        points <- rbind(points, noise_points)
    }

    return(points)
}

alternating_squares_polygon <- function(num_squares = 5,
                                        square_width = 0.5,
                                        spacing = 1,
                                        center_x = 0,
                                        center_y = 0) {
    polygon_list <- list()

    for (i in 1:num_squares) {
        inner_k <- spacing / 2 + (i - 1) * (square_width + spacing)
        outer_k <- inner_k + square_width

        # Outer boundary (CCW)
        outer <- rbind(
            c(center_x - outer_k, center_y + outer_k),
            c(center_x + outer_k, center_y + outer_k),
            c(center_x + outer_k, center_y - outer_k),
            c(center_x - outer_k, center_y - outer_k),
            c(center_x - outer_k, center_y + outer_k) # close
        )

        # Inner boundary (CW for a "hole")
        inner <- rbind(
            c(center_x - inner_k, center_y + inner_k),
            c(center_x - inner_k, center_y - inner_k),
            c(center_x + inner_k, center_y - inner_k),
            c(center_x + inner_k, center_y + inner_k),
            c(center_x - inner_k, center_y + inner_k) # close
        )
        inner <- inner[nrow(inner):1, ] # reverse to CW

        # Combine into a single polygon structure
        poly <- rbind(outer, inner)

        polygon_list[[i]] <- poly
    }

    return(polygon_list)
}



area_concentric_rings_2D <- function(num_rings = 3, spacing = 1, ring_width = 0.5) {
    total_area <- 0
    r_inner <- spacing
    for (i in 1:num_rings) {
        r_outer <- r_inner + ring_width
        area_ring <- pi * (r_outer^2 - r_inner^2)
        total_area <- total_area + area_ring
        r_inner <- r_outer + spacing
    }
    return(total_area)
}


calculate_KL_divergence <- function(true_shape, estimated_shape, dim = 2, grid_res = 10000, num_rings = 3, spacing = 3.0, width = 1.5, shape = "concentric") {
    # Function to calculate KL divergence between true shape and estimated alpha shape
    # true_shape: the true shape representation (e.g., polygon for 2D)
    # estimated_shape: the estimated alpha shape representation
    # dim: dimension (2 or 3)

    if (dim == 2) {
        if (length(true_shape$polygons) == 0 || length(estimated_shape$polygons) == 0) {
            return(NA)
        }



        # bounding box
        all_points <- rbind(do.call(rbind, true_shape$polygons), do.call(rbind, estimated_shape$polygons))
        x_range <- range(all_points[, 1])
        y_range <- range(all_points[, 2])

        x_seq <- seq(x_range[1], x_range[2], length.out = grid_res)
        y_seq <- seq(y_range[1], y_range[2], length.out = grid_res)

        # create a Monte Carlo grid

        # grid <- expand.grid(x = x_seq, y = y_seq)
        grid <- data.frame(x = runif(grid_res * grid_res, min = x_range[1], max = x_range[2]), y = runif(grid_res * grid_res, min = y_range[1], max = y_range[2]))



        # helper to check if points are in any polygon
        points_in_polygons <- function(points, polygons) {
            inside <- rep(FALSE, nrow(points))
            for (poly in polygons) {
                inside <- inside | sp::point.in.polygon(points$x, points$y, poly[, 1], poly[, 2]) > 0
            }
            return(inside)
        }

        if (shape == "squares") {
            inside_est <- points_in_polygons(grid, estimated_shape$polygons)
            inside_true <- c()
            # for grid points, check if in any of the square rings using the math
            for (i in 1:nrow(grid)) {
                point <- grid[i, ]
                is_inside <- FALSE

                for (j in 1:num_rings) {
                    r_inner <- spacing + 2 * (j - 1) * (width + spacing)
                    r_outer <- r_inner + 2 * width
                    max_value <- max(abs(point$x), abs(point$y))
                    if (max_value >= r_inner && max_value <= r_outer) {
                        is_inside <- TRUE
                        break
                    }
                }
                inside_true <- c(inside_true, is_inside)
            }

            P <- inside_true / sum(inside_true)
            Q <- inside_est / sum(inside_est)

            # add small epsilon to avoid log(0)

            eps <- 1e-16
            P <- P + eps
            Q <- Q + eps

            KL <- sum(P * log(P / Q))
            return(KL)
        }

        # true vs estimated occupancy
        inside_true <- points_in_polygons(grid, true_shape$polygons)
        inside_est <- points_in_polygons(grid, estimated_shape$polygons)

        P <- inside_true / sum(inside_true)
        Q <- inside_est / sum(inside_est)

        # add small epsilon to avoid log(0)
        eps <- 1e-16
        P <- P + eps
        Q <- Q + eps

        KL <- sum(P * log(P / Q))
        return(KL)
    } else {
        # Placeholder for 3D KL divergence calculation
        return(NA)
    }
}

IOU_score <- function(true_shape, estimated_shape, dim = 2, grid_res = 10000, shape = "concentric", num_rings = 3, spacing = 3.0, width = 1.5) {
    # Function to calculate Intersection over Union (IoU) score between true shape and estimated alpha shape
    # true_shape: the true shape representation (e.g., polygon for 2D)
    # estimated_shape: the estimated alpha shape representation
    # dim: dimension (2 or 3)

    if (dim == 2) {
        if (length(true_shape$polygons) == 0 || length(estimated_shape$polygons) == 0) {
            return(0)
        }

        # bounding box
        all_points <- rbind(do.call(rbind, true_shape$polygons), do.call(rbind, estimated_shape$polygons))
        x_range <- range(all_points[, 1])
        y_range <- range(all_points[, 2])

        x_seq <- seq(x_range[1], x_range[2], length.out = grid_res)
        y_seq <- seq(y_range[1], y_range[2], length.out = grid_res)

        # create a Monte Carlo grid
        grid <- expand.grid(x = x_seq, y = y_seq)

        # helper to check if points are in any polygon
        points_in_polygons <- function(points, polygons) {
            inside <- rep(FALSE, nrow(points))
            for (poly in polygons) {
                inside <- inside | sp::point.in.polygon(points$x, points$y, poly[, 1], poly[, 2]) > 0
            }
            return(inside)
        }

        if (shape == "squares") {
            inside_est <- points_in_polygons(grid, estimated_shape$polygons)
            inside_true <- c()
            # for grid points, check if in any of the square rings using the math
            for (i in 1:nrow(grid)) {
                point <- grid[i, ]
                is_inside <- FALSE

                for (j in 1:num_rings) {
                    r_inner <- spacing + 2 * (j - 1) * (width + spacing)
                    r_outer <- r_inner + 2 * width
                    max_value <- max(abs(point$x), abs(point$y))
                    if (max_value >= r_inner && max_value <= r_outer) {
                        is_inside <- TRUE
                        break
                    }
                }
                inside_true <- c(inside_true, is_inside)
            }

            intersection <- sum(inside_true & inside_est)
            union <- sum(inside_true | inside_est)

            IoU <- intersection / union
            return(IoU)
        }

        # true vs estimated occupancy
        inside_true <- points_in_polygons(grid, true_shape$polygons)
        inside_est <- points_in_polygons(grid, estimated_shape$polygons)

        intersection <- sum(inside_true & inside_est)
        union <- sum(inside_true | inside_est)

        IoU <- intersection / union
        return(IoU)
    } else {
        # Placeholder for 3D IoU calculation
        return(0)
    }
}

main_test <- function(dim = 2, alpha = 0.2, CL = 1.0, num_points = 1000, noise_in_data = 0.1, title = "Alpha Shape Visualization", r_inner = 1, r_outer = 3, shape = "ring", spacing = 1, ring_width = 0.5, num_rings = 3, sampling_fraction = 0.8) {
    # generate test data

    if (dim == 2) {
        print("Generating 2D test data...")
        if (shape == "ring") {
            points <- ring_test_generation_2D(num_points = num_points, noise = noise_in_data, r_outer = r_outer, r_inner = r_inner)
            true_area <- pi * (r_outer^2 - r_inner^2)
            true_shape <- list(polygons = list(
                matrix(c(
                    r_outer * cos(seq(0, 2 * pi, length.out = 100)),
                    r_outer * sin(seq(0, 2 * pi, length.out = 100))
                ), ncol = 2),
                matrix(c(
                    r_inner * cos(seq(0, 2 * pi, length.out = 100)),
                    r_inner * sin(seq(0, 2 * pi, length.out = 100))
                ), ncol = 2)
            ))
        } else if (shape == "concentric") {
            points <- concentric_rings_2D(
                num_rings = num_rings, spacing = spacing,
                ring_width = ring_width, total_points = num_points, noise = noise_in_data
            )

            # Build true polygons with proper holes
            true_shape <- list(polygons = list())
            r_inner <- spacing
            for (i in 1:num_rings) {
                r_outer <- r_inner + ring_width
                # Outer polygon (counterclockwise)
                true_shape$polygons[[length(true_shape$polygons) + 1]] <- cbind(
                    r_outer * cos(seq(0, 2 * pi, length.out = 200)),
                    r_outer * sin(seq(0, 2 * pi, length.out = 200))
                )
                # Inner polygon (clockwise)
                true_shape$polygons[[length(true_shape$polygons) + 1]] <- cbind(
                    r_inner * cos(seq(0, -2 * pi, length.out = 200)),
                    r_inner * sin(seq(0, -2 * pi, length.out = 200))
                )
                r_inner <- r_outer + spacing
            }

            # True area sum
            true_area <- area_concentric_rings_2D(
                num_rings = num_rings,
                spacing = spacing, ring_width = ring_width
            )
        } else if (shape == "pitchfork") {
            points <- pitchfork_bifurcation_datacloud(num_points = num_points, noise = noise_in_data, direction = "up")
            true_shape <- list(polygons = list(
                polygon_pitchfork(x_min = -2, x_max = 2, r_min = -2, r_max = 2)
            ))
            true_area <- 8
        } else if (shape == "squares") {
            points <- alternating_squares_pointcloud_2D(total_points = num_points, num_squares = num_rings, square_width = ring_width, spacing = spacing, noise = noise_in_data)
            #true_area <- square_area_2D(num_squares = num_rings, square_width = ring_width, spacing = spacing)
            true_shape <- list(polygons = alternating_squares_polygon(num_squares = num_rings, square_width = ring_width, spacing = spacing))
        } else {
            stop("Unknown shape type.")
        }

        print("Computing alpha shape...")

        alpha_shape_result <- alpha_shape_2D(points, alpha = alpha, CL = CL)

        # build a filename safely using base R; "%+%" is not a base R operator and
        # will error if the package that defines it isn't loaded
        save_path <- paste0("alpha_shape_2D_alpha_", alpha, "_CL_", CL, "_noise_", noise_in_data, "_shape_", shape, ".png")

        visualize_alpha_shape_2D(alpha_shape_result, points, title, r_outer = r_outer, r_inner = r_inner, shape = shape, spacing = spacing, ring_width = ring_width, num_rings = num_rings, save_path = save_path)

        # estimated_area <- calculate_area_volume(alpha_shape_result, dim = 2)
        # error <- error_in_2D(true_area, estimated_area)
        # print(paste("Error in 2D area estimation:", error))
        # print(paste("Estimated Area:", estimated_area, "True Area:", true_area))

        KL_divergence <- calculate_KL_divergence(
            true_shape = true_shape,
            estimated_shape = alpha_shape_result,
            dim = 2,
            shape = shape,
            num_rings = num_rings,
            spacing = spacing,
            width = ring_width
        )
        IOU_score_value <- IOU_score(
            true_shape = true_shape,
            estimated_shape = alpha_shape_result,
            dim = 2,
            shape = shape,
            num_rings = num_rings,
            spacing = spacing,
            width = ring_width
        )
        print(paste("KL Divergence:", KL_divergence, "IOU Score:", IOU_score_value))
    } else {
        points <- torus_test_generation_3D(num_points = num_points, noise = noise_in_data)
        print("3D alpha shape not yet implemented.")
    }

    # Return both values as a list so callers can unpack by name
    return(list(KL_divergence = KL_divergence, IOU_score = IOU_score_value, alpha_shape = alpha_shape_result))
}

alpha_variations <- function(alpha_value_list = c(0.1, 0.2),
                             dim = 2,
                             CL_list = c(1.0, 0.95, 0.90),
                             num_points = 10000,
                             noise_in_data = c(0.1, 0.01),
                             r_outer = 5,
                             r_inner = 3,
                             shape = "concentric",
                             spacing = 3.0,
                             ring_width = 1.5,
                             num_rings = 3,
                             specific_parameters = NULL,
                             save_path = NULL) {
    total_trials <- length(alpha_value_list) * length(CL_list) * length(noise_in_data)
    trial_count <- 0
    KL_divergence_results <- data.frame()
    IOU_score_results <- data.frame()

    score_results <- data.frame(
        alpha = numeric(),
        CL = numeric(),
        noise = numeric(),
        KL_divergence = numeric(),
        IOU_score = numeric(),
        shape = character(),
        time_taken = numeric()
    )

    if (!is.null(specific_parameters)) {
        print("Running specific parameter combinations.")
        total_trials <- length(specific_parameters)
        for (params in specific_parameters) {
            # params may come in as a character vector if constructed with c(..., "shape")
            # coerce the numeric entries explicitly to avoid non-numeric errors
            alpha <- as.numeric(params[1])
            CL <- as.numeric(params[2])
            noise <- as.numeric(params[3])
            shape_specific <- as.character(params[4])
            time_start <- Sys.time()
            print(paste("Trial", trial_count + 1, "of", total_trials))
            print(paste("Running for Alpha:", alpha, "CL:", CL, "Noise:", noise, "Shape:", shape_specific))
            title <- paste("Alpha Shape Visualization - Alpha:", alpha, "CL:", CL, "Noise:", noise, "Shape:", shape_specific)
            res <- main_test(dim = dim, alpha = alpha, CL = CL, num_points = num_points, noise_in_data = noise, title = title, r_outer = r_outer, r_inner = r_inner, shape = shape_specific, spacing = spacing, ring_width = ring_width, num_rings = num_rings)
            KL_divergence <- res$KL_divergence
            IOU_score_value <- res$IOU_score
            KL_divergence_results <- rbind(KL_divergence_results, data.frame(alpha = alpha, CL = CL, noise = noise, KL_divergence = KL_divergence))
            IOU_score_results <- rbind(IOU_score_results, data.frame(alpha = alpha, CL = CL, noise = noise, IOU_score = IOU_score_value))
            trial_count <- trial_count + 1
            time_end <- Sys.time()
            time_taken <- as.numeric(time_end - time_start, units = "secs")
            score_results <- rbind(score_results, data.frame(alpha = alpha, CL = CL, noise = noise, KL_divergence = KL_divergence, IOU_score = IOU_score_value, shape = shape_specific, time_taken = time_taken))
        }
    }
    if (is.null(specific_parameters)) {
        for (alpha in alpha_value_list) {
            for (CL in CL_list) {
                for (noise in noise_in_data) {
                    print(paste("Trial", trial_count + 1, "of", total_trials))
                    print(paste("Running for Alpha:", alpha, "CL:", CL, "Noise:", noise))
                    title <- paste("Alpha Shape Visualization - Alpha:", alpha, "CL:", CL, "Noise:", noise)
                    res <- main_test(dim = dim, alpha = alpha, CL = CL, num_points = num_points, noise_in_data = noise, title = title, r_outer = r_outer, r_inner = r_inner, shape = shape, spacing = spacing, ring_width = ring_width, num_rings = num_rings)
                    KL_divergence <- res$KL_divergence
                    IOU_score_value <- res$IOU_score
                    KL_divergence_results <- rbind(KL_divergence_results, data.frame(alpha = alpha, CL = CL, noise = noise, KL_divergence = KL_divergence))
                    IOU_score_results <- rbind(IOU_score_results, data.frame(alpha = alpha, CL = CL, noise = noise, IOU_score = IOU_score_value))
                    score_results <- rbind(score_results, data.frame(alpha = alpha, CL = CL, noise = noise, KL_divergence = KL_divergence, IOU_score = IOU_score_value))
                    trial_count <- trial_count + 1
                }
            }
        }
    }

    if (!is.null(save_path)) {
        save_path <- save_path
    } else {
        if (shape == "concentric") {
            temp <- paste0(str(spacing), "_", str(ring_width), "_", str(num_rings))
            save_path <- paste0("alpha_shape_KL_IOU_concentric_rings_", temp, "_results.csv")
        } else if (shape == "ring") {
            temp <- paste0(str(r_inner), "_", str(r_outer))
            save_path <- paste0("alpha_shape_KL_IOU_ring_", temp, "_results.csv")
        } else if (shape == "pitchfork") {
            save_path <- paste0("alpha_shape_KL_IOU_pitchfork_results.csv")
        } else if (shape == "squares") {
            temp <- paste0(str(ring_width), "_", str(spacing), "_", str(num_rings))
            save_path <- paste0("alpha_shape_KL_IOU_squares_", temp, "_results.csv")
        } else {
            save_path <- paste0("alpha_shape_KL_IOU_", shape, "_results.csv")
        }
    }

    # save results to csv
    write.csv(score_results, file = save_path, row.names = FALSE)
    print(paste("Results saved to", save_path))
    print("Alpha variations completed.")
}

datapoints_variations <- function(
    num_points_list = c(1e1, 1e2, 1e3, 1e4, 1e5, 1e6)) {
    time_taken_list <- c()
    for (num_points in num_points_list) {
        time_start <- Sys.time()
        print(paste("Running for Number of Points:", num_points))
        title <- paste("Alpha Shape Visualization - Number of Points:", num_points)
        main_test(dim = 2, alpha = 1.0, CL = 0.95, shape = "ring", num_points = num_points, noise_in_data = 0.01, title = title)
        time_end <- Sys.time()
        time_taken <- time_end - time_start
        time_taken_list <- c(time_taken_list, as.numeric(time_taken, units = "secs"))
        print(paste("Time taken (seconds):", as.numeric(time_taken, units = "secs")))
    }
    result <- data.frame(
        num_points = num_points_list,
        time_taken_seconds = time_taken_list
    )
    # save to csv
    write.csv(result, file = "alpha_shape_datapoints_timing.csv", row.names = FALSE)
    print("Data points variations completed.")
}

plot_datapoints_timing <- function(csv_file = "alpha_shape_datapoints_timing.csv", scale = 1) {
    data <- read.csv(csv_file)

    if (scale == 1) {
        save_path <- "alpha_shape_timing_linear_linear.png"
        png(save_path)
        plot(data$num_points, data$time_taken_seconds,
            type = "b",
            xlab = "Number of Points", ylab = "Time Taken (seconds)",
            main = "Alpha Shape Computation Time vs Number of Points"
        )
        dev.off()
    } else if (scale == 2) {
        save_path <- "alpha_shape_timing_linear_log.png"
        png(save_path)
        plot(data$num_points, data$time_taken_seconds,
            log = "y", type = "b",
            xlab = "Number of Points", ylab = "Time Taken (seconds, log scale)",
            main = "Alpha Shape Computation Time vs Number of Points"
        )
        dev.off()
    } else if (scale == 3) {
        save_path <- "alpha_shape_timing_log_log.png"
        png(save_path)
        plot(data$num_points, data$time_taken_seconds,
            log = "xy", type = "b",
            xlab = "Number of Points (log scale)", ylab = "Time Taken (seconds, log scale)",
            main = "Alpha Shape Computation Time vs Number of Points"
        )
        dev.off()
    } else if (scale == 4) {
        save_path <- "alpha_shape_timing_n_vs_sqrt_t.png"
        png(save_path)
        plot(data$num_points, sqrt(data$time_taken_seconds),
            type = "b",
            xlab = "Number of Points", ylab = "Square Root of Time Taken (seconds)",
            # main = "Alpha Shape Computation Time vs Number of Points (n vs sqrt t scale)"
        )

        dev.off()
    } else if (scale == 5) { # log n vs log sqrt t
        save_path <- "alpha_shape_timing_log_n_vs_log_sqrt_t.png"
        png(save_path)
        plot(log(data$num_points), log(sqrt(data$time_taken_seconds)),
            type = "b",
            xlab = "Log(Number of Points)", ylab = "Log(Square Root of Time Taken (seconds))",
            main = "Alpha Shape Computation Time vs Number of Points (log-log sqrt scale)"
        )
        dev.off()
    }
}

sample_fraction_variations <- function(
    fraction_list = c(0.5, 0.6, 0.7, 0.8, 0.9, 0.99),
    dim = 2,
    alpha = 1.0,
    CL = 0.95,
    num_points = 10000,
    noise_in_data = 0.01,
    shape = "ring",
    r_outer = 5,
    r_inner = 3,
    num_rings = 2,
    spacing = 3.0,
    ring_width = 1.5) {
    data_results <- data.frame()
    for (fraction in fraction_list) {
        result <- main_test(dim = dim, alpha = alpha, CL = CL, num_points = num_points, noise_in_data = noise_in_data, shape = shape, spacing = spacing, ring_width = ring_width, num_rings = num_rings, r_outer = r_outer, r_inner = r_inner, sampling_fraction = fraction, title = paste("Alpha Shape Visualization - Sampling Fraction:", fraction))
        print(paste("Completed for Sampling Fraction:", fraction))
        # save to csv
        data_results <- rbind(data_results, data.frame(sampling_fraction = fraction, KL_divergence = result$KL_divergence, IOU_score = result$IOU_score))
    }

    write.csv(data_results, file = "alpha_shape_sampling_fraction_results.csv", row.names = FALSE)
}

corner_variations <- function(s = 3.0, w = 1.5, num_rings = 3, alpha_list = c(1.0), CL_list = c(0.95), num_points = 10000, noise_in_data_list = c(0.01)) {
    result_dataframe <- data.frame()

    for (alpha in alpha_list) {
        for (CL in CL_list) {
            for (noise_in_data in noise_in_data_list) {
                print(paste("Running for Alpha:", alpha, "CL:", CL, "Noise:", noise_in_data))
                title <- paste("Alpha Shape Visualization - Alpha:", alpha, "CL:", CL, "Noise:", noise_in_data)
                result <- main_test(dim = 2, alpha = alpha, CL = CL, num_points = num_points, noise_in_data = noise_in_data, title = title, shape = "squares", spacing = s, ring_width = w, num_rings = num_rings)
                KL_divergence <- result$KL_divergence
                IOU_score_value <- result$IOU_score
                result_dataframe <- rbind(result_dataframe, data.frame(alpha = alpha, CL = CL, noise = noise_in_data, KL_divergence = KL_divergence, IOU_score = IOU_score_value))
            }
        }
    }

    write.csv(result_dataframe, file = "corner_variations_results.csv", row.names = FALSE)
}

# set.seed(137)

run_conc_circles <- FALSE
run_squares <- TRUE
run_conc_circles_backup <- FALSE

# alpha_sample_list <- seq(0.1, 2.0, by = 0.2)
alpha_sample_list <- c(0.5, 1.0)
noise_sample_list <- c(0.01, 0.001)

specific_parameter_list <- list(
    # alpha, CL, noise, shape

    # squares
    # c(1.0, 1.0, 0, "squares"),
    # c(1.0, 1.0, 0.01, "squares"),
    # c(1.0, 0.95, 0, "squares"),
    # c(1.0, 0.95, 0.01, "squares"),

    # concentric rings
    # change alpha
    c(1.0, 1.0, 0, "concentric"),
    c(0.5, 1.0, 0, "concentric"),
    c(2.0, 1.0, 0, "concentric"),
    c(1.5, 1.0, 0, "concentric"),
    # add noise
    c(1.0, 1.0, 0.01, "concentric"),
    c(1.0, 1.0, 0.05, "concentric"),
    c(1.0, 1.0, 0.001, "concentric"),
    # change CL
    c(1.0, 0.95, 0.01, "concentric"),
    c(1.0, 0.90, 0.01, "concentric"),
    # c(1.0, 1.0, 0.01, "concentric"), -> reuse above
    # different alpha with noise and CL
    c(0.5, 0.95, 0.01, "concentric"),
    c(2.0, 0.95, 0.01, "concentric"),
    c(1.5, 0.95, 0.01, "concentric")
)

used_points <- 10000
spacing <- 3.0
ring_width <- 1.5
num_rings <- 3


print("Starting alpha variations test...")

if (run_conc_circles) {
    alpha_variations(
        alpha_value_list = alpha_sample_list,
        dim = 2,
        CL_list = c(1.0, 0.95),
        num_points = used_points,
        num_rings = num_rings,
        ring_width = ring_width,
        spacing = spacing,
        noise_in_data = noise_sample_list,
        shape = "concentric",
        specific_parameters = specific_parameter_list,
        save_path = "alpha_variations_concentric_results.csv"
    )
}

corner_analysis_list <- list(
    # alpha CL noise shape
    c(1.0, 1.0, 0.0, "concentric"),
    c(1.0, 1.0, 0.01, "concentric"),
    c(1.0, 0.95, 0.0, "concentric"),
    c(1.0, 0.95, 0.01, "concentric"),
    c(1.0, 1.0, 0.0, "squares"),
    c(1.0, 1.0, 0.01, "squares"),
    c(1.0, 0.95, 0.0, "squares"),
    c(1.0, 0.95, 0.01, "squares")
    
)

backup_concentric <- list(
    # alpha CL noise shape
    c(1.0, 0.95, 0.0, "concentric")
)





if (run_conc_circles_backup) {
    print("Starting backup concentric circles test...")

    alpha_variations(
        alpha_value_list = alpha_sample_list,
        dim = 2,
        CL_list = c(0.95),
        num_points = used_points,
        num_rings = num_rings,
        ring_width = ring_width,
        spacing = spacing,
        noise_in_data = c(0.0),
        shape = "concentric",
        specific_parameters = backup_concentric,
        save_path = "backup_concentric_results.csv"
    )
    print("Backup concentric circles test completed.")
}

if (run_squares) {
    print("Starting corner variations test...")

    alpha_variations(
        alpha_value_list = alpha_sample_list,
        dim = 2,
        CL_list = c(1.0, 0.95),
        num_points = 1000,
        num_rings = 2,
        r_outer = 5,
        r_inner = 3,
        spacing = spacing,
        ring_width = ring_width,
        shape = "squares",
        noise_in_data = noise_sample_list,
        specific_parameters = corner_analysis_list,
        save_path = "corner_analysis_results.csv"
    )
    print("corner analysis test completed.")
}


print("All Alpha variations test completed.")
print("Starting complexity time measurements...")
# datapoints_variations(num_points_list = round(10^seq(1, 4, length.out = 10)))

plot_datapoints_timing(csv_file = "alpha_shape_datapoints_timing.csv", scale = 4)

print("Complexity time measurements completed.")

print("Starting sample fraction variations test...")
# sample_fraction_variations(fraction_list = c(0.5, 0.6, 0.7, 0.8, 0.85,  0.9, 0.95, 0.99), dim = 2, alpha = 1.0, CL = 0.95, num_points = 2000, noise_in_data = 0.01, shape = "squares", r_outer = 5, r_inner = 3, num_rings = num_rings, spacing = spacing, ring_width = ring_width)

print("Sample fraction variations test completed.")

print("All tests completed.")
