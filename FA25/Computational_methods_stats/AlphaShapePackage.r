#options(repos = c(CRAN = "https://cran.rstudio.com/"))

# nolint start

#install.packages("sp")
#install.packages("roxygen2")
#install.packages("dbscan")

# import libraries
library(sp) 
#library(roxygen2)
library(dbscan)

#devtools::create("AlphaShapePackage")
#roxygen2::roxygenise("AlphaShapePackage")

# Helper functions

circumcircle <- function(p1, p2, p3) {

    A <- p2 - p1
    B <- p3 - p1
    d <- 2*(A[1]*B[2] - A[2]*B[1])

    if (abs(d) < 1e-16) {
        return(list(center = c(NA, NA), r = Inf))
    }

    ux <- ((B[2]*(A[1]^2 + A[2]^2) - A[2]*(B[1]^2 + B[2]^2)) / d) + p1[1]
    uy <- ((A[1]*(B[1]^2 + B[2]^2) - B[1]*(A[1]^2 + A[2]^2)) / d) + p1[2]
    center <- c(ux, uy)
    r <- sqrt((center[1] - p1[1])^2 + (center[2] - p1[2])^2)
    return(list(center = center, r = r))
}

point_in_circle <- function(p, center, r) {
    # Function to check if a point is inside a circle
    # p: a vector of length 2 (x, y) coordinates of the point
    # center: a vector of length 2 (x, y) coordinates of the circle center
    # r: radius of the circle
    return(sqrt((p[1] - center[1])^2 + (p[2] - center[2])^2) < r)
}

circumsphere <- function(p1, p2, p3, p4) {
    # Function to calculate the circumsphere of a tetrahedron defined by points p1, p2, p3, p4
    # p1, p2, p3, p4: vectors of length 3 (x, y, z) coordinates of the points

    A <- rbind(p2 - p1, p3 - p1, p4 - p1)
    B <- c(sum((p2 - p1)^2), sum((p3 - p1)^2), sum((p4 - p1)^2)) / 2

    center <- solve(A, B) + p1
    r <- sqrt(sum((center - p1)^2))
    return(list(center = center, r = r))
}

point_in_sphere <- function(p, center, r) {
    # Function to check if a point is inside a sphere
    # p: a vector of length 3 (x, y, z) coordinates of the point
    # center: a vector of length 3 (x, y, z) coordinates of the sphere center
    # r: radius of the sphere
    return(sqrt(sum((p - center)^2)) < r)
}

deluanay_triangulation <- function(points, dim = 2) {
    if (dim != 2 && dim != 3) {
        stop("Dimension must be 2 or 3.")
    }
    if (dim == 2) {

        points <- as.matrix(points)
        if (ncol(points) != 2) {
        stop("Input points must have exactly two columns for x and y coordinates.")
        }

        min_x <- min(points[,1])
        max_x <- max(points[,1])
        min_y <- min(points[,2])
        max_y <- max(points[,2])

        dx <- max_x - min_x
        dy <- max_y - min_y

        delta_max <- max(dx, dy)
        mid_x <- (min_x + max_x) / 2
        mid_y <- (min_y + max_y) / 2

        super <- rbind(
        c(mid_x - 20 * delta_max, mid_y - delta_max),
        c(mid_x, mid_y + 20 * delta_max),
        c(mid_x + 20 * delta_max, mid_y - delta_max)
        )

        triangles <- matrix(NA, nrow = 1, ncol = 3)
        points_all <- rbind(points, super)

        n <- nrow(points)
        super_idx <- (n + 1):(n + 3)
        triangles[1, ] <- super_idx

        for (i in 1:n) {
        point <- points_all[i, ]
        bad_triangles <- c()
        circumcenters <- list()
        circumradii <- c()

        for (j in 1:nrow(triangles)) {
            tri <- triangles[j, ]
            p1 <- points_all[tri[1], ]
            p2 <- points_all[tri[2], ]
            p3 <- points_all[tri[3], ]

            cc <- circumcircle(p1, p2, p3)
            circumcenters[[j]] <- cc$center
            circumradii[j] <- cc$r

            if (point_in_circle(point, cc$center, cc$r)) {
            bad_triangles <- c(bad_triangles, j)
            }
        }

        edge_count <- list()
        for (bt in bad_triangles) {
            tri <- triangles[bt, ]
            edges <- list(
            paste(sort(c(tri[1], tri[2])), collapse = "-"),
            paste(sort(c(tri[2], tri[3])), collapse = "-"),
            paste(sort(c(tri[3], tri[1])), collapse = "-")
            )
            for (edge in edges) {
            if (is.null(edge_count[[edge]])) {
                edge_count[[edge]] <- 1
            } else {
                edge_count[[edge]] <- edge_count[[edge]] + 1
            }
            }
        }

        triangles <- triangles[-bad_triangles, , drop = FALSE]

        for (edge in names(edge_count)) {
            if (edge_count[[edge]] == 1) {
            verts <- as.numeric(unlist(strsplit(edge, "-")))
            new_tri <- c(verts, i)
            triangles <- rbind(triangles, new_tri)
            }
        }
        }

        triangles <- triangles[!apply(triangles, 1, function(tri) any(tri %in% super_idx)), , drop = FALSE]
        return(triangles)


    } else {
        
        points <- as.matrix(points)
        if (ncol(points) != 3) {
        stop("Input points must have exactly three columns for x, y, and z coordinates.")
        }

        n <- nrow(points)
        min_x <- min(points[,1])
        max_x <- max(points[,1])
        min_y <- min(points[,2])
        max_y <- max(points[,2])
        min_z <- min(points[,3])
        max_z <- max(points[,3])

        dx <- max_x - min_x
        dy <- max_y - min_y
        dz <- max_z - min_z
        delta_max <- max(dx, dy, dz)
        mid_x <- (min_x + max_x) / 2
        mid_y <- (min_y + max_y) / 2
        mid_z <- (min_z + max_z) / 2

        super <- rbind(
        c(mid_x - 20 * delta_max, mid_y - 20 * delta_max, mid_z - 20 * delta_max),
        c(mid_x + 20 * delta_max, mid_y + 20 * delta_max, mid_z - 20 * delta_max),
        c(mid_x + 20 * delta_max, mid_y - 20 * delta_max, mid_z + 20 * delta_max),
        c(mid_x - 20 * delta_max, mid_y + 20 * delta_max, mid_z + 20 * delta_max)
        )

        points_all <- rbind(points, super)
        super_idx <- (n + 1):(n + 4)

        tetrahedra <- matrix(NA, nrow = 1, ncol = 4)
        tetrahedra[1, ] <- super_idx

        for (i in 1:n) {
        point <- points_all[i, ]
        bad_tetrahedra <- c()
        circumcenters <- list()
        circumradii <- c()

        for (j in 1:nrow(tetrahedra)) {
            tet <- tetrahedra[j, ]
            p1 <- points_all[tet[1], ]
            p2 <- points_all[tet[2], ]
            p3 <- points_all[tet[3], ]
            p4 <- points_all[tet[4], ]

            cs <- circumsphere(p1, p2, p3, p4)
            circumcenters[[j]] <- cs$center
            circumradii[j] <- cs$r

            if (point_in_sphere(point, cs$center, cs$r)) {
            bad_tetrahedra <- c(bad_tetrahedra, j)
            }
        }

        face_count <- list()
        for (bt in bad_tetrahedra) {
            tet <- tetrahedra[bt, ]
            faces <- list(
            paste(sort(c(tet[1], tet[2], tet[3])), collapse = "-"),
            paste(sort(c(tet[1], tet[2], tet[4])), collapse = "-"),
            paste(sort(c(tet[1], tet[3], tet[4])), collapse = "-"),
            paste(sort(c(tet[2], tet[3], tet[4])), collapse = "-")
            )
            for (face in faces) {
            if (is.null(face_count[[face]])) {
                face_count[[face]] <- 1
            } else {
                face_count[[face]] <- face_count[[face]] + 1
            }
            }
        }

        tetrahedra <- tetrahedra[-bad_tetrahedra, , drop = FALSE]

        for (face in names(face_count)) {
            if (face_count[[face]] == 1) {
            verts <- as.numeric(unlist(strsplit(face, "-")))
            new_tet <- c(verts, i)
            tetrahedra <- rbind(tetrahedra, new_tet)
            }
        }
        }

        tetrahedra <- tetrahedra[!apply(tetrahedra, 1, function(tet) any(tet %in% super_idx)), , drop = FALSE]

        return(tetrahedra)

    }
}

polygon_area <- function(polygon) {
    if (!is.matrix(polygon) || ncol(polygon) != 2) {
    stop("Input polygon must be a matrix with two columns (x and y coordinates).")
    }

    # Calculate the area using the shoelace formula
    x <- polygon[, 1]
    y <- polygon[, 2]

    n <- length(x)
    if (n < 3) {
        return(0)  # Not a polygon
    }

    area <- 0.5 * abs(sum(x * c(y[-1], y[1]) - y * c(x[-1], x[1])))
    return(area)
}

edge_length <- function(p1, p2) {
    return(sqrt(sum((p1 - p2)^2)))
}

triangle_area <- function(p1, p2, p3) {
    a <- edge_length(p1, p2)
    b <- edge_length(p2, p3)
    c <- edge_length(p3, p1)
    s <- (a + b + c) / 2
    area <- sqrt(s * (s - a) * (s - b) * (s - c))
    return(area)
}

tetrahedron_volume <- function(p1, p2, p3, p4) {
    mat <- rbind(
    c(1, p1),
    c(1, p2),
    c(1, p3),
    c(1, p4)
    )
    vol <- abs(det(mat)) / 6
    return(vol)
}

tetrahedron_surface_area <- function(p1, p2, p3, p4) {
    area1 <- triangle_area(p1, p2, p3)
    area2 <- triangle_area(p1, p2, p4)
    area3 <- triangle_area(p1, p3, p4)
    area4 <- triangle_area(p2, p3, p4)
    total_area <- area1 + area2 + area3 + area4
    return(total_area)
}

edges_to_loops <- function(boundary_edges, points) {
  # boundary_edges: list of numeric vectors length 2 (vertex indices)
  # points: original points matrix (n x 2)
  # returns: list of polygons, each polygon is a matrix (m x 2) of coordinates (closed: last == first)
  if (length(boundary_edges) == 0) return(list())

  # map edge key to logical used flag
  edge_keys <- sapply(boundary_edges, function(e) {
    paste(sort(e), collapse = "-")
  }, USE.NAMES = FALSE)

  edge_used <- setNames(rep(FALSE, length(edge_keys)), edge_keys)

  # adjacency: vertex -> neighbor indices (list of integer vectors)
  adj <- list()
  for (e in boundary_edges) {
    a <- as.character(e[1]); b <- as.character(e[2])
    adj[[a]] <- unique(c(adj[[a]], e[2]))
    adj[[b]] <- unique(c(adj[[b]], e[1]))
  }

  loops <- list()

  # function to mark edge used by its key
  mark_edge_used <- function(u, v) {
    k <- paste(sort(c(u, v)), collapse = "-")
    edge_used[k] <<- TRUE
  }
  is_edge_used <- function(u, v) {
    k <- paste(sort(c(u, v)), collapse = "-")
    edge_used[k]
  }

  # iterate until all edges used
  while (any(!edge_used)) {
    # find first unused edge key and its vertices
    unused_keys <- names(edge_used)[!edge_used]
    key0 <- unused_keys[1]
    verts <- as.numeric(unlist(strsplit(key0, "-")))
    start <- verts[1]
    next_v <- verts[2]

    # start walking the loop
    loop_idx <- c(start, next_v)
    mark_edge_used(start, next_v)
    prev_v <- start
    cur_v <- next_v

    # walk until we return to start
    safety <- 0
    while (cur_v != start && safety < 10000) {
      safety <- safety + 1
      neighs <- adj[[as.character(cur_v)]]
      # choose neighbor not equal to prev_v and where edge is not already used, if possible
      candidate <- NA
      for (nv in neighs) {
        if (nv == prev_v) next
        if (!is.na(nv) && !is_edge_used(cur_v, nv)) { candidate <- nv; break }
      }
      # if all remaining neighbors lead to used edges (rare due to boundary structure), choose any neighbor that is not prev_v
      if (is.na(candidate)) {
        for (nv in neighs) {
          if (nv != prev_v) { candidate <- nv; break }
        }
      }
      if (is.na(candidate)) break

      loop_idx <- c(loop_idx, candidate)
      mark_edge_used(cur_v, candidate)
      prev_v <- cur_v
      cur_v <- candidate
    }
    # close the loop by ensuring first == last
    if (loop_idx[1] != loop_idx[length(loop_idx)]) loop_idx <- c(loop_idx, loop_idx[1])
    # convert to coordinates matrix
    poly_coords <- points[loop_idx, , drop = FALSE]
    loops[[length(loops) + 1]] <- poly_coords
  }

  return(loops)
}

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

  if (is.null(epsilon)) epsilon <- auto_epsilon(points, min_points, quantile_epsilon)

  inclusion_counts <- integer(n)
  sample_counts <- integer(n)     # track how many times each point was sampled

  for (b in 1:B) {
    sample_size <- floor(n**sample_fraction)
    sample_idx <- sample(1:n, sample_size)
    sample_points <- points[sample_idx, , drop = FALSE]

    # update how often each index was sampled
    sample_counts[sample_idx] <- sample_counts[sample_idx] + 1L

    # run DBSCAN on the subsample
    db <- dbscan::dbscan(sample_points, eps = epsilon, minPts = min_points)

    # map clustered (non-noise) back to original indices
    core_points_idx <- sample_idx[which(db$cluster != 0)]
    inclusion_counts[core_points_idx] <- inclusion_counts[core_points_idx] + 1L
  }

  # Build per-point threshold based on how often it was sampled
  per_point_threshold <- ceiling(sample_counts * CL)

  # Points that were never sampled should not be selected
  valid <- which(sample_counts > 0)

  # Select indices meeting their per-point threshold
  filtered_idx <- valid[ which(inclusion_counts[valid] >= per_point_threshold[valid]) ]

  return(points[filtered_idx, , drop = FALSE])
}

alpha_shape_2D <- function(points, alpha, CL = 1.0, sampling_fraction = 0.95) {
  points <- as.matrix(points)
  if (ncol(points) != 2) stop("points must be n x 2 for 2D alpha shape")

  if (CL < 1) {
    num_points <- max(4, floor(nrow(points) * CL))
    points <- dbscan_filter(points, min_points = 5, CL = CL, sample_fraction = sampling_fraction)
  }

  if (nrow(points) < 4) {
    warning("Not enough points to form an alpha shape after filtering.")
    return(list(polygons = list(), points = points))
  }

  triangles <- deluanay_triangulation(points, dim = 2)
  if (is.null(triangles) || nrow(triangles) == 0) return(list())

  edge_count <- list()
  keep_tri_idx <- c()
  for (i in 1:nrow(triangles)) {
    tri <- triangles[i, ]
    p1 <- points[tri[1], ]
    p2 <- points[tri[2], ]
    p3 <- points[tri[3], ]

    cc <- circumcircle(p1, p2, p3)
    # skip degenerate (collinear) triangles: cc$r may be Inf
    if (is.finite(cc$r) && cc$r < alpha) {
      # keep the triangle
      keep_tri_idx <- c(keep_tri_idx, i)
      # count its edges
      edges <- list(
        paste(sort(c(tri[1], tri[2])), collapse = "-"),
        paste(sort(c(tri[2], tri[3])), collapse = "-"),
        paste(sort(c(tri[3], tri[1])), collapse = "-")
      )
      for (e in edges) {
        if (is.null(edge_count[[e]])) edge_count[[e]] <- 1 else edge_count[[e]] <- edge_count[[e]] + 1
      }
    }
  }

  # boundary edges are those that appear exactly once among kept triangles
  boundary_keys <- names(edge_count)[sapply(edge_count, function(x) x == 1)]
  if (length(boundary_keys) == 0) {
    # No boundary edges found - this means either no triangles kept or all triangles are internal
    if (length(keep_tri_idx) == 0) {
      warning(paste("No triangles kept with alpha =", alpha, ". Try increasing alpha value."))
    }
    return(list(polygons = list(), points = points))
  }

  # convert keys back to pairs
  boundary_edges <- lapply(boundary_keys, function(k) as.numeric(unlist(strsplit(k, "-"))))

  # build ordered loops (polygons)
  loops <- edges_to_loops(boundary_edges, points)
  
  if (length(loops) == 0) {
    warning("No loops formed from boundary edges.")
    return(list(polygons = list(), points = points))
  }

  # sort loops by absolute area descending (outer loop is largest by area)
  areas <- sapply(loops, function(poly) {
    # expect poly is m x 2 matrix (closed); polygon_area expects matrix of coords
    polygon_area(poly)
  })
  order_idx <- order(areas, decreasing = TRUE)
  loops <- loops[order_idx]

  return(list(polygons = loops, points = points))
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
    plot(points, col = "black", pch = 19, main = title, asp = 1,
         xlab = "x", ylab = "y",
         bg = NA)  # transparent background
  } else {
    points(points, col = "black", pch = 19)
  }

  if (length(loops) == 0) return(invisible(NULL))

  # ---- Alpha-shape outlines (blue) ----
  for (i in seq_along(loops)) {
    loop <- loops[[i]]
    polygon(loop, border = "blue", lwd = 2, col = NA)
  }

    # ---- True shape outlines (red) ----
  if (shape == "ring" && !is.null(r_outer) && !is.null(r_inner)) {
    # outline between the two circles to represent the ring
    theta <- seq(0, 2*pi, length.out = 400)
    lines(r_outer * cos(theta), r_outer * sin(theta),
          col = "red", lwd = 2)

    lines(r_inner * cos(theta), r_inner * sin(theta),
          col = "red", lwd = 2)
  } else if (shape == "concentric" && !is.null(num_rings) && !is.null(ring_width) && !is.null(spacing)) {
    # outline for multiple concentric rings
    r_inner <- spacing
    for (j in 1:num_rings) {
        r_outer <- r_inner + ring_width
        theta <- seq(0, 2*pi, length.out = 400)
        lines(r_outer * cos(theta), r_outer * sin(theta), col = "red", lwd = 2)
        lines(r_inner * cos(theta), r_inner * sin(theta), col = "red", lwd = 2)
        r_inner <- r_outer + spacing
    }
  } else if(shape == "squares" && !is.null(r_outer) && !is.null(r_inner)) {
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


calculate_area_volume <- function(alpha_shape, dim = 2) {
  if (dim == 2) {
    if (is.null(alpha_shape$polygons) || length(alpha_shape$polygons) == 0) {
      return(0)
    }

    # Calculate signed areas for each polygon
    signed_areas <- sapply(alpha_shape$polygons, function(poly) {
        if (nrow(poly) < 3) {
          return(0)
        }
        x <- poly[,1]; y <- poly[,2]
        0.5 * sum(x * c(y[-1], y[1]) - y * c(x[-1], x[1]))
    })

    if (all(signed_areas == 0)) {
      return(0)
    }

    # Use signed areas - this handles holes correctly when orientations are proper
    total_signed_area <- sum(signed_areas)
    
    # If total is very close to zero, try alternative calculation
    if (abs(total_signed_area) < 1e-10) {
        # Use absolute areas - assume largest is outer boundary
        abs_areas <- abs(signed_areas)
        if (length(abs_areas) == 1) {
            result <- abs_areas[1]
        } else {
            # Largest polygon is outer boundary, others are holes
            result <- max(0, abs_areas[1] - sum(abs_areas[-1]))
        }
    } else {
        result <- abs(total_signed_area)
    }
    
    return(result)
  } else {
    stop("3D alpha shape volume not yet implemented.")
  }
}

calculate_surface_perimeter <- function(alpha_shape, dim = 2) {
    # Function to calculate surface area (3D) or perimeter (2D) of the alpha shape
    # alpha_shape: the output from alpha_shape_2D or alpha_shape_3D

    if (dim == 2) {
        return(polygon_perimeter(alpha_shape))
    } else {
        return(tetrahedron_surface_area(alpha_shape))
    }
}


# Test generation functions

torus_test_generation_3D <- function(num_points = 10000, R = 3, r = 1, noise = 0.05) {
    # Function to generate points on a torus
    # num_points: number of points to generate
    # R: major radius
    # r: minor radius

    u <- runif(num_points, 0, 2 * pi)
    v <- runif(num_points, 0, 2 * pi)

    rho <- r * sqrt(runif(num_points, 0, 1))

    x <- (R + rho * cos(v)) * cos(u)
    y <- (R + rho * cos(v)) * sin(u)
    z <- rho * sin(v)

    if (noise > 0) {
        x <- x + rnorm(num_points, 0, sd = noise)
        y <- y + rnorm(num_points, 0, sd = noise)
        z <- z + rnorm(num_points, 0, sd = noise)
    }

    points <- cbind(x, y, z)
    return(points)
}

ring_test_generation_2D <- function(num_points = 10000, r_outer = 3, r_inner = 1, noise = 0.05) {
    # Function to generate points on a ring shape
    # num_points: number of points to generate
    # r_outer: outer radius
    # r_inner: inner radius

    n_noise <- floor(noise * num_points)
    n_ring  <- num_points - n_noise

    theta <- runif(n_ring, 0, 2 * pi)
    rho <- r_inner + (r_outer - r_inner) * runif(n_ring, 0, 1)

    x_ring <- rho * cos(theta)
    y_ring <- rho * sin(theta)

    if (noise > 0) {
        x_noise <- runif(n_noise, -r_outer, r_outer)
        y_noise <- runif(n_noise, -r_outer, r_outer)

        x <- c(x_ring, x_noise)
        y <- c(y_ring, y_noise)
    } else{
        x <- x_ring
        y <- y_ring
    }

    points <- cbind(x, y)
    return(points)
}

concentric_rings_2D <- function(num_rings = 3, spacing = 1, ring_width=1.0, total_points = 1000, noise = 0.01){

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

    if (noise > 0) {
      noisy_points <- floor(noise * total_points)
      total_points <- total_points - noisy_points
    }

    points_per_square_ratio <- seq(1, num_squares) / sum(seq(1, num_squares))

    center_x <- 0
    center_y <- 0

    for (i in 1:(num_squares)) {
       
        inner_k <- spacing + 2 * (i - 1) * (square_width + spacing)
        outer_k <- spacing + 2 * (i - 1) * (square_width + spacing) + 2 * square_width

        num_square_points <- floor(points_per_square_ratio[i] * total_points)

        square_points <- matrix(0, nrow = 0, ncol = 2)

        k <- 0
        while (k < num_square_points) {
            
            x_candidate <- runif(1, -outer_k, outer_k)
            y_candidate <- runif(1, -outer_k, outer_k)

            max_value <- max(abs(x_candidate), abs(y_candidate))

            if (max_value >= inner_k && max_value <= outer_k) {
                k <- k + 1
                good_point <- c(x_candidate, y_candidate)
                square_points <- rbind(square_points, good_point)
            } else {
                next
            }
        }      

        points <- rbind(points, square_points)
    }

    # Add noise points
    if (noise > 0) {
        x_noise <- runif(noisy_points, min(points[,1]) - spacing, max(points[,1]) + spacing)
        y_noise <- runif(noisy_points, min(points[,2]) - spacing, max(points[,2]) + spacing)
        noise_points <- cbind(x_noise, y_noise)
        points <- rbind(points, noise_points)
    }

    return(points)
}

square_area_2D <- function(num_squares = 5, square_width = 0.5, spacing = 1, center_x = 0, center_y = 0) {
    total_area <- 0
    for (i in 0:(num_squares - 1)) {
        x_inner <- center_x + spacing/2 + (i-1)*(square_width + spacing)
        x_outer <- center_x + spacing/2 + (i-1)*(square_width + spacing) + square_width
        y_inner <- center_y + spacing/2 + (i-1)*(square_width + spacing)
        y_outer <- center_y + spacing/2 + (i-1)*(square_width + spacing) + square_width

        area_square <- (2*x_outer * 2*y_outer) - (2*x_inner * 2*y_inner)
        total_area <- total_area + area_square
    }
    return(total_area)
}

alternating_squares_polygon <- function(num_squares = 5,
                                        square_width = 0.5,
                                        spacing = 1,
                                        center_x = 0,
                                        center_y = 0) {
    
    polygon_list <- list()

    for (i in 1:num_squares) {
        
        inner_k <- spacing/2 + (i - 1) * (square_width + spacing)
        outer_k <- inner_k + square_width
        
        # Outer boundary (CCW)
        outer <- rbind(
            c(center_x - outer_k, center_y + outer_k),
            c(center_x + outer_k, center_y + outer_k),
            c(center_x + outer_k, center_y - outer_k),
            c(center_x - outer_k, center_y - outer_k),
            c(center_x - outer_k, center_y + outer_k)   # close
        )
        
        # Inner boundary (CW for a "hole")
        inner <- rbind(
            c(center_x - inner_k, center_y + inner_k),
            c(center_x - inner_k, center_y - inner_k),
            c(center_x + inner_k, center_y - inner_k),
            c(center_x + inner_k, center_y + inner_k),
            c(center_x - inner_k, center_y + inner_k)   # close
        )
        inner <- inner[nrow(inner):1, ]                 # reverse to CW

        # Combine into a single polygon structure
        poly <- rbind(outer, inner)

        polygon_list[[i]] <- poly
    }
    
    return(polygon_list)
}


pitchfork_bifurcation_datacloud <- function(num_points = 1000, noise = 0.05, direction ="up") {

    # base equation: dx = r*x - x^3

    points <- matrix(0, nrow = num_points, ncol = 2)

    positive_points <- matrix(0, nrow = 0, ncol = 2)
    negative_points <- matrix(0, nrow = 0, ncol = 2)

    num_noise <- floor(noise * num_points)

    for (i in 1:num_points){
        x <- runif(1, -2, 2)
        r <- runif(1, -2, 2)
        y <- r * x - x^3
        # assign (r,x) to (x,y) depending on sign(y)
        if (y >=0){
            positive_points <- rbind(positive_points, c(x, y))
        } else {
            negative_points <- rbind(negative_points, c(x, y))
        }

    }

    # add noise points
    if (num_noise > 0) {
        x_noise <- runif(num_noise, -2, 2)
        y_noise <- runif(num_noise, -2, 2)
        noise_points <- cbind(x_noise, y_noise)

        positive_points <- rbind(positive_points, noise_points)
        negative_points <- rbind(negative_points, noise_points)
    } else {
        positive_points <- positive_points
        negative_points <- negative_points
    }

    if (direction == "up") {
        return(positive_points)
    } else if (direction == "down") {
        return(negative_points)
    } else {
        return(points)
    }
}

polygon_pitchfork <- function(x_min = -2, x_max = 2, r_min = -2, r_max = 2) {
    # create polygon representing the pitchfork bifurcation shape
    num_points <- 200
    x_seq <- seq(x_min, x_max, length.out = num_points)
    upper_branch <- cbind(x_seq, (r_max * x_seq - x_seq^3))
    lower_branch <- cbind(rev(x_seq), (r_min * rev(x_seq) - rev(x_seq)^3))

    polygon_coords <- rbind(upper_branch, lower_branch)
    return(polygon_coords)
}

area_concentric_rings_2D <- function(num_rings = 3, spacing = 1, ring_width=0.5){
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

heart_shape_datacloud <- function(num_points = 1000, num_rings = 3, spacing = 1, ring_width=0.5, noise = 0.05){

    points <- matrix(0, nrow = num_points, ncol = 2)

    if (noise > 0){
      pass
    }



} 

# error calculation

error_in_2D <- function(true_area, estimated_area) {
    return(abs(true_area - estimated_area) / true_area * 100)
}

calculate_KL_divergence <- function(true_shape, estimated_shape, dim = 2, grid_res = 10000, num_rings = 3, spacing = 3.0, width = 1.5, shape = "concentric") {
    # Function to calculate KL divergence between true shape and estimated alpha shape
    # true_shape: the true shape representation (e.g., polygon for 2D)
    # estimated_shape: the estimated alpha shape representation
    # dim: dimension (2 or 3)

    if (dim == 2) {
          if (length(true_shape$polygons) == 0 || length(estimated_shape$polygons) == 0)
            return(NA)

        

        # bounding box
        all_points <- rbind(do.call(rbind, true_shape$polygons), do.call(rbind, estimated_shape$polygons))
        x_range <- range(all_points[,1])
        y_range <- range(all_points[,2])

        x_seq <- seq(x_range[1], x_range[2], length.out = grid_res)
        y_seq <- seq(y_range[1], y_range[2], length.out = grid_res)

        # create a Monte Carlo grid

       # grid <- expand.grid(x = x_seq, y = y_seq)
        grid <- data.frame(x = runif(grid_res * grid_res, min = x_range[1], max = x_range[2]),y = runif(grid_res * grid_res, min = y_range[1], max = y_range[2]))
        
        

        # helper to check if points are in any polygon
        points_in_polygons <- function(points, polygons) {
            inside <- rep(FALSE, nrow(points))
            for (poly in polygons) {
            inside <- inside | sp::point.in.polygon(points$x, points$y, poly[,1], poly[,2]) > 0
            }
            return(inside)
        }

        if (shape == "squares"){
            inside_est <- points_in_polygons(grid, estimated_shape$polygons)
            inside_true <- c()
            # for grid points, check if in any of the square rings using the math
            for (i in 1:nrow(grid)){
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
        if (length(true_shape$polygons) == 0 || length(estimated_shape$polygons) == 0)
            return(0)

        # bounding box
        all_points <- rbind(do.call(rbind, true_shape$polygons), do.call(rbind, estimated_shape$polygons))
        x_range <- range(all_points[,1])
        y_range <- range(all_points[,2])

        x_seq <- seq(x_range[1], x_range[2], length.out = grid_res)
        y_seq <- seq(y_range[1], y_range[2], length.out = grid_res)

        # create a Monte Carlo grid
        grid <- expand.grid(x = x_seq, y = y_seq)

        # helper to check if points are in any polygon
        points_in_polygons <- function(points, polygons) {
            inside <- rep(FALSE, nrow(points))
            for (poly in polygons) {
            inside <- inside | sp::point.in.polygon(points$x, points$y, poly[,1], poly[,2]) > 0
            }
            return(inside)
        }

        if (shape == "squares"){
            inside_est <- points_in_polygons(grid, estimated_shape$polygons)
            inside_true <- c()
            # for grid points, check if in any of the square rings using the math
            for (i in 1:nrow(grid)){
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

    if (dim == 2){
        print("Generating 2D test data...")
        if (shape == "ring") {
            points <- ring_test_generation_2D(num_points = num_points, noise = noise_in_data, r_outer = r_outer, r_inner = r_inner)
            true_area <- pi * (r_outer^2 - r_inner^2)
            true_shape <- list(polygons = list(
                matrix(c(r_outer * cos(seq(0, 2*pi, length.out = 100)),
                        r_outer * sin(seq(0, 2*pi, length.out = 100))), ncol = 2),
                matrix(c(r_inner * cos(seq(0, 2*pi, length.out = 100)),
                        r_inner * sin(seq(0, 2*pi, length.out = 100))), ncol = 2)
            ))
        } else if (shape == "concentric") {
            points <- concentric_rings_2D(num_rings = num_rings, spacing = spacing,
                                          ring_width = ring_width, total_points = num_points, noise = noise_in_data)

            # Build true polygons with proper holes
            true_shape <- list(polygons = list())
            r_inner <- spacing
            for (i in 1:num_rings) {
            r_outer <- r_inner + ring_width
            # Outer polygon (counterclockwise)
            true_shape$polygons[[length(true_shape$polygons) + 1]] <- cbind(
                r_outer * cos(seq(0, 2*pi, length.out = 200)),
                r_outer * sin(seq(0, 2*pi, length.out = 200))
            )
            # Inner polygon (clockwise)
            true_shape$polygons[[length(true_shape$polygons) + 1]] <- cbind(
                r_inner * cos(seq(0, -2*pi, length.out = 200)),
                r_inner * sin(seq(0, -2*pi, length.out = 200))
            )
            r_inner <- r_outer + spacing
            }

            # True area sum
            true_area <- area_concentric_rings_2D(num_rings = num_rings,
                                                spacing = spacing, ring_width = ring_width)
        } else if (shape=="pitchfork") {
            points <- pitchfork_bifurcation_datacloud(num_points = num_points, noise = noise_in_data, direction ="up")
            true_shape <- list(polygons = list(
                polygon_pitchfork(x_min = -2, x_max = 2, r_min = -2, r_max = 2)
            ))
            true_area <- 8
        }
        else if (shape=="squares"){
            points <- alternating_squares_pointcloud_2D(total_points = num_points, num_squares = num_rings, square_width = ring_width, spacing = spacing, noise = noise_in_data)
            true_area <- square_area_2D(num_squares = num_rings, square_width = ring_width, spacing = spacing)
            true_shape <- list(polygons = alternating_squares_polygon(num_squares = num_rings, square_width = ring_width, spacing = spacing))
        }
        
        else {
            stop("Unknown shape type.")
        }

        print("Computing alpha shape...")

  alpha_shape_result <- alpha_shape_2D(points, alpha, CL, sampling_fraction = sampling_fraction)

  # build a filename safely using base R; "%+%" is not a base R operator and
  # will error if the package that defines it isn't loaded
  save_path <- paste0("alpha_shape_2D_alpha_", alpha, "_CL_", CL, "_noise_", noise_in_data, "_shape_", shape, ".png")

  visualize_alpha_shape_2D(alpha_shape_result, points, title, r_outer = r_outer, r_inner = r_inner, shape = shape, spacing = spacing, ring_width = ring_width, num_rings = num_rings, save_path = save_path)

        #estimated_area <- calculate_area_volume(alpha_shape_result, dim = 2)
        #error <- error_in_2D(true_area, estimated_area)
        #print(paste("Error in 2D area estimation:", error))
        #print(paste("Estimated Area:", estimated_area, "True Area:", true_area))

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

alpha_variations <- function(
                            alpha_value_list = c(0.1, 0.2), 
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
                            save_path = NULL
                            ) {
        
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
    } else{
    if (shape == "concentric") {
        temp <- paste0(str(spacing), "_", str(ring_width), "_", str(num_rings))
        save_path <- paste0("alpha_shape_KL_IOU_concentric_rings_", temp, "_results.csv")
    } else  if (shape == "ring") {
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
  num_points_list = c(1e1, 1e2, 1e3, 1e4, 1e5, 1e6)
) {
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
        plot(data$num_points, data$time_taken_seconds, type = "b",
             xlab = "Number of Points", ylab = "Time Taken (seconds)",
             main = "Alpha Shape Computation Time vs Number of Points")
        dev.off()
    }
    else if (scale == 2) {
        save_path <- "alpha_shape_timing_linear_log.png"
        png(save_path)
        plot(data$num_points, data$time_taken_seconds, log = "y", type = "b",
             xlab = "Number of Points", ylab = "Time Taken (seconds, log scale)",
             main = "Alpha Shape Computation Time vs Number of Points")
        dev.off()
    }
    else if (scale == 3) {
        save_path <- "alpha_shape_timing_log_log.png"
        png(save_path)
        plot(data$num_points, data$time_taken_seconds, log = "xy", type = "b",
             xlab = "Number of Points (log scale)", ylab = "Time Taken (seconds, log scale)",
             main = "Alpha Shape Computation Time vs Number of Points")
        dev.off()
    }
    else if (scale == 4) {
        save_path <- "alpha_shape_timing_n_vs_sqrt_t.png"
        png(save_path)
        plot(data$num_points, sqrt(data$time_taken_seconds), type = "b",
             xlab = "Number of Points", ylab = "Square Root of Time Taken (seconds)",
             #main = "Alpha Shape Computation Time vs Number of Points (n vs sqrt t scale)"
             )
        
        dev.off()
    }
    else if (scale == 5) { # log n vs log sqrt t
        save_path <- "alpha_shape_timing_log_n_vs_log_sqrt_t.png"
        png(save_path)
        plot(log(data$num_points), log(sqrt(data$time_taken_seconds)), type = "b",
             xlab = "Log(Number of Points)", ylab = "Log(Square Root of Time Taken (seconds))",
             main = "Alpha Shape Computation Time vs Number of Points (log-log sqrt scale)")
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
    ring_width = 1.5
) {
    data_results <- data.frame()
    for (fraction in fraction_list) {
        result <- main_test(dim = dim, alpha = alpha, CL = CL, num_points = num_points, noise_in_data = noise_in_data, shape = shape, spacing=spacing, ring_width = ring_width, num_rings = num_rings, r_outer = r_outer, r_inner = r_inner, sampling_fraction = fraction, title = paste("Alpha Shape Visualization - Sampling Fraction:", fraction))
        print(paste("Completed for Sampling Fraction:", fraction))
        # save to csv
        data_results <- rbind(data_results, data.frame(sampling_fraction = fraction, KL_divergence = result$KL_divergence, IOU_score = result$IOU_score))}
    
    write.csv(data_results, file = "alpha_shape_sampling_fraction_results.csv", row.names = FALSE)
}

corner_variations <- function(s = 3.0, w= 1.5, num_rings = 3, alpha_list = c(1.0), CL_list = c(0.95), num_points = 10000, noise_in_data_list = c(0.01)) {
   
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

#set.seed(137)


#alpha_sample_list <- seq(0.1, 2.0, by = 0.2)
alpha_sample_list <- c(0.5, 1.0)
noise_sample_list <- c(0.01, 0.001)

specific_parameter_list <- list(
  # alpha, CL, noise, shape

    # squares
    #c(1.0, 1.0, 0, "squares"),
    #c(1.0, 1.0, 0.01, "squares"),
    #c(1.0, 0.95, 0, "squares"),
    #c(1.0, 0.95, 0.01, "squares"),

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
alpha_variations(alpha_value_list = alpha_sample_list, 
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

corner_analysis_list <- list(
    # alpha CL noise shape
    c(1.0, 1.0, 0.0, "squares"),
    c(1.0, 1.0, 0.01, "squares"),
    c(1.0, 0.95, 0.0, "squares"),
    c(1.0, 0.95, 0.01, "squares"),
    c(1.0, 1.0, 0.0, "concentric"),
    c(1.0, 1.0, 0.01, "concentric"),
    c(1.0, 0.95, 0.0, "concentric"),
    c(1.0, 0.95, 0.01, "concentric")
)

print("Starting corner variations test...")

alpha_variations(alpha_value_list = alpha_sample_list, 
                dim = 2, 
                CL_list = c(1.0, 0.95), 
                num_points = 1000, 
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

print("All Alpha variations test completed.")
print("Starting complexity time measurements...")
#datapoints_variations(num_points_list = round(10^seq(1, 4, length.out = 10)))

plot_datapoints_timing(csv_file = "alpha_shape_datapoints_timing.csv", scale = 4)

print("Complexity time measurements completed.")

print("Starting sample fraction variations test...")
#sample_fraction_variations(fraction_list = c(0.5, 0.6, 0.7, 0.8, 0.85,  0.9, 0.95, 0.99), dim = 2, alpha = 1.0, CL = 0.95, num_points = 2000, noise_in_data = 0.01, shape = "squares", r_outer = 5, r_inner = 3, num_rings = num_rings, spacing = spacing, ring_width = ring_width)

print("Sample fraction variations test completed.")

print("All tests completed.")

# nolint end

# TO DO LIST:

# 1.) determine if data is correlated with itself
# 2.) determine if multi-testing happens
# 3.) power analysis for DBscan