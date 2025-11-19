import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay
import alphashape
from descartes import PolygonPatch
import matplotlib.patches as mpatches
from shapely.geometry import MultiPoint, Polygon, MultiPolygon
import random
from scipy import ndimage, spatial
from shapely.geometry import Polygon, LineString, Point, MultiPoint
from shapely.ops import unary_union
import pyvisgraph as vg
from sklearn.cluster import DBSCAN
import colormaps


# This script generates a plot for a pitchfork bifurcation
# dx = r*x - x**3

# It color codes the stable and unstable equilibria
# Stable equilibria: solid lines
# Unstable equilibria: dashed lines

# It also shades the regions of increasing behavior and decreasing behavior
# Increasing behavior: light green
# Decreasing behavior: light red


def pitchfork_plot():

    r_values = np.linspace(-2, 2, 400)
    x_values = np.linspace(-2, 2, 400)

    X, R = np.meshgrid(x_values, r_values)
    DX = R * X - X**3

    plt.figure(figsize=(10, 6))
    # Plot stable equilibria
    plt.plot(
        r_values[r_values < 0],
        np.zeros_like(r_values[r_values < 0]),
        "b-",
        label="Stable Equilibria",
    )
    plt.plot(
        r_values[r_values > 0],
        np.sqrt(r_values[r_values > 0]),
        "b-",
        label="Stable Equilibria",
    )
    plt.plot(r_values[r_values > 0], -np.sqrt(r_values[r_values > 0]), "b-")
    # Plot unstable equilibria
    plt.plot(r_values, np.zeros_like(r_values), "r--", label="Unstable Equilibria")
    # Shade regions of increasing behavior
    plt.fill_between(
        r_values,
        -2,
        2,
        where=(R > X**2),
        color="lightgreen",
        alpha=0.5,
        label="Increasing Behavior",
    )
    # Shade regions of decreasing behavior
    plt.fill_between(
        r_values,
        -2,
        2,
        where=(R < X**2),
        color="lightcoral",
        alpha=0.5,
        label="Decreasing Behavior",
    )
    plt.title("Pitchfork Bifurcation Diagram")
    plt.xlabel("Parameter r")
    plt.ylabel("Equilibrium x")
    plt.ylim(-2, 2)
    plt.legend()
    plt.grid()
    plt.show()


# make a ODE based on the concentric ring idea and plot it

# dr/dt = PI_i=1^n (r - i*s - (i-1)*w)(r- i*s -i*w)


def concentric_ring_dynamics(n=3, s=3, w=1.5):
    def dr_dt(r, n, s, w):
        prod = 1
        for i in range(1, n + 1):
            prod *= (r - i * s - (i - 1) * w) * (r - i * s - i * w)
        return -prod

    def max_r(n, s, w):
        return (n + 1) * s + n * w

    r_values = np.linspace(0, max_r(n, s, w), 400)

    dr_values = dr_dt(r_values, n=n, s=s, w=w)

    dr_values /= 1000  # normalize for plotting

    # plot a polar plot of dr/dt vs r

    # if dr > 0, color green
    # if dr < 0, color red

    plt.figure(figsize=(10, 6))
    plt.plot(r_values, dr_values, "k-", label="dr/dt")
    # fill region from top of plot to bottom of plot with the appropriate colors
    # i.e. fill from y_min to y_max
    plt.fill_between(
        r_values,
        -2,
        2,
        where=(dr_values > 0),
        color="lightgreen",
        alpha=0.5,
        label="Increasing r",
    )
    plt.fill_between(
        r_values,
        -2,
        2,
        where=(dr_values < 0),
        color="lightcoral",
        alpha=0.5,
        label="Decreasing r",
    )
    plt.axhline(0, color="black", lw=1.5, ls="--")
    for i in range(1, n + 1):
        r1 = i * s + (i - 1) * w
        r2 = i * s + i * w
        plt.axvline(r1, color="blue", lw=1.5, ls="--")
        plt.axvline(r2, color="blue", lw=1.5, ls="--")
    plt.title("Concentric Ring Dynamics")
    plt.ylim(-2, 2)
    plt.xlabel("Radius r")
    plt.ylabel("dr/dt")
    plt.legend()
    plt.grid()
    plt.savefig("concentric_ring_dynamics.png")

    # plot a polar plot of the same data
    plt.figure(figsize=(8, 8))
    theta = np.linspace(0, 2 * np.pi, 400)
    R, Theta = np.meshgrid(r_values, theta)
    DR = dr_dt(R, n=n, s=s, w=w)
    plt.polar(Theta, R, color="k", alpha=0.1)
    # fill region from center to edge with the appropriate colors
    plt.contourf(
        Theta,
        R,
        DR,
        levels=[-np.inf, 0, np.inf],
        colors=["lightcoral", "lightgreen"],
        alpha=0.5,
    )
    plt.title("Concentric Ring Dynamics (Polar Plot)")
    plt.savefig("concentric_ring_dynamics_polar.png")


def union_of_balls_figure(points=None):
    # create a sample of a union of balls plot
    if points is None:
        points = np.random.rand(15, 2) * 10

    fig, ax = plt.subplots()
    for point in points:
        # use facecolor + edgecolor to avoid Matplotlib warning about 'color' overriding edgecolor
        circle = plt.Circle(
            point, radius=1.0, facecolor="cyan", alpha=0.5, edgecolor="blue"
        )
        ax.add_artist(circle)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt.title("Union of Balls Plot Example")
    plt.savefig("union_of_balls_plot.png")


def voronoi_complex_figure(points=None):
    # create a sample of a voronoi complex plot
    if points is None:
        points = np.random.rand(15, 2) * 10
    vor = Voronoi(points)
    fig = voronoi_plot_2d(
        vor, show_vertices=False, line_colors="orange", line_width=2, point_size=50
    )
    plt.title("Voronoi Complex Plot Example")
    plt.savefig("voronoi_complex_plot.png")


def deluanay_triangulation_figure(points=None):
    # create a sample of a deluanay triangulation plot
    if points is None:
        points = np.random.rand(15, 2) * 10

    tri = Delaunay(points)
    plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy(), color="purple")
    plt.plot(points[:, 0], points[:, 1], "o", color="red")
    plt.title("Delaunay Triangulation Plot Example")
    plt.savefig("delaunay_triangulation_plot.png")


def voronoi_nerve_figure(points=None):
    # create a sample of a voronoi nerve plot
    if points is None:
        points = np.random.rand(15, 2) * 10

    vor = Voronoi(points)
    fig, ax = plt.subplots()

    # draw nerve edges between points whose Voronoi cells share an edge
    for ridge in vor.ridge_points:
        p1 = points[ridge[0]]
        p2 = points[ridge[1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "g--", lw=1.5)

    plt.title("Voronoi Nerve Plot Example")
    plt.savefig("voronoi_nerve_plot.png")


def alpha_shape_figure(points=None, alpha=None):
    # create a sample of an alpha shape plot
    if points is None:
        points = np.random.rand(15, 2) * 10
    # Build alpha shape from points (use create_alpha_shape to handle fallbacks)
    alpha_shape = create_alpha_shape(points, alpha=alpha)

    fig, ax = plt.subplots()

    # Plot shapely Polygon or MultiPolygon using matplotlib patches (avoids descartes/shapely incompatibilities)
    def _plot_polygon(ax, poly, **kwargs):
        coords = np.asarray(poly.exterior.coords)
        patch = mpatches.Polygon(coords, closed=True, **kwargs)
        ax.add_patch(patch)

    if isinstance(alpha_shape, Polygon):
        _plot_polygon(
            ax, alpha_shape, edgecolor="green", facecolor="lightblue", alpha=0.5
        )
    elif isinstance(alpha_shape, MultiPolygon):
        for p in alpha_shape.geoms:
            _plot_polygon(ax, p, edgecolor="green", facecolor="lightblue", alpha=0.5)
    else:
        # as a last resort, try PolygonPatch (older environments)
        try:
            patch = PolygonPatch(
                alpha_shape, edgecolor="green", facecolor="lightblue", alpha=0.5
            )
            ax.add_patch(patch)
        except Exception:
            print(
                "Warning: couldn't plot alpha_shape with PolygonPatch or matplotlib patches."
            )
    ax.scatter(points[:, 0], points[:, 1], color="black", zorder=3)
    plt.title("Alpha Shape Plot Example")
    plt.savefig("alpha_shape_plot.png")


def create_alpha_shape(
    points, alpha=None, try_optimize=True, fallback_convex_hull=True
):
    """Create an alpha shape (shapely Polygon/MultiPolygon) from a numpy array or list of points.

    - points: (N,2) array-like
    - alpha: numeric or None. If None and try_optimize is True, the function will try alphashape.optimizealpha.
    - try_optimize: if True and alpha is None, attempt to find an optimal alpha
    - fallback_convex_hull: if alphashape doesn't produce a Polygon, return the convex hull as last resort

    Returns a shapely.geometry (Polygon or MultiPolygon).
    Also saves WKT to 'alpha_shape_from_points.wkt'.
    """
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an (N,2) array-like")

    geom = None
    # If user provided an alpha, try it first
    if alpha is not None:
        geom = alphashape.alphashape(pts, alpha=alpha)

    # If geom invalid or no alpha provided, optionally optimize
    if geom is None or not isinstance(geom, (Polygon, MultiPolygon)):
        if alpha is None and try_optimize:
            try:
                opt_alpha = alphashape.optimizealpha(pts)
                geom = alphashape.alphashape(pts, alpha=opt_alpha)
                print(f"optimized alpha = {opt_alpha}")
            except Exception as e:
                # optimization can fail on degenerate inputs
                print(f"optimizealpha failed: {e}")

    # Final fallback to convex hull
    if not isinstance(geom, (Polygon, MultiPolygon)):
        if fallback_convex_hull:
            geom = MultiPoint(list(map(tuple, pts))).convex_hull
            print("Falling back to convex hull for alpha shape (not a polygon).")
        else:
            raise RuntimeError("Could not create polygonal alpha shape from points")

    # Save WKT for inspection
    try:
        with open("alpha_shape_from_points.wkt", "w", encoding="utf-8") as f:
            f.write(geom.wkt)
    except Exception as e:
        print(f"Warning: failed to save WKT: {e}")

    return geom


def plot_raw_data(points):
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], color="black")
    plt.title("Raw Data Points")
    plt.savefig("raw_data_points.png")


def convex_hull_figure(points=None):
    # create a sample of a convex hull plot
    if points is None:
        points = np.random.rand(15, 2) * 10

    hull = MultiPoint(list(map(tuple, points))).convex_hull

    fig, ax = plt.subplots()

    # draw convex hull
    if isinstance(hull, Polygon):
        coords = np.asarray(hull.exterior.coords)
        ax.plot(coords[:, 0], coords[:, 1], "r-", lw=2)

    ax.scatter(points[:, 0], points[:, 1], color="black", zorder=3)
    plt.title("Convex Hull Plot Example")
    plt.savefig("convex_hull_plot.png")


def relative_convex_hull_figure(points=None):
    # create a sample of a relative convex hull plot
    if points is None:
        points = np.random.rand(15, 2) * 10

    hull = MultiPoint(list(map(tuple, points))).convex_hull

    fig, ax = plt.subplots()

    # draw convex hull
    if isinstance(hull, Polygon):
        coords = np.asarray(hull.exterior.coords)
        ax.plot(coords[:, 0], coords[:, 1], "r-", lw=2)

    ax.scatter(points[:, 0], points[:, 1], color="black", zorder=3)
    plt.title("Relative Convex Hull Plot Example")
    plt.savefig("relative_convex_hull_plot.png")


def gridCellCentres(xMin, xMax, yMin, yMax, cellSize):
    """
    Create a grid of coordinates for cell centres systematically spaced
    throughout the specified spatial extent.

    Parameters
    ----------
    xMin: number
        the x coordinate minimum of the grid cell centres
    xMax: nmber
        the x coordinate maximum of the grid cell centres
    yMin: number
        the y coordinate minimum of the grid cell centres
    yMax: number
        the y coordinate maximum of the grid cell centres
    cellSize: number
        the size of the cells and hence the spacing betweem cell centres

    Returns
    -------
    out : 2D array
        Float value coordinates for the centre of each cell.
    """
    # Create an array of all cell centres
    nCol = int(xMax / cellSize) + 1
    nRow = int(yMax / cellSize) + 1
    xSpace = np.linspace(xMin, xMax, nCol)
    ySpace = np.linspace(yMax, yMin, nRow)
    gridSpace = np.meshgrid(xSpace, ySpace)
    gridSpaceX = gridSpace[0].flatten()
    gridSpaceY = gridSpace[1].flatten()
    return (np.array([gridSpaceX, gridSpaceY]).T, gridSpace[0].shape)


def vdCircumsRadii(p, dt=None, vd=None):
    """
    Calculate circumcircle circumradii and circumcentres of a Delaunay
    triangulation via a Voronoi diagram.

    Parameters
    ----------
    p: array
        2D array of point locations
    dt: Delaunay triangulation
        A Delaunay triangulation of the point locations
    vd: Voronoi diagram
        A Voronoi diagram of the point locations

    Returns
    -------
    out : tuple of arrays
        The arrays contain the circumradii and circumcentres.
    """
    if dt == None:
        dt = spatial.Delaunay(p)
    if vd == None:
        vd = spatial.Voronoi(p)
    circumcentres = vd.vertices
    simplexVertex0 = p[dt.simplices[:, 0], :]
    circumradii = np.sqrt(np.sum((circumcentres - simplexVertex0) ** 2, axis=1))
    return (circumradii, circumcentres)


def fuzzyAlphaShape(points, xMin, xMax, yMin, yMax, cellSize, alpha, stdv):
    """
    Calculate the generalised fuzzy alpha shape possibiity values for grid cells centres
    systematically spaced throughout the specified spatial extent.

    Parameters
    ----------
    points: array
        2D array of point locations
        xMin: number
        the x coordinate minimum of the grid cell centres
    xMax: nmber
        the x coordinate maximum of the grid cell centres
    yMin: number
        the y coordinate minimum of the grid cell centres
    yMax: number
        the y coordinate maximum of the grid cell centres
    cellSize: number
        the size of the cells and hence the spacing betweem cell centres
    alpha: number
        the alpha threshold for determining set membership
    stdv: number
        the standard deviation around the estimate of the alpha value that is used to
        define set membership and also the degree of generalisation

    Returns
    -------
    out : 2D array
        Float values for fuzzy membership for the centre of each cell.
    """
    # Calculate a Delaunay triangulation
    delaunayTriangulation = spatial.Delaunay(points)
    # Determine set membership possibiity for each simplex based on circumradii
    circumradii = vdCircumsRadii(points, delaunayTriangulation)[0]
    possibility = np.exp(-0.5 * ((circumradii - alpha) / stdv) ** 2)
    possibility[circumradii <= alpha] = 1
    possibility = np.insert(
        possibility, 0, 0
    )  # Insert 0 for cells outside Delaunay triangulation
    # Identify the simplicies that each grid point belongs to
    cellCentres, gridShape = gridCellCentres(xMin, xMax, yMin, yMax, cellSize)
    gridSpaceSimplex = delaunayTriangulation.find_simplex(cellCentres) + 1
    # Index out possibility values
    alphaShapePossibility = possibility[gridSpaceSimplex]
    fuzzAlphaShape = np.reshape(alphaShapePossibility, gridShape)
    return fuzzAlphaShape


def generalisedFuzzyAlphaShape(points, xMin, xMax, yMin, yMax, cellSize, alpha, stdv):
    """
    Calculate the generalised fuzzy alpha shape possibiity values for grid cells centres
    systematically spaced throughout the specified spatial extent.

    Parameters
    ----------
    points: array
        2D array of point locations
        xMin: number
        the x coordinate minimum of the grid cell centres
    xMax: nmber
        the x coordinate maximum of the grid cell centres
    yMin: number
        the y coordinate minimum of the grid cell centres
    yMax: number
        the y coordinate maximum of the grid cell centres
    cellSize: number
        the size of the cells and hence the spacing betweem cell centres
    alpha: number
        the alpha threshold for determining set membership
    stdv: number
        the standard deviation around the estimate of the alpha value that is used to
        define set membership and also the degree of generalisation

    Returns
    -------
    out : 2D array
        Float values for fuzzy membership for the centre of each cell.
    """
    # Calculate a fuzzy alpha shape
    fuzzAlphaShape = fuzzyAlphaShape(
        points, xMin, xMax, yMin, yMax, cellSize, alpha, stdv
    )
    # Generalise the fuzzy alpha shape using a Gaussian moving window low pass smoothing filter
    genFuzzAlphaShape = ndimage.gaussian_filter(
        input=fuzzAlphaShape, sigma=stdv / cellSize, mode="constant", cval=0
    )
    return genFuzzAlphaShape


def plot_fuzzy_alpha_shape(
    points, xMin=0, xMax=10, yMin=0, yMax=10, cellSize=0.1, alpha=0.2, stdv=0.1
):
    fuzzAlphaShape = fuzzyAlphaShape(
        points, xMin, xMax, yMin, yMax, cellSize, alpha, stdv
    )
    plt.imshow(
        fuzzAlphaShape, extent=(xMin, xMax, yMin, yMax), origin="lower", cmap="viridis"
    )
    plt.colorbar(label="Fuzzy Membership Value")
    plt.title("Fuzzy Alpha Shape")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.savefig("fuzzy_alpha_shape.png")


def _ensure_ccw_coords(coords):
    """Return coords with CCW orientation (pyvisgraph accepts either, but consistent orientation helps)."""
    poly = Polygon(coords)
    if not poly.exterior.is_ccw:
        return list(poly.exterior.coords)[::-1][:-1]
    return list(poly.exterior.coords)[:-1]


def build_visgraph_from_polygon(polygon: Polygon):
    """Build and return a pyvisgraph.VisGraph for a single simple polygon (the container B)."""
    coords = list(polygon.exterior.coords)[:-1]
    # pyvisgraph expects a list of polygons, each polygon is list of vg.Point
    poly_pts = [vg.Point(x, y) for (x, y) in _ensure_ccw_coords(coords)]
    G = vg.VisGraph()
    G.build([poly_pts])  # single obstacle which is the polygon boundary
    return G


def relative_convex_hull(points, container=None):
    """
    Compute the relative convex hull of a set of points inside a container polygon.

    Parameters
    ----------
    points : (N,2) array-like
        Input points (must lie within container polygon).
    container : shapely.geometry.Polygon, optional
        Simple polygon defining the bounding region. If None, the minimal bounding box is used.

    Returns
    -------
    shapely.geometry.Polygon
        The relative convex hull polygon.
    """

    if container is None:
        # Use minimal bounding box as container
        minx, miny = np.min(points, axis=0) - 1e-3
        maxx, maxy = np.max(points, axis=0) + 1e-3
        container = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])

    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be an (N,2) array")

    # Convex hull of the input points
    hull = MultiPoint([tuple(p) for p in pts]).convex_hull

    # Clip convex hull with the container polygon
    rch = hull.intersection(container)

    # Ensure valid polygonal output
    if rch.is_empty:
        raise RuntimeError(
            "Relative convex hull is empty — points may lie outside container."
        )
    if rch.geom_type == "MultiPolygon":
        # Union in case of fragmentation due to intersection
        rch = unary_union(rch)

    # Return the outer boundary polygon
    if rch.geom_type == "Polygon":
        return rch
    else:
        raise RuntimeError("Failed to compute a valid relative convex hull geometry.")


def plot_relative_convex_hull(points, container=None):
    rel_hull = relative_convex_hull(points, container=container)

    fig, ax = plt.subplots()

    # draw relative convex hull
    if isinstance(rel_hull, Polygon):
        coords = np.asarray(rel_hull.exterior.coords)
        ax.plot(coords[:, 0], coords[:, 1], "m-", lw=2)

    ax.scatter(points[:, 0], points[:, 1], color="black", zorder=3)
    if container is not None:
        cont_coords = np.asarray(container.exterior.coords)
        ax.plot(cont_coords[:, 0], cont_coords[:, 1], "k--", lw=1)

    plt.title("Relative Convex Hull Plot Example")
    plt.savefig("relative_convex_hull_plot.png")


def IOU_plot():
    # create a plot showing intersection over union for two overlapping circles
    fig, ax = plt.subplots()

    circle1 = plt.Circle(
        (4, 5), radius=3.0, facecolor="cyan", alpha=0.5, edgecolor="blue"
    )
    circle2 = plt.Circle(
        (6, 5), radius=3.0, facecolor="orange", alpha=0.5, edgecolor="red"
    )

    ax.add_artist(circle1)
    ax.add_artist(circle2)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    plt.title("Intersection over Union Example")
    plt.savefig("iou_plot.png")


def DBScan_plot(points):
    if points is None:
        points = np.random.rand(15, 2) * 10

    clustering = DBSCAN(eps=1.5, min_samples=2).fit(points)
    labels = clustering.labels_
    plt.figure()
    unique_labels = set(labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))
    for k in unique_labels:
        class_member_mask = labels == k
        xy = points[class_member_mask]
        plt.scatter(
            xy[:, 0],
            xy[:, 1],
            color=colors(k),
            label=f"Cluster {k}" if k != -1 else "Noise",
        )
    # create small ellipses for each cluster with foci of the ellipse equal to points in the cluster
    # minor axis = to quarter the distance between the foci
    # major axis = to half the distance between the foci
    # the angle of the ellipse is related to the angle between the foci

    for k in unique_labels:
        if k == -1:
            continue  # skip noise
        class_member_mask = labels == k
        xy = points[class_member_mask]
        if len(xy) < 2:
            continue
        center = np.mean(xy, axis=0)
        dists = np.linalg.norm(xy - center, axis=1)
        if len(dists) == 0:
            continue
        max_dist = 4 * np.max(dists)
        angle = np.arctan2(xy[1, 1] - xy[0, 1], xy[1, 0] - xy[0, 0]) * 180 / np.pi
        ellipse = mpatches.Ellipse(
            center,
            width=max_dist,
            height=max_dist / 2,
            angle=angle,
            edgecolor=colors(k),
            facecolor=colors(k),
            lw=2,
            ls="--",
            alpha=0.5,
        )
        plt.gca().add_patch(ellipse)

    plt.title("DBSCAN Clustering Example")
    plt.legend()
    plt.savefig("dbscan_clustering_plot.png")


def hollow_square_plot(w=1.5, s=3, num=1):
    
    plt.close("all")
    plt.figure()
    # color in everything red and then overwrite with green if in the squares

    og_distance = 0

    x = np.linspace(-num*(s+w) -1, (num)*(s+w) +1, 400)
    y = np.linspace(-num*(s+w) -1, (num)*(s+w) +1, 400)
    X, Y = np.meshgrid(x, y)

    img = np.zeros((X.shape[0], X.shape[1], 4))  # RGBA


    for k in range(0,2*num+1):
        if k % 2 == 1:
            color = np.array([0, 1, 0, 0.5])   # green RGBA
            distance = og_distance + w
        else:
            color = np.array([1, 0, 0, 0.5])   # red RGBA
            distance = og_distance + s

        M = (np.maximum(np.abs(X), np.abs(Y)) > og_distance) & (np.maximum(np.abs(X), np.abs(Y)) <= distance)

        img[M] = color

        og_distance = distance

    plt.imshow(img, extent=(x.min(), x.max(), y.min(), y.max()), origin="lower")

    plt.gca().set_aspect("equal", adjustable="box")

    plt.title("Hollow Square Dynamics Example")
    plt.savefig("hollow_square_dynamics_plot.png")


if __name__ == "__main__":

    data_examples = False
    dynamics_examples = True

    random.seed(137)
    np.random.seed(137)
    points = np.random.rand(15, 2) * 10

    # pitchfork_plot()
    if dynamics_examples:
        concentric_ring_dynamics()
        hollow_square_plot(num=3)
    if data_examples:
        plot_raw_data(points)
        voronoi_complex_figure(points)
        union_of_balls_figure(points)
        voronoi_nerve_figure(points)
        deluanay_triangulation_figure(points)
        alpha_shape_figure(points, alpha=0.2)
        plot_fuzzy_alpha_shape(points)
        convex_hull_figure(points)
        plot_relative_convex_hull(points)
    IOU_plot()
    DBScan_plot(points)
