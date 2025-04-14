import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np


def plot_polygon_with_lines(
    v,
    lines=(),
    ax=None,
    label_origin=True,
    origin_color="grey",
    v0_color="black",
    marker_color="black",
    additional_points=None,
):
    if ax is None:
        _, ax = plt.subplots()

    # Draw polygon
    polygon = patches.Polygon(v, closed=True, fill=False, edgecolor="black")
    ax.add_patch(polygon)

    # Draw lines
    for (p0, p1), color, linestyle, has_arrow, lab, text_offset in lines:
        # Draw main line
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linestyle=linestyle, zorder=0)

        # Draw arrowhead if needed
        if has_arrow:
            p0, p1 = np.array(p0), np.array(p1)
            direction = p1 - p0
            norm = np.linalg.norm(direction)
            if norm == 0:
                continue  # avoid division by zero
            direction = direction / norm
            arrow_length = 0.1 * norm  # scale arrowhead size

            # Define common arrow parameters
            start_x = p1[0] - direction[0] * arrow_length
            start_y = p1[1] - direction[1] * arrow_length
            dx = 3.5 * direction[0] * arrow_length
            dy = 3.5 * direction[1] * arrow_length

            # Add arrow patch
            arrowhead = patches.FancyArrow(
                start_x,
                start_y,
                dx,
                dy,
                # width=0.0052,
                width=0.0082,
                color=color,
            )
            ax.add_patch(arrowhead)

            # Add label at the midpoint of the arrow
            ax.text(
                start_x + dx / 2 + text_offset[0]
                if text_offset is not None
                else 0,  # + 0.07,
                start_y + dy / 4 + text_offset[1]
                if text_offset is not None
                else 0,  # - 0.04,
                lab,
                fontsize=10,
                ha="left",
                va="center",
                color=color,
            )

    # Point deep within the difference
    ax.scatter(0, 0, c=v0_color)
    ax.text(0.075, -0.15, r"$V_0$", fontsize=10, color=v0_color)

    # ORIGIN (unknown to us)

    ax.scatter(*ORIGIN, c=origin_color, zorder=10, s=20 if label_origin else 15)
    if label_origin:
        ax.text(ORIGIN[0] + 0.05, ORIGIN[1] - 0.10, r"$O$", fontsize=10, c=origin_color)

    if additional_points is not None:
        for i, pt in enumerate(additional_points):
            # ax.scatter(*pt, color="k", s=15)
            ax.scatter(*pt, color=marker_color)
            ax.text(pt[0] * 1.05, pt[1] * 1.20, rf"$V_{i + 1}$", fontsize=10, color=marker_color)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(np.array(ax.get_xlim())*1.05)
    ax.set_ylim(np.array(ax.get_ylim())*1.05)
    ax.set_aspect("equal")

# TODO: calculate the real vectors as in rs code

def perp(vec2):
    return np.array([-vec2[1], vec2[0]])

if __name__ == "__main__":
    from coxeter.families.common import _make_ngon

    fig, ax = plt.subplots(2, 6, figsize=(12, 4), sharex=True, sharey=True)
    poly = _make_ngon(7, angle=np.pi / 7)[:, :2]

    ORIGIN = np.array([-0.15, 0.48])
    v0 = np.zeros(2) # Not generally true, but ok for now

    ax[0, 0].set_ylabel("Portal Discovery")
    plot_polygon_with_lines(
        poly,
        ax=ax[0, 0],
        origin_color="#72618D",
    )

    # Support point along the vector v_0 O
    v1 = poly[2]
    if np.dot(v1, v0) > 0.0:
        print("Shapes do not overlap!")
    
    
    plot_polygon_with_lines(
        poly,
        ax=ax[0, 1],
        origin_color="#72618D",
        lines=[(((0, 0), ORIGIN), "black", "solid", True, r"", None)],
        additional_points=[v1]
    )
    
    plot_polygon_with_lines(
        poly,
        ax=ax[0, 2],
        origin_color="grey",
        label_origin=False,
        lines=[(((0, 0), ORIGIN * 0.5), "black", "solid", True, r"$n$", [0.07, -0.04])],
        additional_points=[v1],
    )
    
    # Choose the ray that points toward the origin: TODO: is this wrong?
    v_perp_v1v0 = perp(v1-v0)
    if v1.dot(v_perp_v1v0) > 0.0: # TODO: this is greater than in the primary code!
        v_perp_v1v0 *= -1 # Is this wrong?
    
    x0 = poly[2] / 2
    v2 = poly[0]
    plot_polygon_with_lines(
        poly,
        ax=ax[0, 3],
        origin_color="grey",
        label_origin=False,
        # lines=[(((0, 0), ORIGIN*0.5), "black", "solid", True, r"$n$")],
        lines=[
            (((0, 0), v1), "black", "dotted", False, "", None),
            (
                (x0, (x0+v_perp_v1v0/2)),
                "k",
                "solid",
                True,
                r"$n$",
                [-0.10, -0.12],
            ),
        ],
        additional_points=[v1, v2],
    )

    # Find the second support point
    plot_polygon_with_lines(
        poly,
        ax=ax[0, 4],
        origin_color="grey",
        label_origin=False,
        lines=[
            (((0, 0), v1), "black", "dotted", False, "", None),
            (((0, 0), v2), "black", "dotted", False, "", None),
        ],
        additional_points=[v1, v2],
    )
    plot_polygon_with_lines(
        poly,
        ax=ax[0, 5],
        origin_color="#72618D",
        v0_color="grey",
        marker_color="grey",
        label_origin=False,
        lines=[
            (((0, 0), v1), "grey", "dotted", False, "", None),
            (((0, 0), v2), "grey", "dotted", False, "", None),
            ((v1, v2), "black", "dotted", False, "", None), # p0
        ],
        additional_points=[v1, v2],
    )

    p0 = v2-v1
    v_perp_v2v1 = perp(p0)
    if (v1-v0).dot(v_perp_v2v1) < 0.0:
        v_perp_v2v1 = -v_perp_v2v1
    
    plot_polygon_with_lines(
        poly,
        ax=ax[1, 0],
        origin_color="#72618D",
        v0_color="grey",
        marker_color="grey",
        label_origin=False,
        lines=[
            (((0, 0), v1), "grey", "dotted", False, "", None),
            (((0, 0), v2), "grey", "dotted", False, "", None),
            ((v1, v2), "black", "dotted", False, "", None), # p0
            (((v1+v2)/2, v_perp_v2v1/1.5), "black", "solid", True, "", None),
            ((v1+v_perp_v2v1/4.5, v2+v_perp_v2v1/4.5), "black", "dotted", False, "", None),
        ],
        additional_points=[v1, v2],
    )

    print(
        v1.dot(v_perp_v2v1) >= 0.0 # Point is inside the initial portal
    )    

    v3 = poly[1]
    plot_polygon_with_lines(
        poly,
        ax=ax[1, 1],
        origin_color="#72618D",
        label_origin=False,
        v0_color="grey",
        lines=[
            (((0, 0), v1), "grey", "dotted", False, "", None),
            (((0, 0), v2), "grey", "dotted", False, "", None),
            ((v1, v2), "black", "solid", False, "", None), # p0
            (((v1+v2)/2, v_perp_v2v1/1.5), "black", "solid", True, "", None),
        ],
        additional_points=[v1, v2, v3],
    )


    
    
    plt.show()
