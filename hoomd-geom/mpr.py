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
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, linestyle=linestyle)

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
    ax.text(0.05, 0.0, r"$V_0$", fontsize=10)

    # ORIGIN (unknown to us)

    ax.scatter(*ORIGIN, c=origin_color, zorder=10, s=20 if label_origin else 15)
    if label_origin:
        ax.text(ORIGIN[0] + 0.05, ORIGIN[1] - 0.10, r"$O$", fontsize=10, c=origin_color)

    if additional_points is not None:
        for i, pt in enumerate(additional_points):
            # ax.scatter(*pt, color="k", s=15)
            ax.scatter(*pt, color="k")
            ax.text(pt[0] * 1.05, pt[1] * 1.12, rf"$V_{i + 1}$", fontsize=10)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(np.array(ax.get_xlim())*1.05)
    ax.set_ylim(np.array(ax.get_ylim())*1.05)
    ax.set_aspect("equal")


if __name__ == "__main__":
    from coxeter.families.common import _make_ngon

    fig, ax = plt.subplots(2, 6, figsize=(12, 4), sharex=True, sharey=True)
    poly = _make_ngon(7, angle=np.pi / 7)[:, :2]

    ORIGIN = np.array([-0.2, 0.45])

    plot_polygon_with_lines(
        poly,
        ax=ax[0, 0],
        origin_color="#72618D",
    )
    plot_polygon_with_lines(
        poly,
        ax=ax[0, 1],
        origin_color="#72618D",
        lines=[(((0, 0), ORIGIN), "black", "solid", True, "", None)],
        # additional_points=[poly[2]]
    )
    plot_polygon_with_lines(
        poly,
        ax=ax[0, 2],
        origin_color="grey",
        label_origin=False,
        lines=[(((0, 0), ORIGIN * 0.5), "black", "solid", True, r"$n$", [0.07, -0.04])],
        additional_points=[poly[2]],
    )
    x0 = poly[2] / 2
    # x1 = (-x0[1]+x0[0], x0[0]+x0[1])
    x1 = np.asarray((-x0[1] + x0[0], x0[0] + x0[1]))
    x1 += poly[2][::-1] * [1, -1]
    plot_polygon_with_lines(
        poly,
        ax=ax[0, 3],
        origin_color="grey",
        label_origin=False,
        # lines=[(((0, 0), ORIGIN*0.5), "black", "solid", True, r"$n$")],
        lines=[
            (((0, 0), poly[2]), "black", "dotted", False, "", None),
            (
                # ((poly[2] / 2, [-poly[2][1] / 2, -poly[2][0] / 2])),
                (x0, x1),
                "k",
                "solid",
                True,
                r"$n$",
                [-0.10, -0.12],
            ),
        ],
        additional_points=[poly[2], poly[0]],
    )
    plt.show()
