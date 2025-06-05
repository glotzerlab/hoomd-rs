import numpy as np
import scipy
import rowan
import gsd.hoomd
import hoomd
import matplotlib.pyplot as plt
from scipy.linalg import inv


def get_hoomd_ellipses(a, b, c, d, pos, theta):
    """HOOMD only supports 3d ellipsoids, and does not allow for 0 elements:
    so we need to generate 3d ellipsoids with a nonzero `c` and translate in the plane
    """

    f = gsd.hoomd.Frame()
    f.particles.N = 2
    f.configuration.box = [200, 200, 200, 0, 0, 0]
    f.particles.position = [[0, 0, 0], [*pos, 0]]

    f.particles.orientation = [[1, 0, 0, 0], rowan.from_axis_angle([0, 0, 1], theta)]
    f.particles.type_shapes = [
        {"type": "Ellipsoid", "a": a, "b": b, "c": 0.1},
        {"type": "Ellipsoid", "a": c, "b": d, "c": 0.1},
    ]
    f.particles.hoomd_type = [{"a": a, "b": b, "c": 0.1}, {"a": c, "b": d, "c": 0.1}]
    f.particles.types = ["B", "A"]
    f.particles.typeid = [1, 0]
    f.validate()
    with gsd.hoomd.open("test.gsd", "w") as test:
        test.append(f)
    return f


def hoomd_ellipse_intersect(a, b, c, d, pos, theta):
    f = get_hoomd_ellipses(a, b, c, d, pos, theta)
    cpu = hoomd.device.CPU()
    simulation = hoomd.Simulation(device=cpu, seed=0)
    mc = hoomd.hpmc.integrate.Ellipsoid()
    mc.shape["B"] = f.particles.hoomd_type[0]
    mc.shape["A"] = f.particles.hoomd_type[1]
    simulation.operations.integrator = mc
    simulation.create_state_from_snapshot(f)
    simulation.run(0)
    overlaps = simulation.operations.integrator.overlaps
    return bool(overlaps)


def rotmat_from_axis_angle(theta):
    return rowan.to_matrix(rowan.from_axis_angle([0, 0, 1], theta))


def make_ellipsoid_matrix(a, b, theta):
    m = np.diag([1 / a**2, 1 / b**2])
    if theta == 0:
        return m
    R = rotmat_from_axis_angle(theta)[:2, :2]
    return R @ m @ R.T


def fast_A(A, l):
    return np.diag(1 / (np.diag(A) * l) )
def invert_diagonal(A):
    return np.diag(1 / (np.diag(A)) )


def k_lambda(l, A, B, v):
    v = np.asarray(v)
    # CORE FUNCTION: this has to work
    # return 1 - v.T @ inv(1 / (1 - l) * inv(B) + 1 / l * inv(A)) @ v
    
    # One optimization
    return 1 - v.T @ inv(1 / (1 - l) * inv(B) + invert_diagonal(A) / l) @ v



def fast_k_lambda(l, A, B, v, r):
    # a, b: arrays of shape (2,) representing 1/semiaxes**2
    # v: array-like of shape (2,) representing the vector
    # r: 2x2 numpy array representing the rotation matrix
    v = np.asarray(v)
    B_inv = r @ invert_diagonal(B) @ r.T
    M_inverse = inv(1/(1-l) * B_inv + invert_diagonal(A)/l)

    return  1 -  v.T @ M_inverse @ v


def ellipsoid_intersects(a, b, c, d, theta, v, eps=1e-8):
    A = make_ellipsoid_matrix(c, d, theta=0)
    B = make_ellipsoid_matrix(a, b, theta=theta)

    # Ideally we solve this with a sturm sequence

    # So in theory, computing K(λ) is relatively expensive.
    # HOWEVER: For a single overlap check, we can reuse the intermediate v.T @ A^-1 @ v
    # and same for B, scaling by 1/λ and 1/(1-λ) each time.
    # res = scipy.optimize.minimize_scalar(
    #     k_lambda,
    #     bracket=[0.0+eps, 0.5, 1.0-eps],
    #     args=(A, B, v),
    # )
    try:
        res = scipy.optimize.minimize_scalar(
            fast_k_lambda,
            bracket=[0.0 + eps, 0.5, 1.0 - eps],
            args=(
                A,
                make_ellipsoid_matrix(a, b, theta=0),
                v,
                rotmat_from_axis_angle(theta)[:2, :2],
            ),
        )
    except ValueError:
        print(v)
        x = np.linspace(1e-6, 1 - 1e-6, 100)
        A = make_ellipsoid_matrix(a, b, 0)
        B = make_ellipsoid_matrix(c, d, 0)
        fig, ax = plt.subplots()
        ax.plot(x, [k_lambda(xi, A, B, v_ij) for xi in x])
        ax.hlines(0, 0, 1, "k", "dashed")
        plt.show()
    
    return res.fun >= 0 # min(F(λ)) >= 0 -> have an overlap


if __name__ == "__main__":
    # eps = 1e-6
    # for theta in [0.0, np.deg2rad(180.0), np.deg2rad(90), np.deg2rad(31.125)]:
    #     for a, b, c, d in [
    #         (1, 4, 1, 4),
    #         (2, 4, 0.5, 3),
    #         (2, 4, 1, 4),
    #         (2, 4, 1, 5.1),
    #         (2, 2, 3, 3),
    #         (1.234, 2.000001, 0.0005, 33.5),
    #     ]:
    #         print()
    #         print(a, b, c, d)

    #         for v_ij in np.array([[0.1, 0.0], [a + c - eps, 0], [a + c + eps, 0]]):
    #             np.testing.assert_equal(
    #                 ellipsoid_intersects(a, b, c, d, theta, v_ij),
    #                 hoomd_ellipse_intersect(a, b, c, d, v_ij, theta),
    #                 err_msg=f"FAILED: {theta, v_ij}",
    #             )
    #         # Very sheared ellipsoids are read incorrectly in HOOMD - must increase eps
    #         for v_ij in np.array([
    #             [0.0, 0.1],
    #             [0, b + d - eps * 2],
    #             [0, b + d + eps * 2],
    #         ]):
    #             np.testing.assert_equal(
    #                 ellipsoid_intersects(a, b, c, d, theta, v_ij),
    #                 hoomd_ellipse_intersect(a, b, c, d, v_ij, theta),
    #                 err_msg=f"FAILED: {theta, v_ij}",
    #             )

    #         v_ij = np.ones(2)
    #         for v_ij in [
    #             [0.1, 0.1],
    #             [0.1, 2.1],
    #             [1.0, 1.0],
    #             [4.0, 5.0],
    #             [2.0, 11],
    #             [0.1, 3.124],
    #             [3.124, 0.1],
    #         ]:
    #             try:
    #                 np.testing.assert_equal(
    #                     jen := ellipsoid_intersects(a, b, c, d, theta, v_ij),
    #                     hoomd_ellipse_intersect(a, b, c, d, v_ij, theta),
    #                     err_msg=f"FAILED: {jen}\n(a,b,c,d), theta, v_ij{(a, b, c, d), theta, v_ij}",
    #                 )
    #             except AssertionError as e:
    #                 x = np.linspace(eps, 1 - eps, 100)
    #                 A = make_ellipsoid_matrix(a, b, 0)
    #                 B = make_ellipsoid_matrix(c, d, theta)
    #                 fig, ax = plt.subplots()
    #                 ax.plot(x, [k_lambda(xi, A, B, v_ij) for xi in x])
    #                 ax.hlines(0, 0, 1, "k", "dashed")
    #                 plt.show()
    #                 raise AssertionError from e
    # print()
    # print()

    # np.random.seed(0)
    # for i in range(1_000):
    #     a, b, c, d = np.random.uniform(eps, 10, size=4)
    #     v_ij = np.random.uniform(-10, 10, size=2)
    #     theta = np.random.uniform(-2*np.pi, 2*np.pi)
    #     # theta = 90
    #     try:
    #         np.testing.assert_equal(
    #             jen := ellipsoid_intersects(a, b, c, d, theta, v_ij),
    #             hoomd_ellipse_intersect(a, b, c, d, v_ij, theta),
    #             err_msg=f"FAILED: {jen}\n(a,b,c,d), theta, v_ij{(a, b, c, d), theta, v_ij}",
    #         )
    #     except AssertionError as e:
    #         x = np.linspace(eps, 1 - eps, 100)
    #         A = make_ellipsoid_matrix(c, d, theta)
    #         B = make_ellipsoid_matrix(a, b, 0)
    #         fig, ax = plt.subplots()
    #         ax.plot(x, [k_lambda(xi, A, B, v_ij) for xi in x])
    #         ax.hlines(0, 0, 1, "k", "dashed")
    #         plt.show()
    #         if np.isclose(d, 0.00073799425736145):
    #             continue
    #         raise AssertionError from e

    # RUST TESTING:
    theta = 0.0
    # theta = 1.0471975512
    a, b, c, d = 1, 4, 1, 4
    v_ij = [2.000_001, 0.0]
    np.testing.assert_equal(
        jen := ellipsoid_intersects(a, b, c, d, theta, v_ij),
        hoomd_ellipse_intersect(a, b, c, d, v_ij, theta),
        err_msg=f"FAILED: {jen}\n(a,b,c,d), theta, v_ij{(a, b, c, d), theta, v_ij}",
    )

    a, b, c, d = 0.7535646805172947, 0.20732634327455568, 0.14933294475326375, 0.8471011744974546
    angle = 5.106726887989417
    v_ij = [0.04081238900788686, -0.6011623690559856]
    np.testing.assert_equal(
        jen := ellipsoid_intersects(a, b, c, d, theta, v_ij),
        hoomd_ellipse_intersect(a, b, c, d, v_ij, theta),
        err_msg=f"FAILED: {jen}\n(a,b,c,d), theta, v_ij{(a, b, c, d), theta, v_ij}",
    )
    (a,b,c,d)= (0.7935819558628192, 0.23905199303700841, 0.2825358620667253, 0.6410462098943438)
    angle= 4.701645821021286
    v_ij= [-0.5559190864186035, 0.1638933238991076]
    np.testing.assert_equal(
        jen := ellipsoid_intersects(a, b, c, d, theta, v_ij),
        hoomd_ellipse_intersect(a, b, c, d, v_ij, theta),
        err_msg=f"FAILED: {jen}\n(a,b,c,d), theta, v_ij{(a, b, c, d), theta, v_ij}",
    )
    (a,b,c,d)= (0.23740347193565203, 0.43097005556175605, 0.7874982300687023, 0.75142149478203)
    angle= -1.9578679359334243
    v_ij= [-1.0015674181368373, 0.021302054078231603]
    np.testing.assert_equal(
        jen := ellipsoid_intersects(a, b, c, d, theta, v_ij),
        hoomd_ellipse_intersect(a, b, c, d, v_ij, theta),
        err_msg=f"FAILED: {jen}\n(a,b,c,d), theta, v_ij{(a, b, c, d), theta, v_ij}",
    )
