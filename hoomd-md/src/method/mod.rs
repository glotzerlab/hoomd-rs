// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Integration methods.

mod constant_volume;
use std::array;

pub use constant_volume::{ConstantVolume, ConstantVolumeBuilder};

use hoomd_microstate::property::RotationalMotionTypes;
use hoomd_vector::{Angle, Cartesian, InnerProduct, Quaternion, Rotate, Rotation, Versor, Wedge};
mod langevin;
pub use langevin::Langevin;

mod brownian;
pub use brownian::Brownian;

// TODO: fix the katex build problem...

/// Symplecticly integrate rotational degrees of freedom.
/// 
/// This trait binds symplectic rotational integration schemes to the types that
/// represent orientation and its associated quantities: angular momentum,
/// moment of inertia, and net torque. Implement this trait on a type that
/// represents body orientation to make a [`Microstate`] containing such bodies
/// integrateable with [`ConstantVolume`] and [`Langevin`].
/// 
/// [`Microstate`]: hoomd_microstate::Microstate
pub trait SymplecticIntegrateRotation
where
    <Self as SymplecticIntegrateRotation>::Rotation: RotationalMotionTypes,
{
    /// Type that represents a body's orientation.
    type Rotation;

    /// Type that represents a body's net torque.
    type NetTorque;

    /// Integrate orientation forward a full step and angular momentum forward a half step.
    fn half_step_one(
        delta_t: f64,
        net_torque: &Self::NetTorque,
        angular_momentum: &mut <Self::Rotation as RotationalMotionTypes>::AngularMomentum,
        orientation: &mut Self::Rotation,
        moment_of_inertia: &<Self::Rotation as RotationalMotionTypes>::MomentOfInertia,
    );

    /// Integrate angular momentum forward a half step.
    fn half_step_two(
        delta_t: f64,
        net_torque: &Self::NetTorque,
        angular_momentum: &mut <Self::Rotation as RotationalMotionTypes>::AngularMomentum,
        orientation: & Self::Rotation,
        moment_of_inertia: &<Self::Rotation as RotationalMotionTypes>::MomentOfInertia,
    );
}

/// Symplectic rotational integration for bodies in 2-dimensional cartesian space.
impl SymplecticIntegrateRotation for Angle {
    type Rotation = Angle;
    type NetTorque = <Cartesian<2> as Wedge>::Bivector;

    /// Integrate orientation forward a full step and angular momentum forward a half step.
    ///
    /// The first half step of the symplectic rotational integration procedure
    /// in 2-dimensional cartesian space is given by the equations below, which
    /// are applied to each body *i*. Bodies which have
    /// `moment_of_inertia = 0.0` are skipped.
    ///
    /// 1. Angular momentum is integrated forward a half step.
    ///
    ///    ```math
    ///    L_i\left(t + \frac{\Delta t}{2} \right) = L_i(t) + \tau_i(t) \frac{\Delta t}{2}
    ///    ```
    ///
    /// 2. Orientation is integrated forward a full step using the new angular
    ///    momentum.
    ///
    ///    ```math
    ///    \theta_i(t + \Delta t) = \theta_i(t) + \frac{L_i\left( t + \frac{\Delta t}{2} \right)}{I_i} \Delta t
    ///    ```
    fn half_step_one(
        delta_t: f64,
        net_torque: &Self::NetTorque,
        angular_momentum: &mut <Self::Rotation as RotationalMotionTypes>::AngularMomentum,
        orientation: &mut Self::Rotation,
        moment_of_inertia: &<Self::Rotation as RotationalMotionTypes>::MomentOfInertia,
    ) {
        // Return early if there is no rotational degree of freedom.
        if *moment_of_inertia == 0.0 {
            return
        }

        *angular_momentum += net_torque * 0.5 * delta_t;
        orientation.theta += *angular_momentum / moment_of_inertia * delta_t;
        *orientation = orientation.to_reduced();
    }

    /// Integrate angular momentum forward a half step.
    ///
    /// The second half step of the symplectic integration procedure in
    /// 2-dimensional cartesian space is given by the equation below, which is
    /// applied to each body *i*. Bodies which have `moment_of_inertia = 0.0`
    /// are skipped.
    ///
    /// ```math
    /// L_i(t + \Delta t) = L_i\left( t + \frac{\Delta t}{2} \right) + \tau_i \left(t + \frac{\Delta t}{2} \right) \frac{\Delta t}{2}
    /// ```
    fn half_step_two(
        delta_t: f64,
        net_torque: &Self::NetTorque,
        angular_momentum: &mut <Self::Rotation as RotationalMotionTypes>::AngularMomentum,
        _orientation: & Self::Rotation,
        moment_of_inertia: &<Self::Rotation as RotationalMotionTypes>::MomentOfInertia,
    ) {
        // Return early if there is no rotational degree of freedom.
        if *moment_of_inertia == 0.0 {
            return
        }
        
        *angular_momentum += net_torque * 0.5 * delta_t;
    }
}

/// Symplectic rotational integration for bodies in 3-dimensional cartesian space.
impl SymplecticIntegrateRotation for Versor {
    type Rotation = Versor;
    type NetTorque = <Cartesian<3> as Wedge>::Bivector;

    /// Integrate orientation forward a full step and angular momentum forward a half step.
    ///
    /// The first half step of the symplectic rotational integration procedure
    /// in 2-dimensional cartesian space is given by the equations below, which
    /// are applied to each body *i*. In each step, the marker $`'`$ is used
    /// when a variable's value changes during a step to distinguish the value
    /// before ( $`'`$ is present) from the value after ( $`'`$ is absent).
    /// Rotational degrees of freedom with a moment of inertia component of zero
    /// are skipped.
    ///
    /// 1. Angular momentum $`\vec{L}`$ and orientation $`\mathbf{q}`$ are
    ///    integrated forward. These integrations follow a complex, multi-step
    ///    process, so a fuller explanation is provided below. In each step, the
    ///    body index *i* and time *t* are implicit on every variable unless
    ///    otherwise specified.
    ///
    ///    1. Angular momentum and net torque are converted to quaternions
    ///       $`\mathbf{p}`$ and $`\mathbf{f}`$, respectively:
    ///
    ///       ```math
    ///       \begin{align*}
    ///
    ///       \mathbf{p} &= 2\mathbf{S}(\mathbf{q}) \mathbf{L} \\
    ///       \mathbf{f} &= 2\mathbf{S}(\mathbf{q}) \boldsymbol{\tau} \\
    ///
    ///       \end{align*}
    ///       ```
    ///
    ///       where
    ///
    ///       ```math
    ///       \begin{align*}
    ///
    ///       \mathbf{L} &= (0, L_x, L_y, L_z) \\
    ///       \boldsymbol{\tau} &= (0, \tau_x, \tau_y, \tau_z) \\
    ///
    ///       \mathbf{S}(\mathbf{q}) &=
    ///       \begin{pmatrix}
    ///       q_0 & -q_1 & -q_2 & -q_3\\
    ///       q_1 & q_0 & -q_3 & q_2\\
    ///       q_2 & q_3 & q_0 & -q_1\\
    ///       q_3 & -q_2 & q_1 & q_0
    ///       \end{pmatrix}
    ///
    ///       \end{align*}
    ///       ```
    ///
    ///     2. $`\mathbf{p}`$ and $`\mathbf{q}`$ are integrated forward using
    ///        the novel symplectic quaternion scheme (`NO_SQUISH`) algorithm,
    ///        which ensures the integration is both symplectic and preserves
    ///        orientation quaternion unity. There are several steps to this
    ///        algorithm, whose equations are given below.
    ///
    ///        1. $`\mathbf{p}`$ is partially integrated forward a half step.
    ///
    ///            ```math
    ///            \mathbf{p} = \mathbf{p}' + \frac{\Delta t}{2} \mathbf{f}
    ///            ```
    ///
    ///        2. $`\mathbf{p}`$ is integrated forward the remainder of the half
    ///           step and $`\mathbf{q}`$ is integrated forward a full step.
    ///           Properties of quaternion algebra are used to decompose the
    ///           Liouvillian into a sum over permutation matrices applied to
    ///           $`\mathbf{q}`$ and $`\mathbf{p}`$. There are five steps to
    ///           this decomposition:
    ///
    ///           ```math
    ///           \begin{align*}
    ///
    ///           \phi_3 &= \frac{1}{4 I_{33}} \mathrm{dot} \left( \mathbf{p}, P_3 \mathbf{q} \right) \\
    ///           \mathbf{q} &= \cos{(\phi_3 \frac{\Delta t}{2})} \mathbf{q}^{'} +  \sin{(\phi_3 \frac{\Delta t}{2})} P_3 \mathbf{q}^{'} \nonumber \\
    ///           \mathbf{p} &= \cos{(\phi_3 \frac{\Delta t}{2})} \mathbf{p}' +  \sin{(\phi_3 \frac{\Delta t}{2})} P_3 \mathbf{p}' \nonumber \\ \nonumber \\
    ///
    ///           \phi_2 &= \frac{1}{4 I_{22}} \mathrm{dot} \left( \mathbf{p}, P_2 \mathbf{q} \right) \\
    ///           \mathbf{q} &= \cos{(\phi_2 \frac{\Delta t}{2})} \mathbf{q}^{'} +  \sin{(\phi_2 \frac{\Delta t}{2})} P_2 \mathbf{q}^{'} \nonumber \\
    ///           \mathbf{p} &= \cos{(\phi_2 \frac{\Delta t}{2})} \mathbf{p}' +  \sin{(\phi_2 \frac{\Delta t}{2})} P_2 \mathbf{p}' \nonumber \\ \nonumber \\
    ///
    ///           \phi_1 &= \frac{1}{4 I_{11}} \mathrm{dot} \left( \mathbf{p}, P_1 \mathbf{q} \right) \\
    ///           \mathbf{q} &= \cos{(\phi_1 \Delta t)} \mathbf{q}^{'} +  \sin{(\phi_1 \Delta t)} P_1 \mathbf{q}^{'} \nonumber \\
    ///           \mathbf{p} &= \cos{(\phi_1 \Delta t)} \mathbf{p}' +  \sin{(\phi_1 \Delta t)} P_1 \mathbf{p}' \nonumber  \nonumber \\ \nonumber \\
    ///
    ///           \phi_2 &= \frac{1}{4 I_{22}} \mathrm{dot} \left( \mathbf{p}, P_2 \mathbf{q} \right) \\
    ///           \mathbf{q} &= \cos{(\phi_2 \frac{\Delta t}{2})} \mathbf{q}^{'} +  \sin{(\phi_2 \frac{\Delta t}{2})} P_2 \mathbf{q}^{'} \nonumber \\
    ///           \mathbf{p} &= \cos{(\phi_2 \frac{\Delta t}{2})} \mathbf{p}' +  \sin{(\phi_2 \frac{\Delta t}{2})} P_2 \mathbf{p}' \nonumber  \nonumber \\ \nonumber \\
    ///
    ///           \phi_3 &= \frac{1}{4 I_{33}} \mathrm{dot} \left( \mathbf{p}, P_3 \mathbf{q} \right) \\
    ///           \mathbf{q} \left( t + \Delta t \right) &= \cos{(\phi_3 \frac{\Delta t}{2})} \mathbf{q}^{'} +  \sin{(\phi_3 \frac{\Delta t}{2})} P_3 \mathbf{q}^{'} \nonumber \\
    ///           \mathbf{p} \left( t + \frac{\Delta t}{2} \right) &= \cos{(\phi_3 \frac{\Delta t}{2})} \mathbf{p}' +  \sin{(\phi_3 \frac{\Delta t}{2})} P_3 \mathbf{p}' \nonumber    \nonumber \\ \nonumber \\
    ///
    ///           \end{align*}
    ///           ```
    ///
    ///           where $`I_{kk}`$ is the component of the moment of inertia for
    ///           $`k = 1, 2, 3`$ and $`P_k`$ is the corresponding permutation
    ///           matrix such that
    ///
    ///           ```math
    ///           \begin{align*}
    ///
    ///           P_0\mathbf{q} &= (q_0, q_1, q_2, q_3) \\
    ///           P_1\mathbf{q} &= (-q_1, q_0, q_3, -q_2) \\
    ///           P_2\mathbf{q} &= (-q_2, -q_3, q_0, q_1) \\
    ///           P_3\mathbf{q} &= (-q_3, q_2, -q_1, q_0) \\
    ///           (PP^T)_{\alpha \beta} &= \delta_{\alpha \beta} \\
    ///
    ///           \end{align*}
    ///            ```
    ///
    ///     3. $`\mathbf{p}`$ is converted back into vector-form angular momentum:
    ///
    ///        ```math
    ///        \mathbf{L} \left( t + \frac{\Delta t}{2} \right) = \frac{1}{2} \mathbf{S}(\mathbf{q})^T \mathbf{p} \left( t + \frac{\Delta t}{2} \right)
    ///        ```
    ///
    ///        where
    ///
    ///        ```math
    ///        \begin{align*}
    ///        \mathbf{L} &= (0, L_x, L_y, L_z) \\
    ///        \vec{L} &= (L_x, L_y, L_z)
    ///        \end{align*}
    ///        ```
    ///
    /// [rotational kinetic energy]: crate::compute::RotationalKineticEnergy
    fn half_step_one(
        delta_t: f64,
        net_torque: &Self::NetTorque,
        angular_momentum: &mut <Self::Rotation as RotationalMotionTypes>::AngularMomentum,
        orientation: &mut Self::Rotation,
        moment_of_inertia: &<Self::Rotation as RotationalMotionTypes>::MomentOfInertia,
    ) {
        let mut q = *orientation.get();

        // Transform net torque to the body frame and calculate which of the
        // three rotational degrees of freedom are active.
        let mut net_torque = orientation.inverted().rotate(net_torque);
        let active_dof: [bool; 3] = array::from_fn(|i| moment_of_inertia[i] != 0.0);

        // If there are no active degrees of freedom, return early.
        if active_dof.iter().all(|&i| !i) {
            return
        }

        // Limited numerical precision might lead to non-zero torques about axes
        // that should not be integrated. Zeroing these out improves the
        // stability of the integration.
        for i in 0..3 {
            if !active_dof[i] {
                net_torque[i] = 0.0;
            }
        }

        // DynamicOrientedPoint stores angular momentum in vector form. Convert it
        // into a quaternion, integrate the quaternion, then store it back as a vector.
        let s = *angular_momentum;
        let mut p = (q * Quaternion::pure(s)) * 2.0;

        // p = p * rescaling_factor + q * Quaternion::pure(net_torque) * delta_t;
        p += q * Quaternion::pure(net_torque) * delta_t;

        if active_dof[2] {
            let p3 = Quaternion::from([-p.vector[2], p.vector[1], -p.vector[0], p.scalar]);
            let q3 = Quaternion::from([-q.vector[2], q.vector[1], -q.vector[0], q.scalar]);
            let phi3 = (1. / (4. * moment_of_inertia[2]))
                * ((p.scalar * q3.scalar) + p.vector.dot(&q3.vector));
            let c_phi3 = (0.5 * delta_t * phi3).cos();
            let s_phi3 = (0.5 * delta_t * phi3).sin();

            p = p * c_phi3 + p3 * s_phi3;
            q = q * c_phi3 + q3 * s_phi3;
        }

        if active_dof[1] {
            let p2 = Quaternion::from([-p.vector[1], -p.vector[2], p.scalar, p.vector[0]]);
            let q2 = Quaternion::from([-q.vector[1], -q.vector[2], q.scalar, q.vector[0]]);
            let phi2 = (1. / (4. * moment_of_inertia[1]))
                * ((p.scalar * q2.scalar) + p.vector.dot(&q2.vector));
            let c_phi2 = (0.5 * delta_t * phi2).cos();
            let s_phi2 = (0.5 * delta_t * phi2).sin();

            p = p * c_phi2 + p2 * s_phi2;
            q = q * c_phi2 + q2 * s_phi2;
        }

        if active_dof[0] {
            let p1 = Quaternion::from([-p.vector[0], p.scalar, p.vector[2], -p.vector[1]]);
            let q1 = Quaternion::from([-q.vector[0], q.scalar, q.vector[2], -q.vector[1]]);
            let phi1 = (1. / (4. * moment_of_inertia[0]))
                * ((p.scalar * q1.scalar) + p.vector.dot(&q1.vector));
            let c_phi1 = (delta_t * phi1).cos();
            let s_phi1 = (delta_t * phi1).sin();

            p = p * c_phi1 + p1 * s_phi1;
            q = q * c_phi1 + q1 * s_phi1;
        }

        if active_dof[1] {
            let p2 = Quaternion::from([-p.vector[1], -p.vector[2], p.scalar, p.vector[0]]);
            let q2 = Quaternion::from([-q.vector[1], -q.vector[2], q.scalar, q.vector[0]]);
            let phi2 = (1. / (4. * moment_of_inertia[1]))
                * ((p.scalar * q2.scalar) + p.vector.dot(&q2.vector));
            let c_phi2 = (0.5 * delta_t * phi2).cos();
            let s_phi2 = (0.5 * delta_t * phi2).sin();

            p = p * c_phi2 + p2 * s_phi2;
            q = q * c_phi2 + q2 * s_phi2;
        }

        if active_dof[2] {
            let p3 = Quaternion::from([-p.vector[2], p.vector[1], -p.vector[0], p.scalar]);
            let q3 = Quaternion::from([-q.vector[2], q.vector[1], -q.vector[0], q.scalar]);
            let phi3 = (1. / (4. * moment_of_inertia[2]))
                * ((p.scalar * q3.scalar) + p.vector.dot(&q3.vector));
            let c_phi3 = (0.5 * delta_t * phi3).cos();
            let s_phi3 = (0.5 * delta_t * phi3).sin();

            p = p * c_phi3 + p3 * s_phi3;
            q = q * c_phi3 + q3 * s_phi3;
        }

        *orientation = q.to_versor().expect("body orientation should be non-zero");
        *angular_momentum = ((q.conjugate() * p) * 0.5).vector;

    }

    /// Integrate angular momentum forward a half step.
    ///
    /// The second half step of the symplectic integration procedure in
    /// 3-dimensional cartesian space is given by the equations below, which are
    /// applied to each body *i*. The time $`t + \frac{\Delta t}{2}`$ is
    /// implicit on every variable unless otherwise specified. Rotational
    /// degrees of freedom with a moment of inertia component of zero are
    /// skipped.
    ///
    /// 1. Angular momentum and net torque are converted to quaternions
    ///    $`\mathbf{p}`$ and $`\mathbf{f}`$, respectively:
    ///
    ///    ```math
    ///    \begin{align*}
    ///
    ///    \mathbf{p} &= 2\mathbf{S}(\mathbf{q}) \mathbf{L} \\
    ///    \mathbf{f} &= 2\mathbf{S}(\mathbf{q}) \boldsymbol{\tau} \\
    ///
    ///    \end{align*}
    ///    ```
    ///
    ///    where
    ///
    ///    ```math
    ///    \begin{align*}
    ///
    ///    \mathbf{L} &= (0, L_x, L_y, L_z) \\
    ///    \boldsymbol{\tau} &= (0, \tau_x, \tau_y, \tau_z) \\
    ///
    ///    \mathbf{S}(\mathbf{q}) &=
    ///    \begin{pmatrix}
    ///    q_0 & -q_1 & -q_2 & -q_3\\
    ///    q_1 & q_0 & -q_3 & q_2\\
    ///    q_2 & q_3 & q_0 & -q_1\\
    ///    q_3 & -q_2 & q_1 & q_0
    ///    \end{pmatrix}
    ///
    ///    \end{align*}
    ///     ```
    ///
    /// 2. $`\mathbf{p}`$ is integrated forward a half step.
    ///
    ///    ```math
    ///    \mathbf{p}\left( t + \Delta t \right) = \mathbf{p}\left( t + \frac{\Delta t}{2} \right) + \frac{\Delta t}{2} \mathbf{f}
    ///    ```
    ///
    /// 3. $`\mathbf{p}`$ is converted back into vector-form angular momentum:
    ///
    ///    ```math
    ///    \mathbf{L} \left( t + \Delta t \right) = \frac{1}{2} \mathbf{S}(\mathbf{q})^T \mathbf{p} \left( t + \Delta t \right)
    ///    ```
    ///
    ///    where
    ///
    ///    ```math
    ///    \begin{align*}
    ///    \mathbf{L} &= (0, L_x, L_y, L_z) \\
    ///    \vec{L} &= (L_x, L_y, L_z)
    ///    \end{align*}
    ///    ```
    fn half_step_two(
        delta_t: f64,
        net_torque: &Self::NetTorque,
        angular_momentum: &mut <Self::Rotation as RotationalMotionTypes>::AngularMomentum,
        orientation: & Self::Rotation,
        moment_of_inertia: &<Self::Rotation as RotationalMotionTypes>::MomentOfInertia,
    ) {
        // Transform net torque to the body frame and calculate which of the
        // three rotational degrees of freedom are active.
        let mut net_torque = orientation.inverted().rotate(net_torque);
        let active_dof: [bool; 3] = array::from_fn(|i| moment_of_inertia[i] != 0.0);

        // If there are no active degrees of freedom, return early.
        if active_dof.iter().all(|&i| !i) {
            return
        }

        // Limited numerical precision might lead to non-zero torques about axes
        // that should not be integrated. Zeroing these out improves the
        // stability of the integration.
        for i in 0..3 {
            if !active_dof[i] {
                net_torque[i] = 0.0;
            }
        }

        let q = *orientation.get();
        let s = *angular_momentum;

        let mut p = q * Quaternion::pure(s) * 2.0;

        p += (q * Quaternion::pure(net_torque)) * delta_t;

        *angular_momentum = ((q.conjugate() * p) * 0.5).vector;
    }
}
