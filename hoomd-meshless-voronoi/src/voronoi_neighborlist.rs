// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

/// TODO: documentation
/// Contructs power diagrams using TODO: link algorithm. This algorithm works by lifting spheres in N-dimensional space (stored as `PDSeed<N>`) to points in (N+1)-dimensional space, and then constructing the voronoi diagram. 
///
use kd_tree::KdTree;
use robust::{Coord, Coord3D, orient2d, orient3d};
use thiserror::Error;

/// A point in a power diagram. `coordinate` stores the coordinates of the sphere center, `weight` stores the squared radius of the sphere, and `index` labels the point.
pub struct PDSeed<const N: usize> {
    pub coordinate: [f64; N],
    pub weight: f64,
    pub index: usize,
}

impl<const N: usize> PDSeed<N> {
    /// get the coordinate of the sphere center
    pub fn coordinate(&self) -> [f64; N] {
        self.coordinate
    }
    /// get the squared radius of the sphere
    pub fn weight(&self) -> f64 {
        self.weight
    }
    /// get the index of the point
    pub fn index(&self) -> usize {
        self.index
    }
}

/// The "lifted" points which are used to construct the voronoi tesselation.
#[derive(Clone, Debug)]
pub struct LiftedSeed<const N: usize> {
    coordinate: [f64; N],
    index: usize,
}

impl<const N: usize> LiftedSeed<N> {
    /// get the coordinate of the lifted seed
    pub fn coordinate(&self) -> [f64; N] {
        self.coordinate
    }
    /// get the index of the lifted seed
    pub fn index(&self) -> usize {
        self.index
    }
}

/// The voronoi cells constructed from the lifted seeds.
pub struct LiftedCells<const N: usize> {
    center_point_index: usize,
    neighbor_indices: Vec<usize>,
    vertices: Vec<[f64; N]>,
}

#[allow(dead_code)]
impl<const N: usize> LiftedCells<N> {
    /// get the seed of the voronoi cell
    pub fn center_point(&self) -> usize {
        self.center_point_index
    }
    /// get the indices of neighboring voronoi cells
    pub fn neighbor_indices(&self) -> Vec<usize> {
        self.neighbor_indices.clone()
    }
    /// get the vertices of the voronoi cell
    pub fn vertices(&self) -> Vec<[f64; N]> {
        self.vertices.clone()
    }
}

/// A power diagram in N-dimensional space, i.e., the seeds are N-dimensional
pub struct PowerDiagram<const N: usize> {
    cells: Vec<LiftedCells<N>>,
}

impl<const N: usize> PowerDiagram<N> {
    /// get a vector of nearest neighbors. Output is a vector of tuples containing the seed index and a vector of that seed's nearest neighbors.
    pub fn neighborlist(&self) -> Vec<(usize, Vec<usize>)> {
        self.cells
            .iter()
            .map(|ls| (ls.center_point_index, ls.neighbor_indices()))
            .collect()
    }
}

/// Defines methods to build power diagrams
pub trait GeneratePowerDiagram<const N: usize> {
    fn build(
        seeds: &[PDSeed<N>],
        simulation_box_vertices: Vec<[f64; N]>,
        exp_n: usize,
    ) -> Result<PowerDiagram<N>, Error>;
}

impl PowerDiagram<2> {
    /// Clip subject vertices along bisector of pt_i and pt_j
    pub fn clip_2d(
        subject_vertices: &[[f64; 2]],
        center_pt: &[f64; 3],
        pt_i_u: &[f64; 3],
        pt_j_u: &[f64; 3],
    ) -> (Vec<[f64; 2]>, bool) {
        // center at centroid
        let mut temp_cell: Vec<[f64; 2]> = subject_vertices
            .iter()
            .map(|s| [s[0] - center_pt[0], s[1] - center_pt[1]])
            .collect();
        let input_list: Vec<[f64; 2]> = temp_cell.clone();
        temp_cell = vec![];
        let mut clipped = false;
        let pt_i: [f64; 3] = [
            pt_i_u[0] - center_pt[0],
            pt_i_u[1] - center_pt[1],
            pt_i_u[2] - center_pt[2],
        ];
        let pt_j: [f64; 3] = [
            pt_j_u[0] - center_pt[0],
            pt_j_u[1] - center_pt[1],
            pt_j_u[2] - center_pt[2],
        ];

        // put in-plane bisectors in clockwise order
        let lambda = (pt_j[2].powi(2) - pt_i[2].powi(2))
            / (2.0 * ((pt_j[0] - pt_i[0]).powi(2) + (pt_j[1] - pt_i[1]).powi(2)));
        let x_1 = (pt_i[1] - pt_j[1]) + (0.5) * (pt_j[0] + pt_i[0]) + lambda * (pt_j[0] - pt_i[0]);
        let y_1 = (pt_j[0] - pt_i[0]) + (0.5) * (pt_j[1] + pt_i[1]) + lambda * (pt_j[1] - pt_i[1]);
        let x_2 =
            -1.0 * (pt_i[1] - pt_j[1]) + (0.5) * (pt_j[0] + pt_i[0]) + lambda * (pt_j[0] - pt_i[0]);
        let y_2 =
            -1.0 * (pt_j[0] - pt_i[0]) + (0.5) * (pt_j[1] + pt_i[1]) + lambda * (pt_j[1] - pt_i[1]);
        let pt_a = Coord { x: x_1, y: y_1 };
        let pt_b = Coord { x: x_2, y: y_2 };
        let pt_c = Coord { x: 0.0, y: 0.0 };
        let counterclockwise: bool = orient2d(pt_a, pt_b, pt_c) > 0.0;

        let bisect_a: Coord<f64>;
        let bisect_b: Coord<f64>;
        if counterclockwise {
            bisect_a = pt_a;
            bisect_b = pt_b;
        } else {
            bisect_a = pt_b;
            bisect_b = pt_a;
        }
        for i in 0..input_list.len() {
            let current_pt = input_list[(i + 1).rem_euclid(input_list.len())];
            let prev_pt = input_list[i.rem_euclid(input_list.len())];
            // set as Coord struct for robust crate
            let vertex_p1 = Coord {
                x: prev_pt[0],
                y: prev_pt[1],
            };
            let vertex_p2 = Coord {
                x: current_pt[0],
                y: current_pt[1],
            };
            let t: f64 = ((0.5)
                * (pt_j[0].powi(2) + pt_j[1].powi(2) + pt_j[2].powi(2)
                    - pt_i[0].powi(2)
                    - pt_i[1].powi(2)
                    - pt_i[2].powi(2))
                - current_pt[0] * (pt_j[0] - pt_i[0])
                - current_pt[1] * (pt_j[1] - pt_i[1]))
                / ((prev_pt[0] - current_pt[0]) * (pt_j[0] - pt_i[0])
                    + (prev_pt[1] - current_pt[1]) * (pt_j[1] - pt_i[1]));
            // intersection of clipping plane and edge of cell
            let new_vertex: [f64; 2] = [
                prev_pt[0] * t + current_pt[0] * (1_f64 - t),
                prev_pt[1] * t + current_pt[1] * (1_f64 - t),
            ];
            // check if current vertices are inside the clipping plane
            let prev_is_inside = orient2d(bisect_a, bisect_b, vertex_p1) > 0.0;
            let current_is_inside = orient2d(bisect_a, bisect_b, vertex_p2) > 0.0;

            if current_is_inside {
                if !prev_is_inside {
                    temp_cell.push(new_vertex);
                    clipped = true;
                }
                temp_cell.push(current_pt);
            } else if prev_is_inside {
                temp_cell.push(new_vertex);
                clipped = true;
            }
        }
        let final_cell = temp_cell
            .iter()
            .map(|s| [s[0] + center_pt[0], s[1] + center_pt[1]])
            .collect();
        (final_cell, clipped)
    }
}

impl PowerDiagram<3> {
    /// Clip subject vertices along bisector of pt_i and pt_j
    pub fn clip_3d(
        subject_vertices: &[[f64; 3]],
        center_pt: &[f64; 4],
        pt_i_u: &[f64; 4],
        pt_j_u: &[f64; 4],
    ) -> (Vec<[f64; 3]>, bool) {
        // center at centroid
        let mut temp_cell: Vec<[f64; 3]> = subject_vertices
            .iter()
            .map(|s| {
                [
                    s[0] - center_pt[0],
                    s[1] - center_pt[1],
                    s[2] - center_pt[2],
                ]
            })
            .collect();
        let input_list: Vec<[f64; 3]> = temp_cell.clone();
        temp_cell = vec![];
        let mut clipped = false;
        let pt_i: [f64; 4] = [
            pt_i_u[0] - center_pt[0],
            pt_i_u[1] - center_pt[1],
            pt_i_u[2] - center_pt[2],
            pt_i_u[3] - center_pt[3],
        ];
        let pt_j: [f64; 4] = [
            pt_j_u[0] - center_pt[0],
            pt_j_u[1] - center_pt[1],
            pt_j_u[2] - center_pt[2],
            pt_j_u[3] - center_pt[3],
        ];

        // generate three points on bisecting plane, then put them in clockwise order
        let lambda = (pt_j[3].powi(2) - pt_i[3].powi(2))
            / (2.0
                * ((pt_j[0] - pt_i[0]).powi(2)
                    + (pt_j[1] - pt_i[1]).powi(2)
                    + (pt_j[2] - pt_i[2]).powi(2)));
        let x_1 = (pt_i[1] - pt_j[1]) + (0.5) * (pt_j[0] + pt_i[0]) + lambda * (pt_j[0] - pt_i[0]);
        let y_1 = (pt_j[0] - pt_i[0]) + (0.5) * (pt_j[1] + pt_i[1]) + lambda * (pt_j[1] - pt_i[1]);
        let z_1 = (0.5) * (pt_j[2] + pt_i[2]) + lambda * (pt_j[2] - pt_i[2]);
        let x_2 = -1.0 * (pt_i[1] - pt_j[1]) + (0.5) * (pt_j[0] + pt_i[0]);
        let y_2 = -1.0 * (pt_j[0] - pt_i[0]) + (0.5) * (pt_j[1] + pt_i[1]);
        let z_2 = (0.5) * (pt_j[2] + pt_i[2]) + lambda * (pt_j[2] - pt_i[2]);
        let x_3 = (pt_i[2] - pt_j[2]) + (0.5) * (pt_j[0] + pt_i[0]) + lambda * (pt_j[0] - pt_i[0]);
        let y_3 = (0.5) * (pt_j[1] + pt_i[1]) + lambda * (pt_j[1] - pt_i[1]);
        let z_3 = (pt_j[0] - pt_i[0]) + (0.5) * (pt_j[2] + pt_i[2]) + lambda * (pt_j[2] - pt_i[2]);
        let pt_a = Coord3D {
            x: x_1,
            y: y_1,
            z: z_1,
        };
        let pt_b = Coord3D {
            x: x_2,
            y: y_2,
            z: z_2,
        };
        let pt_c = Coord3D {
            x: x_3,
            y: y_3,
            z: z_3,
        };
        let pt_d = Coord3D {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
        // first check if a,b,c occur in counterclockwise order (when viewed from above the plane)
        let bisect_counterclockwise: bool = orient2d(
            Coord { x: x_1, y: y_1 },
            Coord { x: x_2, y: y_2 },
            Coord { x: x_3, y: y_3 },
        ) > 0.0;
        let below: bool = if bisect_counterclockwise {
            orient3d(pt_a, pt_b, pt_c, pt_d) > 0.0
        } else {
            orient3d(pt_a, pt_c, pt_b, pt_d) > 0.0
        };

        let (bisect_a, bisect_b, bisect_c): (Coord3D<f64>, Coord3D<f64>, Coord3D<f64>);

        if below {
            (bisect_a, bisect_b, bisect_c) = (pt_a, pt_b, pt_c);
        } else {
            (bisect_a, bisect_b, bisect_c) = (pt_a, pt_c, pt_b);
        }

        for i in 0..input_list.len() {
            let current_pt = input_list[(i + 1).rem_euclid(input_list.len())];
            let prev_pt = input_list[i.rem_euclid(input_list.len())];
            // set as Coord3D struct for robust crate
            let vertex_p1 = Coord3D {
                x: prev_pt[0],
                y: prev_pt[1],
                z: prev_pt[2],
            };
            let vertex_p2 = Coord3D {
                x: current_pt[0],
                y: current_pt[1],
                z: current_pt[2],
            };
            let t: f64 = ((0.5)
                * (pt_j[0].powi(2) + pt_j[1].powi(2) + pt_j[2].powi(2) + pt_j[3].powi(2)
                    - pt_i[0].powi(2)
                    - pt_i[1].powi(2)
                    - pt_i[2].powi(2)
                    - pt_i[3].powi(2))
                - current_pt[0] * (pt_j[0] - pt_i[0])
                - current_pt[1] * (pt_j[1] - pt_i[1])
                - current_pt[2] * (pt_j[2] - pt_i[2]))
                / ((prev_pt[0] - current_pt[0]) * (pt_j[0] - pt_i[0])
                    + (prev_pt[1] - current_pt[1]) * (pt_j[1] - pt_i[1])
                    + (prev_pt[2] - current_pt[2]) * (pt_j[2] - pt_i[2]));
            // intersection of clipping plane and edge of cell
            let new_vertex: [f64; 3] = [
                prev_pt[0] * t + current_pt[0] * (1_f64 - t),
                prev_pt[1] * t + current_pt[1] * (1_f64 - t),
                prev_pt[2] * t + current_pt[2] * (1_f64 - t),
            ];
            // check if current vertices are inside the clipping plane
            let prev_is_below = orient3d(bisect_a, bisect_b, bisect_c, vertex_p1) > 0.0;
            let current_is_below = orient3d(bisect_a, bisect_b, bisect_c, vertex_p2) > 0.0;

            let origin_below = orient3d(bisect_a, bisect_b, bisect_c, pt_d) > 0.0; //orient2d(Coord{x:x_1,y:y_1}, Coord{x:x_2,y:y_2}, Coord{x:0.0,y:0.0}) > 0.0;
            let prev_is_inside =
                (prev_is_below && origin_below) || (!prev_is_below && !origin_below);
            let current_is_inside =
                (current_is_below && origin_below) || (!current_is_below && !origin_below);
            if current_is_inside {
                if !prev_is_inside {
                    temp_cell.push(new_vertex);
                    clipped = true;
                }
                temp_cell.push(current_pt);
            } else if prev_is_inside {
                temp_cell.push(new_vertex);
                clipped = true;
            }
        }
        let final_cell = temp_cell
            .iter()
            .map(|s| {
                [
                    s[0] + center_pt[0],
                    s[1] + center_pt[1],
                    s[2] + center_pt[2],
                ]
            })
            .collect();
        (final_cell, clipped)
    }
}

impl GeneratePowerDiagram<2> for PowerDiagram<2> {
    fn build(
        seeds: &[PDSeed<2>],
        simulation_box_vertices: Vec<[f64; 2]>,
        exp_n: usize,
    ) -> Result<Self, Error> {
        let mut output_cells: Vec<LiftedCells<2>> = vec![];
        let max_weight_op: Option<f64> = seeds
            .iter()
            .map(|s| s.weight())
            .max_by(|a, b| a.total_cmp(b));
        let w_max: f64 = match max_weight_op {
            Some(max_weight) => max_weight,
            None => return Err(Error::EmptySeeds),
        };
        // lift seeds
        let lift = |s: &PDSeed<2>| {
            let point = [
                s.coordinate()[0],
                s.coordinate()[1],
                (w_max - s.weight()).sqrt(),
            ];
            LiftedSeed {
                coordinate: point,
                index: s.index(),
            }
        };
        let lifted_seeds: Vec<LiftedSeed<3>> = seeds.iter().map(lift).collect();
        let kd_tree = KdTree::build_by_ordered_float(
            lifted_seeds
                .iter()
                .map(|s| (s.coordinate, s.index))
                .collect(),
        );
        let mut stack: Vec<LiftedSeed<3>> = vec![];
        let mut visited: Vec<usize> = vec![];
        for (n, seed) in lifted_seeds.iter().enumerate() {
            if !visited.contains(&n) {
                stack.push(seed.clone());
            }
            visited.push(n);
            while let Some(p_i) = stack.pop() {
                // closure for computing max distances
                let dist = |v: &[f64; 2]| {
                    ((v[0] - p_i.coordinate[0]).powi(2)
                        + (v[1] - p_i.coordinate[1]).powi(2)
                        + p_i.coordinate[2].powi(2))
                    .sqrt()
                };
                let cell = simulation_box_vertices.clone();
                let mut d_max = cell.iter().map(dist).fold(f64::NEG_INFINITY, f64::max);
                let mut k: usize = exp_n;
                let mut k_nlist: Vec<&([f64; 3], usize)> = kd_tree
                    .nearests(&(p_i.coordinate, p_i.index), k)
                    .iter()
                    .map(|neighbor| neighbor.item)
                    .collect();
                let mut j: usize = 1;
                let mut p_i_nlist: Vec<usize> = vec![];
                let mut temp_cell: Vec<[f64; 2]> = cell.clone();
                while d_max
                    > (0.5)
                        * ((p_i.coordinate[0] - k_nlist[j].0[0]).powi(2)
                            + (p_i.coordinate[1] - k_nlist[j].0[1]).powi(2)
                            + (p_i.coordinate[2] - k_nlist[j].0[2]).powi(2))
                        .sqrt()
                {
                    // if quality satisfied, then j-th neighbor shares an edge
                    // now clip the cell
                    let clipped: bool;
                    (temp_cell, clipped) =
                        Self::clip_2d(&temp_cell, &p_i.coordinate, &p_i.coordinate, &k_nlist[j].0);
                    if clipped {
                        p_i_nlist.push(k_nlist[j].1);
                        // add to stack if not already visited
                        if !visited.contains(&k_nlist[j].1) {
                            stack.push(LiftedSeed {
                                coordinate: k_nlist[j].0,
                                index: k_nlist[j].1,
                            });
                        }
                        visited.push(k_nlist[j].1);
                    }
                    // recompute max cell distance using updated cell vertices
                    if j == k_nlist.len() - 1 {
                        d_max = -1.0; //break out of cycle if we've run out of nearest neighbors
                    } else {
                        d_max = temp_cell.iter().map(dist).fold(-1.3_f64, f64::max);
                        j += 1;
                    }
                    if j == k {
                        k *= 2;
                        k_nlist = kd_tree
                            .nearests(&(p_i.coordinate, p_i.index), k)
                            .iter()
                            .map(|neighbor| neighbor.item)
                            .collect();
                    }
                }
                output_cells.push(LiftedCells {
                    center_point_index: p_i.index,
                    neighbor_indices: p_i_nlist,
                    vertices: temp_cell,
                });
            }
        }
        output_cells.sort_by_key(|cell| cell.center_point_index);
        Ok(PowerDiagram {
            cells: output_cells,
        })
    }
}

impl GeneratePowerDiagram<3> for PowerDiagram<3> {
    fn build(
        seeds: &[PDSeed<3>],
        simulation_box_vertices: Vec<[f64; 3]>,
        exp_n: usize,
    ) -> Result<Self, Error> {
        let mut output_cells: Vec<LiftedCells<3>> = vec![];
        let max_weight_op: Option<f64> = seeds
            .iter()
            .map(|s| s.weight())
            .max_by(|a, b| a.total_cmp(b));
        let w_max: f64 = match max_weight_op {
            Some(max_weight) => max_weight,
            None => return Err(Error::EmptySeeds),
        };
        // lift seeds
        let lift = |s: &PDSeed<3>| {
            let point = [
                s.coordinate()[0],
                s.coordinate()[1],
                s.coordinate()[2],
                (w_max - s.weight()).sqrt(),
            ];
            LiftedSeed {
                coordinate: point,
                index: s.index(),
            }
        };
        let lifted_seeds: Vec<LiftedSeed<4>> = seeds.iter().map(lift).collect();
        let kd_tree = KdTree::build_by_ordered_float(
            lifted_seeds
                .iter()
                .map(|s| (s.coordinate, s.index))
                .collect(),
        );
        let mut stack: Vec<LiftedSeed<4>> = vec![];
        let mut visited: Vec<usize> = vec![];
        for (n, l_seed) in lifted_seeds.iter().enumerate() {
            //stack.push(lifted_seeds[n].clone());
            if !visited.contains(&n) {
                stack.push(l_seed.clone());
            }
            visited.push(n);
            while let Some(p_i) = stack.pop() {
                // closure for computing max distances
                let dist = |v: &[f64; 3]| {
                    ((v[0] - p_i.coordinate[0]).powi(2)
                        + (v[1] - p_i.coordinate[1]).powi(2)
                        + (v[2] - p_i.coordinate[2]).powi(2)
                        + p_i.coordinate[3].powi(2))
                    .sqrt()
                };
                let cell = simulation_box_vertices.clone();
                let mut d_max = cell.iter().map(dist).fold(f64::NEG_INFINITY, f64::max);
                let mut k: usize = exp_n;
                let mut k_nlist: Vec<&([f64; 4], usize)> = kd_tree
                    .nearests(&(p_i.coordinate, p_i.index), k)
                    .iter()
                    .map(|neighbor| neighbor.item)
                    .collect();
                let mut j: usize = 1;
                let mut p_i_nlist: Vec<usize> = vec![];
                let mut temp_cell: Vec<[f64; 3]> = cell.clone();
                while d_max
                    > (0.5)
                        * ((p_i.coordinate[0] - k_nlist[j].0[0]).powi(2)
                            + (p_i.coordinate[1] - k_nlist[j].0[1]).powi(2)
                            + (p_i.coordinate[2] - k_nlist[j].0[2]).powi(2)
                            + (p_i.coordinate[3] - k_nlist[j].0[3]).powi(2))
                        .sqrt()
                {
                    // if quality satisfied, then j-th neighbor shares an edge
                    // now clip the cell
                    let clipped: bool;
                    (temp_cell, clipped) =
                        Self::clip_3d(&temp_cell, &p_i.coordinate, &p_i.coordinate, &k_nlist[j].0);
                    if clipped {
                        p_i_nlist.push(k_nlist[j].1);
                        // add to stack if not already visited
                        if !visited.contains(&k_nlist[j].1) {
                            stack.push(LiftedSeed {
                                coordinate: k_nlist[j].0,
                                index: k_nlist[j].1,
                            });
                        }
                        visited.push(k_nlist[j].1);
                    }
                    // recompute max cell distance using updated cell vertices
                    if j == k_nlist.len() - 1 {
                        d_max = -1.0; //break out of cycle if we've run out of nearest neighbors
                    } else {
                        d_max = temp_cell.iter().map(dist).fold(-1.0_f64, f64::max);
                        j += 1;
                    }
                    if j == k {
                        k *= 2;
                        k_nlist = kd_tree
                            .nearests(&(p_i.coordinate, p_i.index), k)
                            .iter()
                            .map(|neighbor| neighbor.item)
                            .collect();
                    }
                }
                output_cells.push(LiftedCells {
                    center_point_index: p_i.index,
                    neighbor_indices: p_i_nlist,
                    vertices: temp_cell,
                });
            }
        }
        output_cells.sort_by_key(|cell| cell.center_point_index);
        Ok(PowerDiagram {
            cells: output_cells,
        })
    }
}

/// Enumerate possible sources of error
#[non_exhaustive]
#[derive(Error, PartialEq, Debug)]
pub enum Error {
    /// Given microstate has no valid indices
    #[error("invalid site index")]
    InvalidSiteIndex,
    /// No nearest neighbors
    #[error("No nearest neighbors (likely an invalid site index)")]
    NoNearestNeighbors,
    /// iterator of seeds is empty
    #[error("given vector of seeds is empty or has no maximum")]
    EmptySeeds,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nlist_2d() -> Result<(), Box<dyn std::error::Error>> {
        let points = [[-0.9, 1.0], [1.1, 1.0], [0.5, -0.5]];
        let seeds: Vec<PDSeed<2>> = points
            .iter()
            .enumerate()
            .map(|(id, s)| PDSeed {
                coordinate: *s,
                weight: 0.0,
                index: id,
            })
            .collect();
        let simulation_box_vertices = vec![[2.0, 2.0], [-2.0, 2.0], [-2.0, -2.0], [2.0, -2.0]];
        let power_diagram = PowerDiagram::<2>::build(&seeds, simulation_box_vertices, 6_usize)?;
        let pd_nlist = power_diagram.neighborlist();

        assert_eq!(
            vec![
                (0_usize, vec![1_usize, 2_usize]),
                (1_usize, vec![2_usize, 0_usize]),
                (2_usize, vec![1_usize, 0_usize])
            ],
            pd_nlist
        );
        Ok(())
    }

    #[test]
    fn nlist_3d() -> Result<(), Box<dyn std::error::Error>> {
        let points = [
            [-0.9, -1.0, -1.1],
            [0.9, 1.1, -1.3],
            [0.0, 0.0, 0.0],
            [1.1, 1.0, 0.5],
            [0.5, -0.5, 0.3],
        ];
        let seeds: Vec<PDSeed<3>> = points
            .iter()
            .enumerate()
            .map(|(id, s)| PDSeed {
                coordinate: *s,
                weight: 0.0,
                index: id,
            })
            .collect();
        let simulation_box_vertices = vec![
            [2.0, 2.0, 2.0],
            [-2.0, 2.0, 2.0],
            [-2.0, -2.0, 2.0],
            [2.0, -2.0, 2.0],
            [2.0, 2.0, -2.0],
            [-2.0, 2.0, -2.0],
            [-2.0, -2.0, -2.0],
            [2.0, -2.0, -2.0],
        ];
        let power_diagram = PowerDiagram::<3>::build(&seeds, simulation_box_vertices, 6_usize)?;
        let pd_nlist = power_diagram.neighborlist();

        assert_eq!(
            vec![
                (0_usize, vec![2_usize, 4_usize, 1_usize]),
                (1_usize, vec![3_usize, 2_usize, 4_usize, 0_usize]),
                (2_usize, vec![4_usize, 3_usize, 0_usize, 1_usize]),
                (3_usize, vec![2_usize, 4_usize, 1_usize]),
                (4_usize, vec![2_usize, 3_usize, 0_usize, 1_usize])
            ],
            pd_nlist
        );
        Ok(())
    }

    #[test]
    fn clipping_in_2d() -> Result<(), Box<dyn std::error::Error>> {
        let example_box = vec![[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]];

        let point_1_a: [f64; 3] = [-0.5, -0.5, 0.0];
        let point_2_a: [f64; 3] = [1.5, 1.5, 0.0];
        let clipped_a =
            PowerDiagram::<2>::clip_2d(&example_box, &[0.0, 0.0, 0.0], &point_1_a, &point_2_a);
        assert_eq!(
            clipped_a.0,
            vec![
                [0.0, 1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 0.0]
            ]
        );

        let point_1_b: [f64; 3] = [-0.5, 0.5, 1.0];
        let point_2_b: [f64; 3] = [1.5, 0.5, 1.0];
        let clipped_b =
            PowerDiagram::<2>::clip_2d(&example_box, &[0.75, 0.0, 1.0], &point_1_b, &point_2_b);
        assert_eq!(
            clipped_b.0,
            vec![[0.5, 1.0], [0.5, -1.0], [1.0, -1.0], [1.0, 1.0]]
        );

        let point_1_c: [f64; 3] = [0.0, 0.0, 1.0];
        let point_2_c: [f64; 3] = [0.0, 1.5, 1.0];
        let clipped_c =
            PowerDiagram::<2>::clip_2d(&example_box, &[0.0, 0.0, 0.0], &point_1_c, &point_2_c);
        assert_eq!(
            clipped_c.0,
            vec![[-1.0, 0.75], [-1.0, -1.0], [1.0, -1.0], [1.0, 0.75]]
        );
        Ok(())
    }

    #[test]
    fn clipping_in_3d() -> Result<(), Box<dyn std::error::Error>> {
        let example_cube = vec![
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
        ];

        let point_1_a: [f64; 4] = [0.25, 0.0, 0.0, 0.0];
        let point_2_a: [f64; 4] = [1.25, 0.0, 0.0, 0.0];
        let clipped_a = PowerDiagram::<3>::clip_3d(
            &example_cube,
            &[0.85, 0.0, 0.0, 0.0],
            &point_1_a,
            &point_2_a,
        );
        assert_eq!(
            clipped_a.0,
            vec![
                [0.75, 1.0, 1.0],
                [0.75, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [0.75, 1.0, -1.0],
                [0.75, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, 1.0]
            ]
        );
        Ok(())
    }
}
