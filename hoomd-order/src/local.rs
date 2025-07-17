// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//TODO: documentation 

/*! Implement Voronoi tesselations of a given point set 
*/
use hoomd_vector::{Vector, Cartesian, InnerProduct};
use hoomd_microstate::{Microstate, property::Point, boundary::Open};
use hoomd_manifold::Minkowski;
use std::array;
use crate::PowerDiagram;

/** Define power diagram object
TODO: documentation
*/
#[derive(Clone, Debug, PartialEq)]
pub struct PowerDiagramCenters {
    pub centers: Vec<Cartesian<3>>,
    pub radii: Vec<f64>,
    pub site_tags: Vec<usize>,
}

impl PowerDiagram for Microstate<Point<Cartesian<2>>, Point<Cartesian<2>>, Open> {
    fn power_diagram(&self) -> PowerDiagramCenters {
        let mut circle_centers = Vec::new();
        let mut circle_radii = Vec::new();
        let mut circle_tags = Vec::new();
        for site in self.sites() {
            circle_centers.push(
                Cartesian::from([
                    site.properties.position[0],
                    site.properties.position[1],
                    0.0,]));
            circle_radii.push(0.0_f64);
            circle_tags.push(site.site_tag);
        }
        PowerDiagramCenters{
            centers : circle_centers,
            radii : circle_radii,
            site_tags : circle_tags,
        }
    }
}

impl PowerDiagram for Microstate<Point<Cartesian<3>>, Point<Cartesian<3>>, Open> {
    fn power_diagram(&self) -> PowerDiagramCenters {
        let mut circle_centers = Vec::new();
        let mut circle_radii = Vec::new();
        let mut circle_tags = Vec::new();
        for site in self.sites() {
            circle_centers.push(site.properties.position);
            circle_radii.push(0.0_f64);
            circle_tags.push(site.site_tag);
        }
        PowerDiagramCenters{
            centers : circle_centers,
            radii : circle_radii,
            site_tags : circle_tags,
        }
    }
}

impl PowerDiagram for Microstate<Point<Minkowski<3>>, Point<Minkowski<3>>, Open> {
    fn power_diagram(&self) -> PowerDiagramCenters {
        let mut circle_centers = Vec::new();
        let mut circle_radii = Vec::new();
        let mut circle_tags = Vec::new();
        let zeros = Minkowski::<3>::default();
        for site in self.sites() {
            let rho = (-1.0 * site.properties.position.distance_squared(&zeros)).sqrt();
            let point = Cartesian::from([
                site.properties.position[0] * rho / site.properties.position[2],
                site.properties.position[1] * rho / site.properties.position[2]]);
            let point_norm = point.norm_squared();
            circle_centers.push(
                Cartesian::from([
                    point.coordinates[0] / (2.0*(1.0-point_norm).sqrt()),
                    point.coordinates[1] / (2.0*(1.0-point_norm).sqrt()),
                    0.0,]));
            circle_radii.push(point_norm / (4.0*(1.0-point_norm)) - 1.0/((1.0-point_norm).sqrt()));
            circle_tags.push(site.site_tag);
        }
        PowerDiagramCenters{
            centers : circle_centers,
            radii : circle_radii,
            site_tags : circle_tags,
        }
    }
}

/** Define the neighbor list
TODO: documentation
*/
#[derive(Clone, Debug, PartialEq)]
pub struct NeighborList {
    /// ordered, nested vector of 2-tuples with nearest-neighbor pairs
    pub neighbors: Vec<(u32,u32)>
}

impl NeighborList {
    #[inline]
    pub fn nearest_neighbors(power_diagram: PowerDiagramCenters) -> NeighborList {
        NeighborList{neighbors : vec![(1,1)]}
    }
}

