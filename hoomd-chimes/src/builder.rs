// Copyright (c) 2024-2025 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.
/*!
Builder for constructing `ChIMES` potential from
the `ChIMES` parameter file.
 */
use std::error::Error;
use std::{fmt, fs};

use crate::potential::{
    ChimesChebyshevExpansion, ChimesPenalty, ChimesSmoothing, ChimesTransformation,
    ChimesTwobPotential, CubicSmooth, TersoffSmooth,
};
use crate::transformation::{DirectTransformation, InverseTransformation, MorseTransformation};
use hoomd_interaction::{PairwiseCutoff, pairwise::Isotropic};

/// Custom error for invalid format or data
/// found in the parameter file.
#[derive(Debug)]
struct InvalidFormatError(String);

impl fmt::Display for InvalidFormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for InvalidFormatError {}

/// Custom error for not able to
/// construct potential from
/// the parsed data.
#[derive(Debug)]
struct PotentialConstructionError(String);

impl fmt::Display for PotentialConstructionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Error for PotentialConstructionError {}

/**
Builder for constructing `ChIMES` potential from
the `ChIMES` parameter file.

Given a known maximum two-body Chebyshev polynomial
orders `N`, [`ChimesBuilder`] can be used to parse the
potential parameter in the `ChIMES` parameter TXT file,
as described in the Generating a `ChIMES` model
writen in the [ChIMES-LSQ].

The `parse` function perform the TXT file parsing
given a `file_path`, pointing to the TXT file. The
current implementation only parse out the two-body
potential component.

Given a valid `pair_idx`, representing one of
the particle pair type recognized by [`ChimesBuilder`] during
the excution of `parse`. The `get_twob_chimes_potential` assemble the complete
`ChIMES` potential functional, wrapped in [`PairwiseCutoff`]
, and return it.

[ChIMES-LSQ]: <https://chimes-lsq.readthedocs.io/en/latest/lsq_input_file.html>


# Example
```
use hoomd_chimes::parser::ChimesBuilder;

// Maximum two-body order is 12 for the example parameter file.
const N: usize = 12;
let file_path = "./test-data/C-twobody.txt";

let chimes_builder = ChimesBuilder::<N>::parse(file_path).expect("Failed to parse parameter file");
let chimes_fn = chimes_builder.get_twob_chimes_potential(0).expect("Error assemling ChIMES potential");
assert_eq!(chimes_fn.0.interaction.type1, String::from("C"));
```
*/
#[derive(Clone, Debug, PartialEq)]
pub struct ChimesBuilder<const N: usize> {
    /// Chebyshev polynomial orders and related parameters from PAIRTYP.
    pub poly_order: Vec<usize>,
    /// Number of atom types.
    pub atom_types: usize,
    /// A tuple stores parameters of each particle type.
    /// The elements of the tuple stores parameters as follows:
    /// (particle types, masses)
    pub type_data: (Vec<String>, Vec<f64>),
    /// Number of pair types.
    pub atom_pair_types: usize,
    /// Distance transformation style, assuming the same for all pair types.
    pub xform_style: String,
    /// A tuple with length of `atom_pair_types`
    /// stores pairwise `ChIMES` parameters of each pair type.
    /// Each element of the tuple stores the parameters as follows:
    /// (first particle types, second particle types,
    /// inner radial cutoffs, outer radial cutofsf, morse lambdas).
    pub pair_data: (
        Vec<String>,
        Vec<String>,
        Vec<f64>,
        Vec<f64>,
        Vec<Option<f64>>,
    ),
    /// FCUT type and value, assuming the same for all pair types.
    /// See [`ChimesSmoothing`].
    pub fcut: (String, Option<f64>),
    /// Indexes represent each pair type.
    pub pair_type_index: Vec<usize>,
    /// A vector stores Chebyshev polynomial coefficient of each
    /// pair type. See [`Chimes2b`].
    pub cheby_2b_coeffs: Vec<Vec<f64>>,
    /// A vector contains indexes of
    /// pair types slow mapping.
    pub pair_idx_slow_map: Vec<usize>,
    /// A vector contains string of
    /// pair types for the corresponding
    /// `pair_idx_slow_map`.
    pub pair_type_slow_map: Vec<String>,
    /// A vector contains indexes of
    /// pair types fast mapping.
    pub pair_idx_fast_map: Vec<usize>,
    /// A vector contains string of
    /// pair types for the corresponding
    /// `pair_idx_fast_map`.
    pub pair_type_fast_map: Vec<String>,
    /// Single particle energy for each particle type.
    pub energy_offset: Vec<f64>,
    /// The smooth kick-in distance of [`ChimesPenalty`].
    pub penalty_dist: f64,
    /// The penalty strength of [`ChimesPenalty`].
    pub penalty_scaling: f64,
}

impl<const N: usize> ChimesBuilder<N> {
    /// parse the `ChIMES` parameter file given the file path.
    ///
    ///
    /// # Errors
    ///
    /// Will return `Err` if the parser
    /// cannot read the parameter file .
    ///
    /// Will return `Err` if `N` does not
    /// match the two-body order read in the
    /// parameter file.
    #[allow(clippy::too_many_lines, reason = "Parse complex TXT file")]
    #[allow(
        clippy::cast_sign_loss,
        reason = "Parse the line with positive and negative number but use it as index"
    )]
    #[inline]
    pub fn parse(file_path: &str) -> Result<Self, Box<dyn Error>> {
        let content = fs::read_to_string(file_path)?;
        let lines: Vec<&str> = content
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('!') && !line.starts_with("##"))
            .collect();

        // Extract PAIRTYP by iterate through each line, break the loop when it's found
        let mut poly_order: Vec<usize> = Vec::new();
        for line in &lines {
            if line.starts_with("PAIRTYP:") {
                let pairtyp = Self::parse_i32_vec(line.trim_start_matches("PAIRTYP: CHEBYSHEV "))?;

                if pairtyp.len() < 3 {
                    return Err(Box::new(InvalidFormatError(
                        "PAIRTYP: CHEBYSHEV must contain at least  contain the 2-body order".into(),
                    )));
                }

                poly_order.push(pairtyp[0] as usize);

                if pairtyp.len() >= 4 {
                    poly_order.push(pairtyp[1] as usize);
                }

                if pairtyp.len() >= 5 {
                    poly_order.push(pairtyp[2] as usize);
                }
                break;
            }
        }
        if poly_order.is_empty() {
            return Err(Box::new(InvalidFormatError(
                "Missing PAIRTYP: CHEBYSHEV line".into(),
            )));
        }
        if N != poly_order[0] {
            return Err(Box::new(InvalidFormatError(format!(
                "Mismatch two-body order found in the parameter file = {}, assuming two-body order N={}",
                poly_order[0], N
            ))));
        }

        // Extract the particle types and particle pair types related chimes parameters
        let mut atom_types: usize = 0;
        let mut type_data: (Vec<String>, Vec<f64>) = (Vec::new(), Vec::new()); // Atom type, mass
        let mut pair_data: (
            Vec<String>,
            Vec<String>,
            Vec<f64>,
            Vec<f64>,
            Vec<Option<f64>>,
        ) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()); // type_1, type_2, inner, outer, morse lambda
        let mut xform_style = String::new();
        let mut atom_pair_types: usize = 0;
        let mut fcut = (String::new(), None);
        let mut energy_offset: Vec<f64> = Vec::new();
        let mut penalty_dist = 0.01;
        let mut penalty_scaling = 1e+4;
        for (idx, line) in lines.iter().enumerate() {
            if line.starts_with("ATOM TYPES: ") {
                atom_types = line
                    .trim_start_matches("ATOM TYPES: ")
                    .parse::<usize>()
                    .map_err(|e| Box::new(e) as Box<dyn Error>)?;
                energy_offset = vec![0.0; atom_types];
            }

            if line.starts_with("# TYPEIDX #") {
                let start = idx + 1;
                let end = start + atom_types;

                for line_after_start in &lines[start..end] {
                    let total_type_data: Vec<&str> = line_after_start.split_whitespace().collect();
                    type_data.0.push(total_type_data[1].to_string());
                    type_data.1.push(
                        total_type_data[3]
                            .parse::<f64>()
                            .map_err(|e| Box::new(e) as Box<dyn Error>)?,
                    );
                }
            }

            if line.starts_with("ATOM PAIRS: ") {
                atom_pair_types = line
                    .trim_start_matches("ATOM PAIRS: ")
                    .parse::<usize>()
                    .map_err(|e| Box::new(e) as Box<dyn Error>)?;
            }

            if line.starts_with("# PAIRIDX #") {
                if line.contains("# USEPOVR #") {
                    continue;
                }
                let start = idx + 1;
                let end = start + atom_pair_types;

                let mut tmp_xform_style = String::new();
                for (i, line_after_start) in lines[start..end].iter().enumerate() {
                    let total_pair_type_data: Vec<&str> =
                        line_after_start.split_whitespace().collect();

                    let (xform_style_idx, morse_idx): (usize, usize) =
                        match total_pair_type_data.len() {
                            8 => (6, 7),
                            7 => (5, 6),
                            _ => {
                                return Err(Box::new(InvalidFormatError(format!(
                                    "Incorrect input at the line {} \nExpect 7 or 8 entries\n",
                                    start + i + 1
                                ))));
                            }
                        };

                    pair_data.0.push(total_pair_type_data[1].to_string());
                    pair_data.1.push(total_pair_type_data[2].to_string());
                    pair_data.2.push(
                        total_pair_type_data[3]
                            .parse::<f64>()
                            .map_err(|e| Box::new(e) as Box<dyn Error>)?,
                    );
                    pair_data.3.push(
                        total_pair_type_data[4]
                            .parse::<f64>()
                            .map_err(|e| Box::new(e) as Box<dyn Error>)?,
                    );

                    if i == 0 {
                        tmp_xform_style = total_pair_type_data[xform_style_idx].to_string();
                    } else if total_pair_type_data[xform_style_idx] != tmp_xform_style {
                        return Err(Box::new(InvalidFormatError(
                            "Distance transformation style must be the same for all pair types"
                                .into(),
                        )));
                    }

                    xform_style.clone_from(&tmp_xform_style);

                    if tmp_xform_style == "MORSE" && total_pair_type_data.len() > morse_idx {
                        pair_data.4.push(Some(
                            total_pair_type_data[morse_idx]
                                .parse::<f64>()
                                .map_err(|e| Box::new(e) as Box<dyn Error>)?,
                        ));
                    } else {
                        return Err(Box::new(InvalidFormatError(format!(
                            "Missing morse lambda value at line {}",
                            start + i + 1
                        ))));
                    }
                }
            }

            if line.starts_with("FCUT TYPE: ") {
                let fcut_line: Vec<&str> = line
                    .trim_start_matches("FCUT TYPE: ")
                    .split_whitespace()
                    .collect();

                let fcut_style = fcut_line[0].to_string();
                if fcut_style != "CUBIC" && fcut_style != "TERSOFF" {
                    return Err(Box::new(InvalidFormatError(
                        "Error: unknown FCUT TYPE".into(),
                    )));
                }
                fcut.0 = fcut_style;
                if fcut_line.len() > 1 {
                    fcut.1 = Some(
                        fcut_line[1]
                            .parse::<f64>()
                            .map_err(|e| Box::new(e) as Box<dyn Error>)?,
                    );
                }
            }

            if line.starts_with("PAIR CHEBYSHEV PENALTY DIST: ") {
                let panelty_dist_line: Vec<&str> = line
                    .trim_start_matches("PAIR CHEBYSHEV PENALTY DIST: ")
                    .split_whitespace()
                    .collect();

                penalty_dist = panelty_dist_line[0]
                    .parse::<f64>()
                    .map_err(|e| Box::new(e) as Box<dyn Error>)?;
            }

            if line.starts_with("PAIR CHEBYSHEV PENALTY SCALING: ") {
                let panelty_scaling_line: Vec<&str> = line
                    .trim_start_matches("PAIR CHEBYSHEV PENALTY SCALING: ")
                    .split_whitespace()
                    .collect();
                penalty_scaling = panelty_scaling_line[0]
                    .parse::<f64>()
                    .map_err(|e| Box::new(e) as Box<dyn Error>)?;
            }

            if line.starts_with("NO ENERGY OFFSETS: ") {
                let n_energy_offset_line: Vec<&str> = line
                    .trim_start_matches("NO ENERGY OFFSETS: ")
                    .split_whitespace()
                    .collect();

                let n_offset = n_energy_offset_line[0]
                    .parse::<usize>()
                    .map_err(|e| Box::new(e) as Box<dyn Error>)?;

                if n_offset != atom_types {
                    return Err(Box::new(InvalidFormatError(
                        "ERROR: Number of energy offsets do not match number of atom types".into(),
                    )));
                }

                let start = idx + 1;
                let end = start + atom_types;

                for (i, line_after_start) in lines[start..end].iter().enumerate() {
                    let energy_offset_data: Vec<&str> =
                        line_after_start.split_whitespace().collect();
                    energy_offset[i] = energy_offset_data[3]
                        .parse::<f64>()
                        .map_err(|e| Box::new(e) as Box<dyn Error>)?;
                }

                break;
            }
        }

        if atom_types == 0 {
            return Err(Box::new(InvalidFormatError(
                "Missing ATOM TYPES line".into(),
            )));
        }

        // Extract the two-body coefficient and interaction
        // topology mapping.
        let mut pair_type_index: Vec<usize> = Vec::new();
        let mut cheby_2b_coeffs: Vec<Vec<f64>> = Vec::new();
        let mut pair_idx_slow_map: Vec<usize> = Vec::new();
        let mut pair_type_slow_map: Vec<String> = Vec::new();
        let mut pair_idx_fast_map: Vec<usize> = Vec::new();
        let mut pair_type_fast_map: Vec<String> = Vec::new();
        for (idx, line) in lines.iter().enumerate() {
            if line.starts_with("PAIRTYPE PARAMS: ") {
                let pair_params_head_line: Vec<&str> = line
                    .trim_start_matches("PAIRTYPE PARAMS: ")
                    .split_whitespace()
                    .collect();
                let tmp_pair_type_index = pair_params_head_line[0]
                    .parse::<usize>()
                    .map_err(|e| Box::new(e) as Box<dyn Error>)?;
                pair_type_index.push(tmp_pair_type_index);

                let start = idx + 1;
                let end = start + poly_order[0];

                let mut tmp_2b_coeff: Vec<f64> = Vec::new();
                for line_after_start in &lines[start..end] {
                    let order_coeff = Self::parse_f64_vec(line_after_start)?;
                    tmp_2b_coeff.push(order_coeff[1]);
                }
                cheby_2b_coeffs.push(tmp_2b_coeff);
            }

            if line.starts_with("PAIRMAPS: ") {
                let pair_map_header: Vec<&str> = line
                    .trim_start_matches("PAIRMAPS: ")
                    .split_whitespace()
                    .collect();
                let n_pair_maps = pair_map_header[0]
                    .parse::<usize>()
                    .map_err(|e| Box::new(e) as Box<dyn Error>)?;

                let start = idx + 1;
                let end = start + n_pair_maps;
                for line_after_start in &lines[start..end] {
                    let pair_idx_type: Vec<&str> = line_after_start.split_whitespace().collect();
                    let tmp_pair_idx = pair_idx_type[0]
                        .parse::<usize>()
                        .map_err(|e| Box::new(e) as Box<dyn Error>)?;
                    pair_idx_slow_map.push(tmp_pair_idx);
                    pair_type_slow_map.push(pair_idx_type[1].to_string());
                }

                for ii in 0..atom_types {
                    for jj in 0..atom_types {
                        let typei = type_data.0[ii].clone();
                        let typej = type_data.0[jj].clone();
                        let typeij = typei + &typej;

                        //let index_of_pair_match = pair_type_slow_map
                        //    .iter()
                        //    .position(|s| s.contains(&typeij))
                        //    .expect("Error finding pair type from the slow map");
                        // idiomatically propogate the error, instead of unwrap which
                        // may cause panics
                        let index_of_pair_match = pair_type_slow_map
                            .iter()
                            .position(|s| *s == typeij)
                            .ok_or_else(|| {
                                InvalidFormatError(format!(
                                    "Error reading pair type slow mapping at line {start}"
                                ))
                            })?;
                        pair_idx_fast_map.push(pair_idx_slow_map[index_of_pair_match]);

                        for iii in 0..pair_data.0.len() {
                            if (type_data.0[ii] == pair_data.0[iii]
                                && type_data.0[jj] == pair_data.1[iii])
                                || (type_data.0[jj] == pair_data.0[iii]
                                    && type_data.0[ii] == pair_data.1[iii])
                            {
                                pair_type_fast_map
                                    .push(pair_data.0[iii].clone() + &pair_data.1[iii]);
                                break;
                            }
                        }
                    }
                }
                break;
            }
        }

        Ok(ChimesBuilder {
            poly_order,
            atom_types,
            type_data,
            atom_pair_types,
            xform_style,
            pair_data,
            fcut,
            pair_type_index,
            cheby_2b_coeffs,
            pair_idx_slow_map,
            pair_type_slow_map,
            pair_idx_fast_map,
            pair_type_fast_map,
            energy_offset,
            penalty_dist,
            penalty_scaling,
        })
    }

    /// Assemble two-body `ChIMES` potential function given
    /// a `pair_idx`.
    ///
    /// # Errors
    ///
    /// Will return `Err` if the `pair_idx` provided
    /// do not exist in the parameter file.
    #[inline]
    pub fn get_twob_chimes_potential(
        &self,
        pair_idx: usize,
    ) -> Result<PairwiseCutoff<Isotropic<ChimesTwobPotential<N>>>, Box<dyn Error>> {
        if pair_idx >= self.pair_data.0.len() {
            return Err(Box::new(PotentialConstructionError(format!(
                "Intend to access the potential model with pair idx {}, but only found {} pairs",
                pair_idx,
                self.pair_data.0.len()
            ))));
        }
        let transformatiom_fn = self.get_tranformation(pair_idx)?;
        let cheby2b: ChimesChebyshevExpansion<ChimesTransformation, N> =
            ChimesChebyshevExpansion::new(
                transformatiom_fn,
                self.cheby_2b_coeffs[pair_idx].clone(),
                self.pair_data.2[pair_idx],
            );
        let chimes_2b_model = self.get_smoothing(cheby2b, pair_idx)?;
        let penalty_fn = ChimesPenalty {
            r_in: self.pair_data.2[pair_idx],
            a: self.penalty_scaling,
            dt: self.penalty_dist,
        };

        let chimes_potential = ChimesTwobPotential {
            type1: self.pair_data.0[pair_idx].clone(),
            type2: self.pair_data.1[pair_idx].clone(),
            chimes: chimes_2b_model,
            penalty: penalty_fn
        };

        Ok(PairwiseCutoff(Isotropic {
            interaction: chimes_potential,
            r_cut: self.pair_data.3[pair_idx],
        }))
    }

    /// Assemble transformation function.
    fn get_tranformation(&self, pair_idx: usize) -> Result<ChimesTransformation, Box<dyn Error>> {
        match self.xform_style.as_str() {
            "MORSE" => Ok(ChimesTransformation::Morse(MorseTransformation {
                lambda: self.pair_data.4[pair_idx].expect("Error reading morse lambda"),
                r_in: self.pair_data.2[pair_idx],
                r_out: self.pair_data.3[pair_idx],
            })),
            "INVERSE" => Ok(ChimesTransformation::Inverse(InverseTransformation {
                r_in: self.pair_data.2[pair_idx],
                r_out: self.pair_data.3[pair_idx],
            })),
            "DIRECT" => Ok(ChimesTransformation::Direct(DirectTransformation {
                r_in: self.pair_data.2[pair_idx],
                r_out: self.pair_data.3[pair_idx],
            })),
            _ => Err(Box::new(PotentialConstructionError(format!(
                "Unknown transformation style: {}",
                self.xform_style
            )))),
        }
    }

    /// Assemble smoothing function.
    fn get_smoothing(
        &self,
        f: ChimesChebyshevExpansion<ChimesTransformation, N>,
        pair_idx: usize,
    ) -> Result<ChimesSmoothing<ChimesTransformation, N>, Box<dyn Error>> {
        match self.fcut.0.as_str() {
            "CUBIC" => Ok(ChimesSmoothing::Cubic(CubicSmooth {
                f,
                r_out: self.pair_data.3[pair_idx],
            })),
            "TERSOFF" => Ok(ChimesSmoothing::Tersoff(TersoffSmooth {
                f,
                r_out: self.pair_data.3[pair_idx],
                r_in: self.pair_data.2[pair_idx],
                fo: self.fcut.1.expect("Error reading tersoff fo"),
            })),
            _ => Err(Box::new(PotentialConstructionError(format!(
                "Unknown smoothing style: {}",
                self.fcut.0
            )))),
        }
    }
    /// Parses a space-separated line into a vector of i32.
    fn parse_i32_vec(line: &str) -> Result<Vec<i32>, Box<dyn Error>> {
        let values = line
            .split_whitespace()
            .map(|s| s.parse::<i32>().map_err(|e| Box::new(e) as Box<dyn Error>))
            .collect::<Result<Vec<i32>, _>>()?;
        Ok(values)
    }

    /// Parses a space-separated line into a vector of f64.
    fn parse_f64_vec(line: &str) -> Result<Vec<f64>, Box<dyn Error>> {
        let values = line
            .split_whitespace()
            .map(|s| s.parse::<f64>().map_err(|e| Box::new(e) as Box<dyn Error>))
            .collect::<Result<Vec<f64>, _>>()?;
        Ok(values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::*;

    #[rstest]
    fn parse_carbon_two_body() {
        const N: usize = 12;
        let file_path = "./test-data/C-twobody.txt";

        // Read the entire file content into a String
        // This returns a Result, so we use `?` to propagate any errors
        let params = ChimesBuilder::<N>::parse(file_path).expect("Failed to parse parameter file");

        let expected_poly_order = vec![12, 0, 0];
        let expected_type_data = (vec![String::from("C")], vec![12.011]);
        let expected_xform_style = String::from("MORSE");
        let expected_pair_data = (
            vec![String::from("C")],
            vec![String::from("C")],
            vec![1.0],
            vec![3.15],
            vec![Some(1.25)],
        );
        let expected_fcut: (String, Option<f64>) = (String::from("CUBIC"), None);
        let expected_energy_offset = vec![0.0];
        let expected_penalty_dist = 0.01;
        let expected_penalty_scaling = 1e+8;
        let expected_pair_type_index = [0];
        let expected_cheby_2b_coeffs = vec![vec![
            285.738_833_080_72,
            -213.713_887_523_72,
            358.533_310_990_31,
            -172.124_004_865_49,
            44.775_023_503_15,
            -34.154_784_921_509,
            30.632_345_544_482,
            -33.336_059_893_072,
            11.483_163_813_684,
            -0.990_867_207_911_8,
            -3.383_013_890_418_8,
            1.248_010_862_845_3,
        ]];
        let expected_pair_idx_slow_map = vec![0];
        let expected_pair_type_slow_map = vec!["CC"];
        let expected_pair_idx_fast_map = vec![0];
        let expected_pair_type_fast_map = vec!["CC"];

        assert_eq!(params.poly_order, expected_poly_order);
        assert_eq!(params.type_data, expected_type_data);
        assert_eq!(params.xform_style, expected_xform_style);
        assert_eq!(params.pair_data, expected_pair_data);
        assert_eq!(params.fcut, expected_fcut);
        assert_eq!(params.energy_offset, expected_energy_offset);
        assert_eq!(params.penalty_dist, expected_penalty_dist);
        assert_eq!(params.penalty_scaling, expected_penalty_scaling);
        assert_eq!(params.pair_type_index, expected_pair_type_index);
        assert_eq!(params.cheby_2b_coeffs, expected_cheby_2b_coeffs);
        assert_eq!(params.pair_idx_slow_map, expected_pair_idx_slow_map);
        assert_eq!(params.pair_type_slow_map, expected_pair_type_slow_map);
        assert_eq!(params.pair_idx_fast_map, expected_pair_idx_fast_map);
        assert_eq!(params.pair_type_fast_map, expected_pair_type_fast_map);

        let chimes_pot = params
            .get_twob_chimes_potential(0)
            .expect("Error assembling ChIMES potential");
        assert_eq!(chimes_pot.0.interaction.type1, String::from("C"));
        assert_eq!(chimes_pot.0.interaction.type2, String::from("C"));
    }

    #[rstest]
    fn parse_azoimide_twobodypart() {
        const N: usize = 12;
        let file_path = "./test-data/HN-fourbody.txt";

        let params = ChimesBuilder::<N>::parse(file_path).expect("Failed to parse parameter file");

        let expected_poly_order = vec![12, 8, 4];
        let expected_type_data = (
            vec![String::from("N"), String::from("H")],
            vec![14.0064, 1.0079],
        );
        let expected_xform_style = String::from("MORSE");
        let expected_pair_data = (
            vec![String::from("N"), String::from("H"), String::from("N")],
            vec![String::from("N"), String::from("H"), String::from("H")],
            vec![0.793, 0.451, 0.666],
            vec![8.0, 8.0, 8.0],
            vec![Some(1.15), Some(0.8), Some(1.0)],
        );
        let expected_fcut: (String, Option<f64>) = (String::from("TERSOFF"), Some(0.5));
        let expected_energy_offset = vec![-126.828700616, -59.0402284083];
        let expected_penalty_dist = 0.02;
        let expected_penalty_scaling = 1e+6;
        let expected_pair_type_index = [0, 1, 2];
        let expected_cheby_2b_coeffs = vec![
            vec![
                17.341_009_355_697_139,
                57.774_119_773_766_508,
                76.220_068_702_688_152,
                40.597_946_713_780_949,
                -3.090_950_293_142_406_7,
                -9.568_323_288_481_783_7,
                -1.398_446_509_119_888_8,
                -0.284_883_283_536_185_81,
                -0.217_319_593_051_684_55,
                0.156_890_951_659_628_66,
                0.056_716_863_392_651_549,
                0.214_859_981_309_824_75,
            ],
            vec![
                12.505_788_696_001_36,
                28.829_120_708_867_389,
                24.295_927_978_434_74,
                9.351_183_377_623_707_2,
                0.809_540_068_944_617_27,
                1.108_761_409_653_912_5,
                2.308_831_039_570_009_6,
                -0.171_536_478_297_497_44,
                0.035_436_599_340_995_96,
                0.501_230_272_711_822_86,
                -0.056_263_800_428_875_528,
                0.203_871_981_318_344_89,
            ],
            vec![
                1.701_481_886_171_263_5,
                10.542_458_855_813_658,
                13.624_902_173_816_118,
                -1.951_617_500_501_013_2,
                -11.233_817_110_914_156,
                -4.308_217_478_930_769_7,
                -0.189_647_654_903_229_32,
                -0.921_076_193_724_393_2,
                -0.159_350_209_186_874_81,
                0.267_439_675_848_717_86,
                0.034_690_942_326_356_888,
                0.002_082_396_054_432_101_4,
            ],
        ];
        let expected_pair_idx_slow_map = vec![1, 2, 2, 0];
        let expected_pair_type_slow_map = vec!["HH", "HN", "NH", "NN"];
        let expected_pair_idx_fast_map = vec![0, 2, 2, 1];
        let expected_pair_type_fast_map = vec!["NN", "NH", "NH", "HH"];

        assert_eq!(params.poly_order, expected_poly_order);
        assert_eq!(params.type_data, expected_type_data);
        assert_eq!(params.xform_style, expected_xform_style);
        assert_eq!(params.pair_data, expected_pair_data);
        assert_eq!(params.fcut, expected_fcut);
        assert_eq!(params.energy_offset, expected_energy_offset);
        assert_eq!(params.penalty_dist, expected_penalty_dist);
        assert_eq!(params.penalty_scaling, expected_penalty_scaling);
        assert_eq!(params.pair_type_index, expected_pair_type_index);
        assert_eq!(params.cheby_2b_coeffs, expected_cheby_2b_coeffs);
        assert_eq!(params.pair_idx_slow_map, expected_pair_idx_slow_map);
        assert_eq!(params.pair_type_slow_map, expected_pair_type_slow_map);
        assert_eq!(params.pair_idx_fast_map, expected_pair_idx_fast_map);
        assert_eq!(params.pair_type_fast_map, expected_pair_type_fast_map);

        let nn_pot = params
            .get_twob_chimes_potential(0)
            .expect("Error assembling ChIMES potential");
        let nh_pot = params
            .get_twob_chimes_potential(1)
            .expect("Error assembling ChIMES potential");
        let hh_pot = params
            .get_twob_chimes_potential(2)
            .expect("Error assembling ChIMES potential");
        assert_eq!(nn_pot.0.interaction.type1, String::from("N"));
        assert_eq!(nn_pot.0.interaction.type2, String::from("N"));
        assert_eq!(nh_pot.0.interaction.type1, String::from("H"));
        assert_eq!(nh_pot.0.interaction.type2, String::from("H"));
        assert_eq!(hh_pot.0.interaction.type1, String::from("N"));
        assert_eq!(hh_pot.0.interaction.type2, String::from("H"));
    }

    #[rstest]
    #[should_panic]
    // Only find one pair type 
    // but intend to access the pair type with index 1.
    fn invalid_pair_index() {
        const N: usize = 12;
        let file_path = "./test-data/C-twobody.txt";

        // Read the entire file content into a String
        // This returns a Result, so we use `?` to propagate any errors
        let params = ChimesBuilder::<N>::parse(file_path).expect("Failed to parse parameter file");

        let _ = params.get_twob_chimes_potential(1).expect("Error assembling ChIMES potential");
    }
}
