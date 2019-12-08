// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Modified Cholesky decompositions
//!
//! Given a symmetric matrix A which is potentially not positive definite, a modified Cholesky
//! algorithm obtains the Cholesky decomposition `LL^T` of the positive definite matrix
//! `P(A + E)P^T` where `E` is symmetric and `>= 0`, `P` is a permutation matrix and `L` is lower
//! triangular.
//! If `A` is already positive definite, then `E = 0`.
//! The perturbation `E` should be as small as possible for `A + E` to be "sufficiently positive
//! definite".
//! This is used in optimization methods where indefinite Hessians can be problematic.
//!
//! This crate implements the algorithms by Gill, Murray and Wright
//! ([GMW81](trait.ModCholeskyGMW81.html)) and Schnabel and Eskow
//! ([SE90](trait.ModCholeskySE90.html) and [SE99](trait.ModCholeskySE99.html)).
//! All algorithms are currently based on `ndarray` but will also be implemented for `nalgebra` in
//! the future.
//!
//! # Example
//!
//! ```rust
//! # extern crate openblas_src;
//! # use modcholesky::utils::{diag_mat_from_arr, index_to_permutation_mat};
//! use modcholesky::ModCholeskySE99;
//!
//! let a = ndarray::arr2(&[[1.0, 1.0, 2.0],
//!                         [1.0, 1.0, 3.0],
//!                         [2.0, 3.0, 1.0]]);
//!
//! // Perform modified Cholesky decomposition
//! // The `Decomposition` struct holds L, E and P
//! let decomp = a.mod_cholesky_se99();
//!
//! println!("L:\n{:?}", decomp.l);
//! println!("E:\n{:?}", decomp.e);
//! println!("P:\n{:?}", decomp.p);
//!
//! # let l = decomp.l;
//! #
//! # let res_l: ndarray::Array2<f64> = ndarray::arr2(&[
//! #     [1.732050807568877, 0.0, 0.0],
//! #     [0.5773502691896257, 1.698920954907997, 0.0],
//! #     [1.154700538379251, 1.37342077428181, 0.006912871809428971],
//! # ]);
//! # let res = res_l.dot(&res_l.t());
//! #
//! # assert!(res_l.all_close(&l, 1e-12));
//! #
//! # let e = diag_mat_from_arr(decomp.e.as_slice().unwrap());
//! # let p = index_to_permutation_mat(decomp.p.as_slice().unwrap());
//! # let paptpept = p.dot(&a.dot(&p.t())) + p.dot(&e.dot(&p.t()));
//! # assert!(paptpept.all_close(&l.dot(&l.t()), 1e-12));
//! # assert!(l.dot(&(l.t())).all_close(&res, 1e-12));
//! ```
//!
//! # TODOs
//!
//! * Be generic over element type (f64/f32/complex32/complex64)
//! * Implement algorithms for nalgebra types
//!
//! # References
//!
//! * Philip E. Gill, Walter Murray and Margaret H. Wright.
//!   Practical Optimization.
//!   Emerald Group Publishing Limited. ISBN 978-0122839528. 1982
//! * Semyon Aranovich Gershgorin.
//!   Über die Abgrenzung der Eigenwerte einer Matrix.
//!   Izv. Akad. Nauk. USSR Otd. Fiz.-Mat. Nauk, 6: 749–754, 1931.
//! * Robert B. Schnabel and Elizabeth Eskow.
//!   A new modified Cholesky factorization.
//!   SIAM J. Sci. Stat. Comput. Vol. 11, No. 6, pp. 1136-1158, November 1990
//! * Elizabeth Eskow and Robert B. Schnabel.
//!   Algorithm 695: Software for a new modified Cholesky factorization.
//!   ACM Trans. Math. Softw. Vol. 17, p. 306-312, 1991
//! * Sheung Hun Cheng and Nicholas J. Higham.
//!   A modified Cholesky algorithm based on a symmetric indefinite factorization.
//!   SIAM J. Matrix Anal. Appl. Vol. 19, No. 4, pp. 1097-1110, October 1998
//! * Robert B. Schnabel and Elizabeth Eskow.
//!   A revised modified Cholesky factorization.
//!   SIAM J. Optim. Vol. 9, No. 4, pp. 1135-1148, 1999
//! * Jorge Nocedal and Stephen J. Wright.
//!   Numerical Optimization.
//!   Springer. ISBN 0-387-30303-0. 2006.

// necessary to get clippy to shut up about the s! macro
#![allow(clippy::deref_addrof)]
// I really do not like the a..=b syntax
#![allow(clippy::range_plus_one)]

#[cfg(test)]
extern crate openblas_src;

mod gershgorin;
mod gmw81;
mod se90;
mod se99;
pub mod utils;

pub use crate::gershgorin::*;
pub use crate::gmw81::ModCholeskyGMW81;
pub use crate::se90::ModCholeskySE90;
pub use crate::se99::ModCholeskySE99;

pub struct Decomposition<L, E, P> {
    pub l: L,
    pub e: E,
    pub p: P,
}

impl<L, E, P> Decomposition<L, E, P> {
    pub fn new(l: L, e: E, p: P) -> Self {
        Decomposition { l, e, p }
    }
}
