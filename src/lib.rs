// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Modified Cholesky decompositions
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

mod gershgorin;
mod gmw81;
mod se90;
mod se99;
mod utils;

pub use crate::gershgorin::*;
pub use crate::gmw81::ModCholeskyGMW81;
pub use crate::se90::ModCholeskySE90;
pub use crate::se99::ModCholeskySE99;
