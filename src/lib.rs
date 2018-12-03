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
//! * Think about `ModCholesky` struct
//! * Be generic over element type (f64/f32/complex32/complex64)
//!
//! # References
//!
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

mod gershgorin;
mod se90;
mod utils;

pub use crate::gershgorin::*;
pub use crate::se90::ModCholeskySE90;

#[cfg(test)]
mod tests {
    // use super::*;
    // #[test]
    // fn test_modified_cholesky() {
    //     use super::ModifiedCholesky;
    //     let a: ndarray::Array2<f64> =
    //         ndarray::arr2(&[[4.0, 2.0, 1.0], [2.0, 6.0, 3.0], [1.0, 3.0, -0.004]]);
    //     let (l, d, _) = a.modified_cholesky().unwrap();
    //     let f = l.dot(&d).dot(&(l.t()));
    //     let res: ndarray::Array2<f64> =
    //         ndarray::arr2(&[[4.0, 2.0, 1.0], [2.0, 6.0, 3.0], [1.0, 3.0, 3.004]]);
    //     assert!(f.all_close(&res, 2.0 * std::f64::EPSILON));
    //     // let dsqrt = d.map(|x| x.sqrt());
    //     // let m = l.dot(&dsqrt);
    //     // println!("l: {:?}", l);
    //     // println!("d: {:?}", d);
    //     // println!("f: {:?}", f);
    //     // println!("m: {:?}", m);
    // }
}
