// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Schnabel & Eskow algorithm (1990)
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

use crate::utils::{eigenvalues_2x2, index_of_largest, swap_columns, swap_rows};
use failure::{bail, Error};

pub trait ModCholeskySE90 {
    fn mod_cholesky_se90(&mut self) -> Result<(), Error> {
        bail!("Not implemented!")
    }
}

impl ModCholeskySE90 for ndarray::Array2<f64> {
    fn mod_cholesky_se90(&mut self) -> Result<(), Error> {
        assert!(self.is_square());
        use ndarray::s;

        // cbrt = cubic root
        let tau = std::f64::EPSILON.cbrt();

        let n = self.raw_dim()[0];

        let mut phaseone = true;

        let gamma = self
            .diag()
            .fold(0.0, |acc, x| if x.abs() > acc { x.abs() } else { acc });

        let mut j = 0;
        // Phase one, `self` potentially positive-definite
        while j < n && phaseone {
            // Pivot on maximum diagonal of remaining submatrix
            let max_idx = index_of_largest(&self.diag().slice(s![j..]));
            if max_idx != j {
                swap_rows(self, j, j + max_idx);
                swap_columns(self, j, j + max_idx);
            }
            let tmp = (j + 1..n).fold(std::f64::INFINITY, |acc, i| {
                let nv = self[(i, i)] - self[(i, j)].powi(2) / self[(j, j)];
                if nv < acc {
                    nv
                } else {
                    acc
                }
            });
            if tmp < tau * gamma {
                // Go to phase two
                phaseone = false;
                break;
            } else {
                // Perform jth iteration of factorization
                self[(j, j)] = self[(j, j)].sqrt();
                for i in (j + 1)..n {
                    self[(i, j)] /= self[(j, j)];
                    for k in (j + 1)..(i + 1) {
                        self[(i, k)] -= self[(i, j)] * self[(k, j)];
                    }
                }
                j += 1;
            }
        }

        let mut delta;
        let mut delta_prev = 0.0;

        // Phase two, `self` not positive-definite
        if !phaseone {
            let k = j;

            // Calculate lower Gershgorin bounds of self_{k+1}
            let mut g = ndarray::Array::zeros(n);
            for i in k..n {
                g[i] = self[(i, i)]
                    - self.slice(s![i, k..i]).map(|x| x.abs()).scalar_sum()
                    - self.slice(s![(i + 1).., i]).map(|x| x.abs()).scalar_sum();
            }

            // Modified Cholesky decomposition
            for j in k..(n - 2) {
                // Pivot on maximum lower Gershgorin bound estimate
                let max_idx = index_of_largest(&g.slice(s![j..]));
                if max_idx != j {
                    swap_rows(self, j, j + max_idx);
                    swap_columns(self, j, j + max_idx);
                }

                // Calculate E_jj and add to diagonal
                let normj = self.slice(s![(j + 1).., j]).map(|x| x.abs()).scalar_sum();
                delta = 0.0f64
                    .max(delta_prev)
                    .max(-self[(j, j)] + normj.max(tau * gamma));
                if delta > 0.0 {
                    self[(j, j)] += delta;
                    delta_prev = delta;
                }

                // Update Gershgorin bound estimates
                if (self[(j, j)] - normj).abs() > 100.0 * std::f64::EPSILON {
                    let tmp = 1.0 - normj / self[(j, j)];
                    for i in (j + 1)..n {
                        g[i] += self[(i, j)].abs() * tmp;
                    }
                }

                // perform jth iteration of factorization
                self[(j, j)] = self[(j, j)].sqrt();
                for i in (j + 1)..n {
                    self[(i, j)] /= self[(j, j)];
                    for k in (j + 1)..(i + 1) {
                        self[(i, k)] -= self[(i, j)] * self[(k, j)];
                    }
                }
            }

            // final 2x2 submatrix

            // this fixes the final 2x2 submatrix' symmetry
            self[(n - 2, n - 1)] = self[(n - 1, n - 2)];

            let (lhi, llo) = eigenvalues_2x2(&self.slice(s![(n - 2).., (n - 2)..]));
            delta = 0.0f64
                .max(-llo + tau * gamma.max(1.0 / (1.0 - tau) * (lhi - llo)))
                .max(delta_prev);
            if delta > 0.0 {
                self[(n - 2, n - 2)] += delta;
                self[(n - 1, n - 1)] += delta;
                // delta_prev = delta;
            }
            self[(n - 2, n - 2)] = self[(n - 2, n - 2)].sqrt();
            self[(n - 1, n - 2)] /= self[(n - 2, n - 2)];
            self[(n - 1, n - 1)] = (self[(n - 1, n - 1)] - self[(n - 1, n - 2)].powi(2)).sqrt();
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modified_cholesky_se90() {
        let mut a: ndarray::Array2<f64> =
            ndarray::arr2(&[[1.0, 1.0, 2.0], [1.0, 1.0, 3.0], [2.0, 3.0, 1.0]]);
        let res = ndarray::arr2(&[[3.0, 1.0, 2.0], [1.0, 3.2196, 3.0], [2.0, 3.0, 3.2196]]);
        a.mod_cholesky_se90().unwrap();
        // set upper triangle off diagonals to zero because its just garbage there
        a[(0, 1)] = 0.0;
        a[(0, 2)] = 0.0;
        a[(1, 2)] = 0.0;
        assert!(a.dot(&(a.t())).all_close(&res, 1e-4));
    }

    // #[test]
    // fn test_modified_cholesky_schnabel1_2() {
    //     use super::ModCholeskySchnabel1;
    //     let mut a: ndarray::Array2<f64> = ndarray::arr2(&[
    //         [0.3571, -0.1030, 0.0274, -0.0459],
    //         [-0.1030, 0.2525, 0.0736, -0.3845],
    //         [0.0274, 0.0736, 0.2340, -0.2878],
    //         [-0.0459, -0.3845, -0.2878, 0.5549],
    //     ]);
    //     a.mod_cholesky_schnabel1_inplace().unwrap();
    //
    //     // set upper triangle off diagonals to zero because its just garbage there
    //     a[(0, 1)] = 0.0;
    //     a[(0, 2)] = 0.0;
    //     a[(0, 3)] = 0.0;
    //     a[(1, 2)] = 0.0;
    //     a[(1, 3)] = 0.0;
    //     a[(2, 3)] = 0.0;
    //     let m = a.dot(&a.t());
    //     // println!("{:?}", m.into_diag());
    //     // println!(
    //     //     "{:?}",
    //     //     m.into_diag() - ndarray::arr1(&[0.3571, 0.2525, 0.2340, 0.5549])
    //     // );
    //     println!("{:?}", a);
    //     println!("{:?}", a.dot(&(a.t())));
    // }

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
