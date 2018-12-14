// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Schnabel & Eskow algorithm (1999)
//!
//! # References
//!
//! * Semyon Aranovich Gershgorin.
//!   Über die Abgrenzung der Eigenwerte einer Matrix.
//!   Izv. Akad. Nauk. USSR Otd. Fiz.-Mat. Nauk, 6: 749–754, 1931.
//! * Robert B. Schnabel and Elizabeth Eskow.
//!   A revised modified Cholesky factorization.
//!   SIAM J. Optim. Vol. 9, No. 4, pp. 1135-1148, 1999

use crate::utils::{eigenvalues_2x2, index_of_largest, swap_columns, swap_rows};
use failure::{bail, Error};

pub trait ModCholeskySE99 {
    fn mod_cholesky_se99(&mut self) -> Result<(), Error> {
        bail!("Not implemented!")
    }
}

impl ModCholeskySE99 for ndarray::Array2<f64> {
    fn mod_cholesky_se99(&mut self) -> Result<(), Error> {
        assert!(self.is_square());
        use ndarray::s;

        let n = self.raw_dim()[0];

        // let mut l = self.clone();
        let mut l = ndarray::Array2::zeros((n, n));

        // cbrt = cubic root
        let tau = std::f64::EPSILON.cbrt();
        let tau_bar = std::f64::EPSILON.cbrt();
        let mu = 0.1_f64;

        let mut phaseone = true;

        let gamma = self
            .diag()
            .fold(0.0, |acc, x| if x.abs() > acc { x.abs() } else { acc });

        let mut j = 0;

        // Phase one, `self` potentially positive-definite
        while j < n && phaseone {
            let aii_max = self
                .diag()
                .slice(s![j..])
                .fold(
                    std::f64::NEG_INFINITY,
                    |acc, &x| if x > acc { x } else { acc },
                );
            let aii_min = self
                .diag()
                .slice(s![j..])
                .fold(std::f64::INFINITY, |acc, &x| if x < acc { x } else { acc });
            if aii_max < tau_bar * gamma || aii_min < -mu * aii_max {
                phaseone = false;
                break;
            } else {
                // Pivot on maximum diagonal of remaining submatrix
                let max_idx = index_of_largest(&self.diag().slice(s![j..]));
                if max_idx != 0 {
                    swap_rows(self, j, j + max_idx);
                    swap_columns(self, j, j + max_idx);
                }
                let tmp = ((j + 1)..n).fold(std::f64::INFINITY, |acc, i| {
                    let nv = self[(i, i)] - self[(i, j)].powi(2) / self[(j, j)];
                    if nv < acc {
                        nv
                    } else {
                        acc
                    }
                });
                if tmp < -mu * gamma {
                    // Go to phase two
                    phaseone = false;
                    break;
                } else {
                    // Perform jth iteration of factorization
                    // self[(j, j)] = self[(j, j)].sqrt();
                    l[(j, j)] = self[(j, j)].sqrt();
                    // self[(j, j)] = l[(j, j)];
                    for i in (j + 1)..n {
                        // self[(i, j)] /= self[(j, j)];
                        l[(i, j)] = self[(i, j)] / l[(j, j)];
                        // self[(i, j)] = l[(i, j)];
                        for k in (j + 1)..=i {
                            // self[(i, k)] -= self[(i, j)] * self[(k, j)];
                            self[(i, k)] -= l[(i, j)] * l[(k, j)];
                        }
                    }
                    j += 1;
                }
            }
        }
        println!("j: {}", j);

        let mut delta;
        let mut delta_prev = 0.0;

        // Phase two, A not positive definite
        if !phaseone && j == n - 1 {
            delta = -self[(j, j)] + (tau_bar * gamma).max(tau * (-self[(j, j)]) / (1.0 - tau));
            self[(j, j)] += delta;
            // self[(j, j)] = (self[(j, j)] + delta).sqrt();
            l[(j, j)] = (self[(j, j)]).sqrt();
        }

        if !phaseone && j < n - 1 {
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
                // println!("j: {}", j);
                // Pivot on maximum lower Gershgorin bound estimate
                let max_idx = index_of_largest(&g.slice(s![j..]));
                if max_idx != 0 {
                    swap_rows(self, j, j + max_idx);
                    swap_columns(self, j, j + max_idx);
                }

                // Calculate E_jj and add to diagonal
                let normj = self.slice(s![(j + 1).., j]).map(|x| x.abs()).scalar_sum();
                delta = 0.0f64
                    .max(delta_prev)
                    .max(-self[(j, j)] + normj.max(tau_bar * gamma));
                if delta > 0.0 {
                    self[(j, j)] += delta;
                    delta_prev = delta;
                }

                // Update Gershgorin bound estimates
                if (self[(j, j)] - normj).abs() > 1.0 * std::f64::EPSILON {
                    let tmp = 1.0 - normj / self[(j, j)];
                    for i in (j + 1)..n {
                        g[i] += self[(i, j)].abs() * tmp;
                    }
                }

                // perform jth iteration of factorization
                // self[(j, j)] = self[(j, j)].sqrt();
                // for i in (j + 1)..n {
                //     self[(i, j)] /= self[(j, j)];
                //     for k in (j + 1)..(i + 1) {
                //         self[(i, k)] -= self[(i, j)] * self[(k, j)];
                //     }
                // }
                l[(j, j)] = self[(j, j)].sqrt();
                for i in (j + 1)..n {
                    // self[(i, j)] /= self[(j, j)];
                    l[(i, j)] = self[(i, j)] / l[(j, j)];
                    for k in (j + 1)..=i {
                        // self[(i, k)] -= self[(i, j)] * self[(k, j)];
                        self[(i, k)] -= l[(i, j)] * l[(k, j)];
                    }
                }
            }

            // final 2x2 submatrix

            // this fixes the final 2x2 submatrix' symmetry
            self[(n - 2, n - 1)] = self[(n - 1, n - 2)];
            // self[(n - 1, n - 2)] = self[(n - 2, n - 1)];
            println!("last: {:?}", self.slice(s![(n - 2).., (n - 2)..]));
            println!("last: {:?}", self);

            let (lhi, llo) = eigenvalues_2x2(&self.slice(s![(n - 2).., (n - 2)..]));
            delta = 0.0f64
                .max(-llo + (tau_bar * gamma).max(tau * (lhi - llo) / (1.0 - tau)))
                .max(delta_prev);
            if delta > 0.0 {
                self[(n - 2, n - 2)] += delta;
                self[(n - 1, n - 1)] += delta;
                // delta_prev = delta;
            }
            // self[(n - 2, n - 2)] = self[(n - 2, n - 2)].sqrt();
            // self[(n - 1, n - 2)] /= self[(n - 2, n - 2)];
            // self[(n - 1, n - 1)] = (self[(n - 1, n - 1)] - self[(n - 1, n - 2)].powi(2)).sqrt();
            l[(n - 2, n - 2)] = self[(n - 2, n - 2)].sqrt();
            l[(n - 1, n - 2)] = self[(n - 1, n - 2)] / l[(n - 2, n - 2)];
            l[(n - 1, n - 1)] = (self[(n - 1, n - 1)] - l[(n - 1, n - 2)].powi(2)).sqrt();
        }
        *self = l;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use crate::utils::*;

    #[test]
    fn test_modified_cholesky_se99() {
        // let mut a: ndarray::Array2<f64> =
        //     ndarray::arr2(&[[1.0, 1.0, 2.0], [1.0, 1.0, 3.0], [2.0, 3.0, 1.0]]);
        let mut a: ndarray::Array2<f64> =
            ndarray::arr2(&[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 3.0, 1.0]]);
        println!("A: {:?}", a);
        let res = ndarray::arr2(&[[3.0, 1.0, 2.0], [1.0, 3.2196, 3.0], [2.0, 3.0, 3.2196]]);
        a.mod_cholesky_se99().unwrap();
        // set upper triangle off diagonals to zero because its just garbage there
        // a[(0, 1)] = 0.0;
        // a[(0, 2)] = 0.0;
        // a[(1, 2)] = 0.0;
        // println!("L: {:?}", a);
        // println!("LLT: {:?}", a.dot(&a.t()));
        // println!("LTL: {:?}", a.t().dot(&a));
        // println!("RES: {:?}", res);
        assert!(a.dot(&(a.t())).all_close(&res, 1e-4));
    }

    // #[test]
    // fn test_modified_cholesky_se99_difficult() {
    //     // let mut a: ndarray::Array2<f64> = ndarray::arr2(&[
    //     //     [1890.3, -1705.6, -315.8, 3000.3],
    //     //     [-1705.6, 1538.3, 284.9, -2706.6],
    //     //     [-315.8, 284.9, 52.5, -501.2],
    //     //     [3000.3, -2706.6, -501.2, 4760.8],
    //     // ]);
    //     // let mut a: ndarray::Array2<f64> = ndarray::arr2(&[
    //     //     [1890.3, -1705.6, -315.8, 3000.3],
    //     //     [0.0, 1538.3, 284.9, -2706.6],
    //     //     [0.0, 0.0, 52.5, -501.2],
    //     //     [0.0, 0.0, 0.0, 4760.8],
    //     // ]);
    //     let mut a: ndarray::Array2<f64> = ndarray::arr2(&[
    //         [1890.3, 0.0, 0.0, 0.0],
    //         [-1705.6, 1538.3, 0.0, 0.0],
    //         [-315.8, 284.9, 52.5, 0.0],
    //         [3000.3, -2706.6, -501.2, 4760.8],
    //     ]);
    //     let mut res: ndarray::Array2<f64> = ndarray::arr2(&[
    //         [1890.3 + 0.6649, -1705.6, -315.8, 3000.3],
    //         [-1705.6, 1538.3 + 0.3666, 284.9, -2706.6],
    //         [-315.8, 284.9, 52.5 + 0.6649, -501.2],
    //         [3000.3, -2706.6, -501.2, 4760.8],
    //     ]);
    //     swap_rows(&mut res, 0, 3);
    //     swap_columns(&mut res, 0, 3);
    //
    //     a.mod_cholesky_se99().unwrap();
    //     // set upper triangle off diagonals to zero because its just garbage there
    //     // a[(0, 1)] = 0.0;
    //     // a[(0, 2)] = 0.0;
    //     // a[(0, 3)] = 0.0;
    //     // a[(1, 2)] = 0.0;
    //     // a[(1, 3)] = 0.0;
    //     // a[(2, 3)] = 0.0;
    //     println!("L: {:?}", a);
    //     println!("LLT: {:?}", a.dot(&a.t()));
    //     // println!("LTL: {:?}", a.t().dot(&a));
    //     println!("RES: {:?}", res);
    //     assert!(a.dot(&(a.t())).all_close(&res, 1e-2));
    // }

    // #[test]
    // fn test_modified_cholesky_se99_6x6() {
    //     // let mut a: ndarray::Array2<f64> = ndarray::arr2(&[
    //     //     [14.8253, -6.4243, 7.8746, -1.2498, 10.2733, 10.2733],
    //     //     [-6.4243, 15.1024, -1.1155, -0.2761, -8.2117, -8.2117],
    //     //     [7.8746, -1.1155, 51.8519, -23.3482, 12.5902, 12.5902],
    //     //     [-1.2498, -0.2761, -23.3482, 22.7967, -9.8958, -9.8958],
    //     //     [10.2733, -8.2117, 12.5902, -9.8958, 21.0656, 21.0656],
    //     //     [10.2733, -8.2117, 12.5902, -9.8958, 21.0656, 21.0656],
    //     // ]);
    //     let mut a: ndarray::Array2<f64> = ndarray::arr2(&[
    //         [14.8253, 0.0, 0.0, 0.0, 0.0, 0.0],
    //         [-6.4243, 15.1024, 0.0, 0.0, 0.0, 0.0],
    //         [7.8746, -1.1155, 51.8519, 0.0, 0.0, 0.0],
    //         [-1.2498, -0.2761, -23.3482, 22.7967, 0.0, 0.0],
    //         [10.2733, -8.2117, 12.5902, -9.8958, 21.0656, 0.0],
    //         [10.2733, -8.2117, 12.5902, -9.8958, 21.0656, 21.0656],
    //     ]);
    //     // let mut a: ndarray::Array2<f64> = ndarray::arr2(&[
    //     //     [14.8253, -6.4243, 7.8746, -1.2498, 10.2733, 10.2733],
    //     //     [0.0, 15.1024, -1.1155, -0.2761, -8.2117, -8.2117],
    //     //     [0.0, 0.0, 51.8519, -23.3482, 12.5902, 12.5902],
    //     //     [0.0, 0.0, 0.0, 22.7967, -9.8958, -9.8958],
    //     //     [0.0, 0.0, 0.0, 0.0, 21.0656, 21.0656],
    //     //     [0.0, 0.0, 0.0, 0.0, 0.0, 21.0656],
    //     // ]);
    //     let mut res: ndarray::Array2<f64> = ndarray::arr2(&[
    //         [14.8253, -6.4243, 7.8746, -1.2498, 10.2733, 10.2733],
    //         [-6.4243, 15.1024, -1.1155, -0.2761, -8.2117, -8.2117],
    //         [7.8746, -1.1155, 51.8519, -23.3482, 12.5902, 12.5902],
    //         [-1.2498, -0.2761, -23.3482, 22.7967, -9.8958, -9.8958],
    //         [10.2733, -8.2117, 12.5902, -9.8958, 21.0656, 21.0656],
    //         [10.2733, -8.2117, 12.5902, -9.8958, 21.0656, 21.0656],
    //     ]);
    //     // swap_rows(&mut res, 0, 3);
    //     // swap_columns(&mut res, 0, 3);
    //
    //     // use crate::se90::*;
    //     a.mod_cholesky_se99().unwrap();
    //     // set upper triangle off diagonals to zero because its just garbage there
    //     // a[(0, 1)] = 0.0;
    //     // a[(0, 2)] = 0.0;
    //     // a[(0, 3)] = 0.0;
    //     // a[(0, 4)] = 0.0;
    //     // a[(0, 5)] = 0.0;
    //     // a[(1, 2)] = 0.0;
    //     // a[(1, 3)] = 0.0;
    //     // a[(1, 4)] = 0.0;
    //     // a[(1, 5)] = 0.0;
    //     // a[(2, 3)] = 0.0;
    //     // a[(2, 4)] = 0.0;
    //     // a[(2, 5)] = 0.0;
    //     // a[(3, 4)] = 0.0;
    //     // a[(3, 5)] = 0.0;
    //     // a[(4, 5)] = 0.0;
    //     println!("L: {:?}", a);
    //     println!("LLT: {:?}", a.dot(&a.t()).diag());
    //     println!("RES: {:?}", res.diag());
    //     assert!(a.dot(&(a.t())).all_close(&res, 1e-2));
    // }
}
