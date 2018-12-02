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
//! * Semyon Aranovich Gerschgorin.
//!   Über die Abgrenzung der Eigenwerte einer Matrix.
//!   Izv. Akad. Nauk. USSR Otd. Fiz.-Mat. Nauk, 6: 749–754, 1931.
//! * Robert B. Schnabel and Elizabeth Eskow.
//!   A new modified Cholesky factorization.
//!   SIAM J. Sci. Stat. Comput. Vol. 11, No. 6, pp. 1136-1158, November 1990
//! * Sheung Hun Cheng and Nicholas J. Higham.
//!   A modified Cholesky algorithm based on a symmetric indefinite factorization.
//!   SIAM J. Matrix Anal. Appl. Vol. 19, No. 4, pp. 1097-1110, October 1998
//! * Robert B. Schnabel and Elizabeth Eskow.
//!   A revised modified Cholesky factorization.
//!   SIAM J. Optim. Vol. 9, No. 4, pp. 1135-1148, 1999

mod gershgorin;

// use crate::gershgorin::*;
use failure::{bail, Error};

// pub struct ModCholesky<L, E> {
//     l: L,
//     e: E,
// }

pub trait ModCholeskySchnabel1
where
    Self: Sized,
{
    type L;
    fn mod_cholesky_schnabel1(&self) -> Result<Self::L, Error> {
        bail!("Not implemented!")
    }

    fn mod_cholesky_schnabel1_inplace(&mut self) -> Result<(), Error> {
        bail!("Not implemented!")
    }
}

impl ModCholeskySchnabel1 for ndarray::Array2<f64> {
    type L = ndarray::Array2<f64>;

    fn mod_cholesky_schnabel1_inplace(&mut self) -> Result<(), Error> {
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
                println!("{:?}", g);
                println!("max: {}: j: {}", max_idx, j);
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
                if (self[(j, j)] - normj).abs() > 1.0 * std::f64::EPSILON {
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
            self[(n - 1, n - 2)] = self[(n - 1, n - 2)] / self[(n - 2, n - 2)];
            self[(n - 1, n - 1)] = (self[(n - 1, n - 1)] - self[(n - 1, n - 2)].powi(2)).sqrt();
        }

        Ok(())
    }
}

fn eigenvalues_2x2(mat: &ndarray::ArrayView2<f64>) -> (f64, f64) {
    let a = mat[(0, 0)];
    let b = mat[(0, 1)];
    let c = mat[(1, 0)];
    let d = mat[(1, 1)];
    let tmp = ((-(a + d) / 2.0).powi(2) - a * d + b * c).sqrt();
    let l1 = (a + d) / 2.0 + tmp;
    let l2 = (a + d) / 2.0 - tmp;
    if l1.abs() > l2.abs() {
        (l1, l2)
    } else {
        (l2, l1)
    }
}

fn swap_columns<T>(mat: &mut ndarray::Array2<T>, idx1: usize, idx2: usize)
where
    ndarray::OwnedRepr<T>: ndarray::Data,
{
    let s = mat.raw_dim();
    for i in 0..s[0] {
        mat.swap((i, idx1), (i, idx2));
    }
}

fn swap_rows<T>(mat: &mut ndarray::Array2<T>, idx1: usize, idx2: usize)
where
    ndarray::OwnedRepr<T>: ndarray::Data,
{
    let s = mat.raw_dim();
    for i in 0..s[1] {
        mat.swap((idx1, i), (idx2, i));
    }
}

fn index_of_largest<'a, T>(c: &ndarray::ArrayView1<T>) -> usize
where
    <ndarray::ViewRepr<&'a T> as ndarray::Data>::Elem:
        std::cmp::PartialOrd + num::traits::Signed + Clone,
{
    // let mut max = num::abs(c[0].clone());
    let mut max = c[0].clone();
    let mut max_idx = 0;
    c.iter()
        .enumerate()
        .map(|(i, ci)| {
            // let ci = num::abs(ci.clone());
            let ci = ci.clone();
            if ci > max {
                max = ci;
                max_idx = i
            }
        })
        .count();
    max_idx
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_swap_columns() {
        let mut a: ndarray::Array2<i64> = ndarray::arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        super::swap_columns(&mut a, 1, 2);
        let c: ndarray::Array2<i64> = ndarray::arr2(&[[1, 3, 2], [4, 6, 5], [7, 9, 8]]);
        a.iter()
            .zip(c.iter())
            .map(|(x, y)| assert_eq!(*x, *y))
            .count();
        // this should work, but it doesn't.
        // assert_eq!(b, c);
    }

    #[test]
    fn test_swap_rows() {
        let mut a: ndarray::Array2<i64> = ndarray::arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        super::swap_rows(&mut a, 1, 2);
        let c: ndarray::Array2<i64> = ndarray::arr2(&[[1, 2, 3], [7, 8, 9], [4, 5, 6]]);
        a.iter()
            .zip(c.iter())
            .map(|(x, y)| assert_eq!(*x, *y))
            .count();
        // this should work, but it doesn't.
        // assert_eq!(b, c);
    }

    #[test]
    fn test_swap_rows_and_columns() {
        let mut a: ndarray::Array2<i64> = ndarray::arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
        super::swap_rows(&mut a, 1, 2);
        super::swap_columns(&mut a, 1, 2);
        let c: ndarray::Array2<i64> = ndarray::arr2(&[[1, 3, 2], [7, 9, 8], [4, 6, 5]]);
        a.iter()
            .zip(c.iter())
            .map(|(x, y)| assert_eq!(*x, *y))
            .count();
        // this should work, but it doesn't.
        // assert_eq!(b, c);
    }

    #[test]
    fn test_biggest_index() {
        use ndarray::s;
        let j = 1;
        let a: ndarray::Array2<i64> =
            ndarray::arr2(&[[1, 2, 3, 0], [4, 2, 6, 0], [7, 8, 3, 0], [3, 4, 2, 8]]);
        let idx = super::index_of_largest(&a.diag().slice(s![j..]));
        assert_eq!(idx + j, 3);
        // this should work, but it doesn't.
        // assert_eq!(b, c);
    }

    #[test]
    fn test_modified_cholesky_schnabel1() {
        use super::ModCholeskySchnabel1;
        let mut a: ndarray::Array2<f64> =
            ndarray::arr2(&[[1.0, 1.0, 2.0], [1.0, 1.0, 3.0], [2.0, 3.0, 1.0]]);
        let res = ndarray::arr2(&[[3.0, 1.0, 2.0], [1.0, 3.2196, 3.0], [2.0, 3.0, 3.2196]]);
        a.mod_cholesky_schnabel1_inplace().unwrap();
        a[(0, 1)] = 0.0;
        a[(0, 2)] = 0.0;
        a[(1, 2)] = 0.0;
        println!("{:?}", a);
        println!("{:?}", a.dot(&(a.t())));
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
    //     // let mut a: ndarray::Array2<f64> =
    //     //     ndarray::arr2(&[[1.0, 1.0, 2.0], [1.0, 1.0, 3.0], [2.0, 3.0, 1.0]]);
    //     let l = a.mod_cholesky_schnabel1_inplace().unwrap();
    //     a[(0, 1)] = 0.0;
    //     a[(0, 2)] = 0.0;
    //     a[(1, 2)] = 0.0;
    //     a[(0, 3)] = 0.0;
    //     a[(1, 3)] = 0.0;
    //     a[(2, 3)] = 0.0;
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
