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
use crate::Decomposition;
use failure::{bail, Error};

pub trait ModCholeskySE90
where
    Self: Sized,
{
    fn mod_cholesky_se90(&self) -> Result<Decomposition<Self, Self, Self>, Error> {
        bail!("Not implemented!")
    }
}

impl ModCholeskySE90 for ndarray::Array2<f64> {
    fn mod_cholesky_se90(&self) -> Result<Decomposition<Self, Self, Self>, Error> {
        assert!(self.is_square());
        use ndarray::s;

        let n = self.raw_dim()[0];

        let mut l = self.clone();
        let mut e = ndarray::Array2::zeros((n, n));
        let mut p = ndarray::Array2::eye(n);

        // cbrt = cubic root
        let tau = std::f64::EPSILON.cbrt();

        let mut phaseone = true;

        let gamma = l
            .diag()
            .fold(0.0, |acc, x| if x.abs() > acc { x.abs() } else { acc });

        let mut j = 0;
        // Phase one, `self` potentially positive-definite
        while j < n && phaseone {
            // Pivot on maximum diagonal of remaining submatrix
            let max_idx = index_of_largest(&l.diag().slice(s![j..]));
            if max_idx != 0 {
                swap_rows(&mut l, j, j + max_idx);
                swap_columns(&mut l, j, j + max_idx);
                swap_rows(&mut p, j, j + max_idx);
            }
            let tmp = (j + 1..n).fold(std::f64::INFINITY, |acc, i| {
                let nv = l[(i, i)] - l[(i, j)].powi(2) / l[(j, j)];
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
                l[(j, j)] = l[(j, j)].sqrt();
                for i in (j + 1)..n {
                    l[(i, j)] /= l[(j, j)];
                    for k in (j + 1)..(i + 1) {
                        l[(i, k)] -= l[(i, j)] * l[(k, j)];
                        // TEST
                        l[(k, i)] = l[(i, k)];
                    }
                }
                j += 1;
            }
        }

        // let mut delta;
        let mut delta_prev = 0.0;

        // Phase two, `self` not positive-definite
        if !phaseone {
            let k = j;

            // Calculate lower Gershgorin bounds of self_{k+1}
            let mut g = ndarray::Array::zeros(n);
            for i in k..n {
                g[i] = l[(i, i)]
                    - l.slice(s![i, k..i]).map(|x| x.abs()).scalar_sum()
                    - l.slice(s![(i + 1).., i]).map(|x| x.abs()).scalar_sum();
            }

            // Modified Cholesky decomposition
            for j in k..(n - 2) {
                // Pivot on maximum lower Gershgorin bound estimate
                let max_idx = index_of_largest(&g.slice(s![j..]));
                if max_idx != 0 {
                    swap_rows(&mut l, j, j + max_idx);
                    swap_columns(&mut l, j, j + max_idx);
                    swap_rows(&mut p, j, j + max_idx);
                }

                // Calculate E_jj and add to diagonal
                let normj = l.slice(s![(j + 1).., j]).map(|x| x.abs()).scalar_sum();
                e[(j, j)] = 0.0f64
                    .max(delta_prev)
                    .max(-l[(j, j)] + normj.max(tau * gamma));
                if e[(j, j)] > 0.0 {
                    l[(j, j)] += e[(j, j)];
                    delta_prev = e[(j, j)];
                }

                // Update Gershgorin bound estimates
                if (l[(j, j)] - normj).abs() > 100.0 * std::f64::EPSILON {
                    let tmp = 1.0 - normj / l[(j, j)];
                    for i in (j + 1)..n {
                        g[i] += l[(i, j)].abs() * tmp;
                    }
                }

                // perform jth iteration of factorization
                l[(j, j)] = l[(j, j)].sqrt();
                for i in (j + 1)..n {
                    l[(i, j)] /= l[(j, j)];
                    for k in (j + 1)..(i + 1) {
                        l[(i, k)] -= l[(i, j)] * l[(k, j)];
                        // TEST
                        l[(k, i)] = l[(i, k)];
                    }
                }
            }

            // final 2x2 submatrix
            let (lhi, llo) = eigenvalues_2x2(&l.slice(s![(n - 2).., (n - 2)..]));
            e[(n - 2, n - 2)] = 0.0f64
                .max(-llo + tau * gamma.max(1.0 / (1.0 - tau) * (lhi - llo)))
                .max(delta_prev);
            e[(n - 1, n - 1)] = e[(n - 2, n - 2)];
            if e[(n - 2, n - 2)] > 0.0 {
                l[(n - 2, n - 2)] += e[(n - 2, n - 2)];;
                l[(n - 1, n - 1)] += e[(n - 1, n - 1)];;
                // delta_prev = delta;
            }
            l[(n - 2, n - 2)] = l[(n - 2, n - 2)].sqrt();
            l[(n - 1, n - 2)] /= l[(n - 2, n - 2)];
            l[(n - 1, n - 1)] = (l[(n - 1, n - 1)] - l[(n - 1, n - 2)].powi(2)).sqrt();
        }

        // Make lower triangular
        for i in 0..n {
            l.slice_mut(s![i, (i + 1)..]).fill(0.0);
        }

        Ok(Decomposition::new(l, p.dot(&e.dot(&p.t())), p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modified_cholesky_se90() {
        let a: ndarray::Array2<f64> =
            ndarray::arr2(&[[1.0, 1.0, 2.0], [1.0, 1.0, 3.0], [2.0, 3.0, 1.0]]);
        let res = ndarray::arr2(&[[3.0, 1.0, 2.0], [1.0, 3.2197, 3.0], [2.0, 3.0, 3.2197]]);
        let decomp = a.mod_cholesky_se90().unwrap();
        let l = decomp.l;
        let e = decomp.e;
        let p = decomp.p;
        let paptpept = p.dot(&a.dot(&p.t())) + p.dot(&e.dot(&p.t()));
        // println!("A:\n{:?}", a);
        // println!("L:\n{:?}", l);
        // println!("E:\n{:?}", e);
        // println!("P:\n{:?}", p);
        // println!("LLT:\n{:?}", l.dot(&l.t()));
        // println!("P*A*P^T + P*E*P^T:\n{:?}", paptpept);
        // println!("RES:\n{:?}", res);
        assert!(paptpept.all_close(&l.dot(&l.t()), 1e-4));
        assert!(l.dot(&(l.t())).all_close(&res, 1e-4));
    }
}
