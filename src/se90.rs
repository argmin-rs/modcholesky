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

            let tmp = ((j + 1)..n).fold(std::f64::INFINITY, |acc, i| {
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
                    l[(j, i)] /= l[(j, j)];
                    for k in (j + 1)..=i {
                        l[(i, k)] -= l[(i, j)] * l[(k, j)];
                        // TEST
                        l[(k, i)] = l[(i, k)];
                    }
                }
                j += 1;
            }
        }

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
                    g.swap(j, j + max_idx);
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
                if (l[(j, j)] - normj).abs() > 1.0 * std::f64::EPSILON {
                    let tmp = 1.0 - normj / l[(j, j)];
                    for i in (j + 1)..n {
                        g[i] += l[(i, j)].abs() * tmp;
                    }
                }

                // Perform jth iteration of factorization
                l[(j, j)] = l[(j, j)].sqrt();
                for i in (j + 1)..n {
                    l[(i, j)] /= l[(j, j)];
                    l[(j, i)] /= l[(j, j)];
                    for k in (j + 1)..=i {
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
        for i in 0..(n - 1) {
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
        let res_l_up: ndarray::Array2<f64> = ndarray::arr2(&[
            [1.732050807568877, 0.5773502691896257, 1.154700538379251],
            [0.0, 1.698920954907997, 1.37342077428181],
            [0.0, 0.0, 0.006912871809428971],
        ]);
        let res = res_l_up.t().dot(&res_l_up);
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
        assert!(paptpept.all_close(&l.dot(&l.t()), 1e-12));
        assert!(l.dot(&(l.t())).all_close(&res, 1e-12));
    }

    #[test]
    fn test_modified_cholesky_se90_difficult() {
        let a: ndarray::Array2<f64> = ndarray::arr2(&[
            [1890.3, -1705.6, -315.8, 3000.3],
            [-1705.6, 1538.3, 284.9, -2706.6],
            [-315.8, 284.9, 52.5, -501.2],
            [3000.3, -2706.6, -501.2, 4760.8],
        ]);
        let res_l_up: ndarray::Array2<f64> = ndarray::arr2(&[
            [
                33.19487912314187,
                -15.09871441738697,
                8.582649117145946,
                -9.513515588608952,
            ],
            [0.0, 74.71431471239089, -34.4915568315879, 38.23441539976376],
            [0.0, 0.0, 36.39190351527789, -8.386049991060743],
            [0.0, 0.0, 0.0, 36.29044868461993],
        ]);
        let res = res_l_up.t().dot(&res_l_up);

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
        assert!(paptpept.all_close(&l.dot(&l.t()), 1e-12));
        assert!(l.dot(&(l.t())).all_close(&res, 1e-12));
    }

    #[test]
    fn test_modified_cholesky_se90_6x6() {
        let a: ndarray::Array2<f64> = ndarray::arr2(&[
            [14.8253, -6.4243, 7.8746, -1.2498, 10.2733, 10.2733],
            [-6.4243, 15.1024, -1.1155, -0.2761, -8.2117, -8.2117],
            [7.8746, -1.1155, 51.8519, -23.3482, 12.5902, 12.5902],
            [-1.2498, -0.2761, -23.3482, 22.7967, -9.8958, -9.8958],
            [10.2733, -8.2117, 12.5902, -9.8958, 21.0656, 21.0656],
            [10.2733, -8.2117, 12.5902, -9.8958, 21.0656, 21.0656],
        ]);
        let res_l_up: ndarray::Array2<f64> = ndarray::arr2(&[
            [
                7.200826341469429,
                -3.242433422611255,
                -0.1549127762706699,
                1.093568935922023,
                1.748438221248757,
                1.748438221248757,
            ],
            [
                0.0,
                3.504757552232888,
                -0.2220964936286686,
                0.6551164905259115,
                -1.205962298692896,
                -1.205962298692896,
            ],
            [
                0.0,
                0.0,
                4.746236644186862,
                -1.287207862277906,
                -1.729514390954478,
                -1.729514390954478,
            ],
            [
                0.0,
                0.0,
                0.0,
                4.363600851232103,
                1.587006643784825,
                1.587006643784825,
            ],
            [0.0, 0.0, 0.0, 0.0, 4.306053379607389, 2.564856408184844],
            [0.0, 0.0, 0.0, 0.0, 0.0, 3.458844794641899],
        ]);
        let res = res_l_up.t().dot(&res_l_up);

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
        assert!(paptpept.all_close(&l.dot(&l.t()), 1e-3));
        assert!(l.dot(&(l.t())).all_close(&res, 1e-3));
    }
}
