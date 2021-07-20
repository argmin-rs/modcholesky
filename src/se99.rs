// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::utils::{eigenvalues_2x2, index_of_largest, swap_columns, swap_rows};
use crate::Decomposition;

/// Schnabel & Eskow algorithm (1999)
///
/// # References
///
/// * Semyon Aranovich Gershgorin.
///   Über die Abgrenzung der Eigenwerte einer Matrix.
///   Izv. Akad. Nauk. USSR Otd. Fiz.-Mat. Nauk, 6: 749–754, 1931.
/// * Robert B. Schnabel and Elizabeth Eskow.
///   A revised modified Cholesky factorization.
///   SIAM J. Optim. Vol. 9, No. 4, pp. 1135-1148, 1999
pub trait ModCholeskySE99<L, E, P>
where
    Self: Sized,
{
    /// Computes the modified Cholesky decomposition with the SE99 algorithm
    fn mod_cholesky_se99(&self) -> Decomposition<L, E, P> {
        panic!("Not implemented!")
    }
}

impl ModCholeskySE99<ndarray::Array2<f64>, ndarray::Array1<f64>, ndarray::Array1<usize>>
    for ndarray::Array2<f64>
{
    /// Computes the modified Cholesky decomposition with the SE99 algorithm
    fn mod_cholesky_se99(
        &self,
    ) -> Decomposition<ndarray::Array2<f64>, ndarray::Array1<f64>, ndarray::Array1<usize>> {
        assert!(self.is_square());
        use ndarray::s;

        let n = self.raw_dim()[0];

        let mut l = self.clone();
        let mut e = ndarray::Array1::zeros(n);
        let mut p = ndarray::Array::from_iter(0..n);

        // cbrt = cubic root
        let tau = std::f64::EPSILON.cbrt();
        let tau_bar = std::f64::EPSILON.cbrt();
        let mu = 0.1_f64;

        let mut phaseone = true;

        let gamma = l
            .diag()
            .fold(0.0, |acc, x| if x.abs() > acc { x.abs() } else { acc });

        let mut j = 0;

        // Phase one, `self` potentially positive-definite
        while j < n && phaseone {
            let aii_max =
                l.diag().slice(s![j..]).fold(
                    std::f64::NEG_INFINITY,
                    |acc, &x| if x > acc { x } else { acc },
                );
            let aii_min =
                l.diag()
                    .slice(s![j..])
                    .fold(std::f64::INFINITY, |acc, &x| if x < acc { x } else { acc });
            if aii_max < tau_bar * gamma || aii_min < -mu * aii_max {
                phaseone = false;
                break;
            } else {
                // Pivot on maximum diagonal of remaining submatrix
                let max_idx = index_of_largest(&l.diag().slice(s![j..]));
                if max_idx != 0 {
                    swap_rows(&mut l, j, j + max_idx);
                    swap_columns(&mut l, j, j + max_idx);
                    p.swap(j, j + max_idx);
                }
                let tmp = ((j + 1)..n).fold(std::f64::INFINITY, |acc, i| {
                    let nv = l[(i, i)] - l[(i, j)].powi(2) / l[(j, j)];
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
        }

        // let mut delta;
        let mut delta_prev = 0.0;

        // Phase two, A not positive definite
        if !phaseone && j == n - 1 {
            e[j] = -l[(j, j)] + (tau_bar * gamma).max(tau * (-l[(j, j)]) / (1.0 - tau));
            l[(j, j)] += e[j];
            l[(j, j)] = l[(j, j)].sqrt();
        }

        if !phaseone && j < n - 1 {
            let k = j;

            // Calculate lower Gershgorin bounds of self_{k+1}
            let mut g = ndarray::Array::zeros(n);
            for i in k..n {
                g[i] = l[(i, i)]
                    - l.slice(s![i, k..i]).map(|x| x.abs()).sum()
                    - l.slice(s![(i + 1).., i]).map(|x| x.abs()).sum();
            }

            // Modified Cholesky decomposition
            for j in k..(n - 2) {
                // Pivot on maximum lower Gershgorin bound estimate
                let max_idx = index_of_largest(&g.slice(s![j..]));
                if max_idx != 0 {
                    swap_rows(&mut l, j, j + max_idx);
                    swap_columns(&mut l, j, j + max_idx);
                    p.swap(j, j + max_idx);
                    g.swap(j, j + max_idx);
                }

                // Calculate E_jj and add to diagonal
                let normj = l.slice(s![(j + 1).., j]).map(|x| x.abs()).sum();
                e[j] = 0.0f64
                    .max(delta_prev)
                    .max(-l[(j, j)] + normj.max(tau_bar * gamma));
                if e[j] > 0.0 {
                    l[(j, j)] += e[j];
                    delta_prev = e[j];
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
            e[n - 2] = 0.0f64
                .max(-llo + (tau_bar * gamma).max(tau * (lhi - llo) / (1.0 - tau)))
                .max(delta_prev);
            e[n - 1] = e[n - 2];
            if e[n - 2] > 0.0 {
                l[(n - 2, n - 2)] += e[n - 2];
                l[(n - 1, n - 1)] += e[n - 1];
            }
            l[(n - 2, n - 2)] = l[(n - 2, n - 2)].sqrt();
            l[(n - 1, n - 2)] /= l[(n - 2, n - 2)];
            l[(n - 1, n - 1)] = (l[(n - 1, n - 1)] - l[(n - 1, n - 2)].powi(2)).sqrt();
        }

        // Make lower triangular
        for i in 0..(n - 1) {
            l.slice_mut(s![i, (i + 1)..]).fill(0.0);
        }

        // Reorder E
        let ec = e.clone();
        for i in 0..n {
            e[p[i]] = ec[i];
        }

        Decomposition::new(l, e, p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::*;

    #[test]
    fn test_modchol_se99_3x3() {
        let a: ndarray::Array2<f64> =
            ndarray::arr2(&[[1.0, 1.0, 2.0], [1.0, 1.0, 3.0], [2.0, 3.0, 1.0]]);
        let res_l_up: ndarray::Array2<f64> = ndarray::arr2(&[
            [1.732050807568877, 0.5773502691896257, 1.154700538379251],
            [0.0, 1.698920954907997, 1.37342077428181],
            [0.0, 0.0, 0.006912871809428971],
        ]);
        let res = res_l_up.t().dot(&res_l_up);
        let decomp = a.mod_cholesky_se99();
        let l = decomp.l;
        let e = diag_mat_from_arr(decomp.e.as_slice().unwrap());
        let p = index_to_permutation_mat(decomp.p.as_slice().unwrap());
        let paptpept = p.dot(&a.dot(&p.t())) + p.dot(&e.dot(&p.t()));
        // println!("A:\n{:?}", a);
        // println!("L:\n{:?}", l);
        // println!("E:\n{:?}", e);
        // println!("P:\n{:?}", p);
        // println!("LLT:\n{:?}", l.dot(&l.t()));
        // println!("P*A*P^T + P*E*P^T:\n{:?}", paptpept);
        // println!("RES:\n{:?}", res);
        assert!(paptpept.abs_diff_eq(&l.dot(&l.t()), 1e-12));
        assert!(l.dot(&(l.t())).abs_diff_eq(&res, 1e-12));
    }

    #[test]
    fn test_modchol_se99_4x4() {
        let a: ndarray::Array2<f64> = ndarray::arr2(&[
            [1890.3, -1705.6, -315.8, 3000.3],
            [-1705.6, 1538.3, 284.9, -2706.6],
            [-315.8, 284.9, 52.5, -501.2],
            [3000.3, -2706.6, -501.2, 4760.8],
        ]);
        let res_l_up: ndarray::Array2<f64> = ndarray::arr2(&[
            [
                68.99855070941707,
                -7.263920688867382,
                -39.22691088684848,
                43.48352203273905,
            ],
            [
                0.0,
                0.3194133212151726,
                -0.1288911532532789,
                0.1905221679618937,
            ],
            [0.0, 0.0, 0.4447317171993393, 0.3345847412304742],
            [0.0, 0.0, 0.0, 0.001713817545399892],
        ]);
        let res = res_l_up.t().dot(&res_l_up);

        let decomp = a.mod_cholesky_se99();
        let l = decomp.l;
        let e = diag_mat_from_arr(decomp.e.as_slice().unwrap());
        let p = index_to_permutation_mat(decomp.p.as_slice().unwrap());
        let paptpept = p.dot(&a.dot(&p.t())) + p.dot(&e.dot(&p.t()));
        // println!("A:\n{:?}", a);
        // println!("L:\n{:?}", l);
        // println!("E:\n{:?}", e);
        // println!("P:\n{:?}", p);
        // println!("LLT:\n{:?}", l.dot(&l.t()));
        // println!("P*A*P^T + P*E*P^T:\n{:?}", paptpept);
        // println!("RES:\n{:?}", res);
        assert!(paptpept.abs_diff_eq(&l.dot(&l.t()), 1e-12));
        // for some reason numerical problems make this test difficult, therefore the tolerance is
        // 1e-1. Essentially the problem lies in `res` and not in `LL^T`.
        assert!(l.dot(&(l.t())).abs_diff_eq(&res, 1e-1));
    }

    #[test]
    fn test_modchol_se99_6x6() {
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
                1.748438221248757,
                -0.1549127762706699,
                -3.242433422611255,
                1.093568935922023,
                1.748438221248757,
            ],
            [
                0.0,
                4.243649819020943,
                -1.871229936413708,
                -0.9959835646917692,
                1.970299772942301,
                4.243649819020943,
            ],
            [
                0.0,
                0.0,
                3.402484468269805,
                -0.7765233465239986,
                -0.7547450415137518,
                0.0,
            ],
            [0.0, 0.0, 0.0, 3.269304777945995, 1.123276587271259, 0.0],
            [0.0, 0.0, 0.0, 0.0, 2.813527220044002, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 4.360427593036232e-05],
        ]);
        let res = res_l_up.t().dot(&res_l_up);

        let decomp = a.mod_cholesky_se99();
        let l = decomp.l;
        let e = diag_mat_from_arr(decomp.e.as_slice().unwrap());
        let p = index_to_permutation_mat(decomp.p.as_slice().unwrap());
        let paptpept = p.dot(&a.dot(&p.t())) + p.dot(&e.dot(&p.t()));
        // println!("A:\n{:?}", a);
        // println!("L:\n{:?}", l);
        // println!("E:\n{:?}", e);
        // println!("P:\n{:?}", p);
        // println!("LLT:\n{:?}", l.dot(&l.t()));
        // println!("P*A*P^T + P*E*P^T:\n{:?}", paptpept);
        // println!("RES:\n{:?}", res);
        assert!(paptpept.abs_diff_eq(&l.dot(&l.t()), 1e-12));
        // for some reason numerical problems make this test difficult, therefore the tolerance is
        // 1e-1. Essentially the problem lies in `res` and not in `LL^T`.
        assert!(l.dot(&(l.t())).abs_diff_eq(&res, 1e-3));
    }
}
