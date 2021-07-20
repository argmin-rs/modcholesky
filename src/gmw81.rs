// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

use crate::utils::{index_of_largest_abs, swap_columns, swap_rows};
use crate::Decomposition;

/// Gill, Murray and Wright (1981)
///
/// Algorithm 6.5 in "Numerical Optimization" by Nocedal and Wright
///
/// # References
///
/// * Philip E. Gill, Walter Murray and Margaret H. Wright.
///   Practical Optimization.
///   Emerald Group Publishing Limited. ISBN 978-0122839528. 1982
/// * Jorge Nocedal and Stephen J. Wright.
///   Numerical Optimization.
///   Springer. ISBN 0-387-30303-0. 2006.

pub trait ModCholeskyGMW81<L, E, P>
where
    Self: Sized,
{
    /// Computes the modified Cholesky decomposition with the GMW81 algorithm
    fn mod_cholesky_gmw81(&self) -> Decomposition<L, E, P> {
        panic!("Not implemented!")
    }
}

impl ModCholeskyGMW81<ndarray::Array2<f64>, ndarray::Array1<f64>, ndarray::Array1<usize>>
    for ndarray::Array2<f64>
{
    /// Computes the modified Cholesky decomposition with the GMW81 algorithm.
    /// Based on algorithm 6.5 in "Numerical Optimization" by Nocedal and Wright.
    fn mod_cholesky_gmw81(
        &self,
    ) -> Decomposition<ndarray::Array2<f64>, ndarray::Array1<f64>, ndarray::Array1<usize>> {
        use ndarray::s;
        debug_assert!(self.is_square());
        let n = self.raw_dim()[0];
        let mut l = self.clone();
        let mut e = ndarray::Array1::zeros(n);
        let mut p: ndarray::Array1<usize> = ndarray::Array::from_iter(0..n);

        let diag_max = l
            .diag()
            .fold(0.0, |acc, x| if x.abs() > acc { x.abs() } else { acc });
        let off_diag_max =
            l.indexed_iter()
                .filter(|((i, j), _)| i != j)
                .fold(
                    0.0,
                    |acc, ((_, _), x)| if x.abs() > acc { x.abs() } else { acc },
                );

        let delta = std::f64::EPSILON * 1.0f64.max(diag_max + off_diag_max);
        let beta = (diag_max
            .max(off_diag_max / ((n as f64).powi(2) - 1.0).sqrt())
            .max(std::f64::EPSILON))
        .sqrt();

        let mut c: ndarray::Array2<f64> = ndarray::Array2::zeros(self.raw_dim());
        c.diag_mut().assign(&l.diag());
        let mut d: ndarray::Array1<f64> = ndarray::Array::zeros(n);

        for j in 0..n {
            let max_idx = index_of_largest_abs(&c.diag().slice(s![j..]));
            if max_idx != 0 {
                swap_rows(&mut c, j, j + max_idx);
                swap_columns(&mut c, j, j + max_idx);
                swap_rows(&mut l, j, j + max_idx);
                swap_columns(&mut l, j, j + max_idx);
                p.swap(j, j + max_idx);
            }
            for s in 0..j {
                l[(j, s)] = c[(j, s)] / d[s];
            }

            for i in j..n {
                c[(i, j)] = l[(i, j)] - (&l.slice(s![j, 0..j]) * &c.slice(s![i, 0..j])).sum();
            }

            let theta = if j < (n - 1) {
                c.slice(s![(j + 1).., j]).fold(
                    0.0,
                    |acc, &x| {
                        if x.abs() > acc {
                            x.abs()
                        } else {
                            acc
                        }
                    },
                )
            } else {
                0.0
            };

            d[j] = c[(j, j)].abs().max((theta / beta).powi(2)).max(delta);

            if j < (n - 1) {
                for i in (j + 1)..n {
                    let c2 = c[(i, j)].powi(2);
                    c[(i, i)] -= c2 / d[j];
                }
            }
            e[j] = d[j] - c[(j, j)];
        }
        let mut dout = ndarray::Array2::eye(n);
        dout.diag_mut()
            .assign(&(d.iter().map(|x| x.sqrt()).collect::<ndarray::Array1<f64>>()));

        // Set diagonal to ones
        l.diag_mut().assign(&ndarray::Array1::ones(n));

        // Make lower triangular
        for i in 0..(n - 1) {
            l.slice_mut(s![i, (i + 1)..]).fill(0.0);
        }

        // multiply with dout and return
        l = l.dot(&dout);

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
    fn test_modchol_gmw81_3x3() {
        let a: ndarray::Array2<f64> =
            ndarray::arr2(&[[4.0, 2.0, 1.0], [2.0, 6.0, 3.0], [1.0, 3.0, -0.004]]);
        let res: ndarray::Array2<f64> =
            ndarray::arr2(&[[4.0, 2.0, 1.0], [2.0, 6.0, 3.0], [1.0, 3.0, 3.004]]);

        let decomp = a.mod_cholesky_gmw81();
        let l = decomp.l;
        let e = diag_mat_from_arr(decomp.e.as_slice().unwrap());
        let p = index_to_permutation_mat(decomp.p.as_slice().unwrap());
        let res = p.dot(&res.dot(&p.t()));
        let paptpept = p.dot(&a.dot(&p.t())) + p.dot(&e.dot(&p.t()));
        // println!("A:\n{:?}", a);
        // println!("L:\n{:?}", l);
        // println!("E:\n{:?}", e);
        // println!("P:\n{:?}", p);
        // println!("LLT:\n{:?}", l.dot(&l.t()));
        // println!("P*A*P^T + P*E*P^T:\n{:?}", paptpept);
        // println!("RES:\n{:?}", res);
        assert!(paptpept.abs_diff_eq(&l.dot(&l.t()), 1e-2));
        assert!(l.dot(&(l.t())).abs_diff_eq(&res, 1e-1));
    }

    #[test]
    fn test_modchol_gmw81_3x3_2() {
        let a: ndarray::Array2<f64> =
            ndarray::arr2(&[[1.0, 1.0, 2.0], [1.0, 1.0, 3.0], [2.0, 3.0, 1.0]]);
        let res_l_up: ndarray::Array2<f64> = ndarray::arr2(&[
            [1.941967086829294, 0.5149417859767794, 1.029883571953559],
            [0.0, 2.398008844267161, 1.029883571953559],
            [0.0, 0.0, 1.058924144384121],
        ]);
        let res = res_l_up.t().dot(&res_l_up);
        let decomp = a.mod_cholesky_gmw81();
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
    fn test_modchol_gmw81_4x4() {
        let a: ndarray::Array2<f64> = ndarray::arr2(&[
            [1890.3, -1705.6, -315.8, 3000.3],
            [-1705.6, 1538.3, 284.9, -2706.6],
            [-315.8, 284.9, 52.5, -501.2],
            [3000.3, -2706.6, -501.2, 4760.8],
        ]);
        let res_l_up: ndarray::Array2<f64> = ndarray::arr2(&[
            [
                68.99855070941707,
                43.48352203273905,
                -39.22691088684848,
                -7.263920688867382,
            ],
            [
                0.0,
                0.7188103864735332,
                0.1728464514497895,
                0.08466115623673466,
            ],
            [0.0, 0.0, 0.6931187636550633, -0.0805099135864835],
            [0.0, 0.0, 0.0, 0.5274401688501188],
        ]);
        let res = res_l_up.t().dot(&res_l_up);

        let decomp = a.mod_cholesky_gmw81();
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
        assert!(paptpept.abs_diff_eq(&l.dot(&l.t()), 1e-1));
        assert!(l.dot(&(l.t())).abs_diff_eq(&res, 1e-1));
    }

    #[test]
    fn test_modchol_gmw81_6x6() {
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
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.490116119384766e-08],
        ]);
        let res = res_l_up.t().dot(&res_l_up);

        let decomp = a.mod_cholesky_gmw81();
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
}
