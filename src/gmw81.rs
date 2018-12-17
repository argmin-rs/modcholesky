// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Gill, Murray and Wright (1981)
//!
//! Algorithm 6.5 in "Numerical Optimization" by Nocedal and Wright
//!
//! # References
//!
//! * Philip E. Gill, Walter Murray and Margaret H. Wright.
//!   Practical Optimization.
//!   Emerald Group Publishing Limited. ISBN 978-0122839528. 1982
//! * Jorge Nocedal and Stephen J. Wright.
//!   Numerical Optimization.
//!   Springer. ISBN 0-387-30303-0. 2006.

use crate::utils::{index_of_largest_abs, swap_columns, swap_rows};
use crate::Decomposition;
use failure::Error;

pub trait ModCholeskyGMW81
where
    Self: Sized,
{
    fn mod_cholesky_gmw81(&self) -> Result<Decomposition<Self, Self, Self>, Error>;
}

impl ModCholeskyGMW81 for ndarray::Array2<f64> {
    /// Algorithm 6.5 in "Numerical Optimization" by Nocedal and Wright
    fn mod_cholesky_gmw81(&self) -> Result<Decomposition<Self, Self, Self>, Error> {
        use ndarray::s;
        debug_assert!(self.is_square());
        let n = self.raw_dim()[0];
        let mut l = self.clone();
        let mut p = ndarray::Array2::eye(n);
        let mut e = ndarray::Array2::zeros((n, n));

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
                swap_rows(&mut p, j, j + max_idx);
            }
            for s in 0..j {
                l[(j, s)] = c[(j, s)] / d[s];
            }

            for i in j..n {
                c[(i, j)] =
                    l[(i, j)] - (&l.slice(s![j, 0..j]) * &c.slice(s![i, 0..j])).scalar_sum();
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
            e[(j, j)] = d[j] - c[(j, j)];
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
        Ok(Decomposition::new(l.clone(), p.dot(&e.dot(&p.t())), p))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_modchol_gmw81_3x3() {
        let a: ndarray::Array2<f64> =
            ndarray::arr2(&[[4.0, 2.0, 1.0], [2.0, 6.0, 3.0], [1.0, 3.0, -0.004]]);
        let res: ndarray::Array2<f64> =
            ndarray::arr2(&[[4.0, 2.0, 1.0], [2.0, 6.0, 3.0], [1.0, 3.0, 3.004]]);

        let decomp = a.mod_cholesky_gmw81().unwrap();
        let l = decomp.l;
        // let e = decomp.e;
        // let p = decomp.p;
        // let paptpept = p.dot(&a.dot(&p.t())) + p.dot(&e.dot(&p.t()));
        // println!("A:\n{:?}", a);
        // println!("L:\n{:?}", l);
        // println!("E:\n{:?}", e);
        // println!("P:\n{:?}", p);
        // println!("LLT:\n{:?}", l.dot(&l.t()));
        // println!("P*A*P^T + P*E*P^T:\n{:?}", paptpept);
        // println!("RES:\n{:?}", res);
        // assert!(paptpept.all_close(&l.dot(&l.t()), 1e-2));
        assert!(l.dot(&(l.t())).all_close(&res, 1e-12));
    }
}
