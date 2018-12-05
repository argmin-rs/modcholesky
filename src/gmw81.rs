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
use failure::Error;

pub trait ModCholeskyGMW81
where
    Self: Sized,
{
    fn mod_cholesky_gmw81(&mut self) -> Result<(), Error>;
}

impl ModCholeskyGMW81 for ndarray::Array2<f64> {
    /// Algorithm 6.5 in "Numerical Optimization" by Nocedal and Wright
    fn mod_cholesky_gmw81(&mut self) -> Result<(), Error> {
        use ndarray::s;
        debug_assert!(self.is_square());
        let n = self.raw_dim()[0];
        let a_diag = self.diag();

        let diag_max = a_diag.fold(0.0, |acc, x| if x.abs() > acc { x.abs() } else { acc });
        let off_diag_max =
            self.indexed_iter()
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
        c.diag_mut().assign(&a_diag);
        let mut d: ndarray::Array1<f64> = ndarray::Array::zeros(n);

        for j in 0..n {
            let max_idx = index_of_largest_abs(&c.diag().slice(s![j..]));
            swap_rows(&mut c, j, j + max_idx);
            swap_columns(&mut c, j, j + max_idx);
            for s in 0..j {
                self[(j, s)] = c[(j, s)] / d[s];
            }

            for i in j..n {
                c[(i, j)] =
                    self[(i, j)] - (&self.slice(s![j, 0..j]) * &c.slice(s![i, 0..j])).scalar_sum();
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
        }
        let mut dout = ndarray::Array2::eye(n);
        dout.diag_mut()
            .assign(&(d.iter().map(|x| x.sqrt()).collect::<ndarray::Array1<f64>>()));

        // Set diagonal to ones
        self.diag_mut().assign(&ndarray::Array1::ones(n));

        // multiply with dout and return
        *self = self.dot(&dout);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_modified_cholesky() {
        let mut a: ndarray::Array2<f64> =
            ndarray::arr2(&[[4.0, 2.0, 1.0], [2.0, 6.0, 3.0], [1.0, 3.0, -0.004]]);
        a.mod_cholesky_gmw81().unwrap();
        // set upper triangle off diagonals to zero because its just garbage there
        a[(0, 1)] = 0.0;
        a[(0, 2)] = 0.0;
        a[(1, 2)] = 0.0;
        let f = a.dot(&(a.t()));
        let res: ndarray::Array2<f64> =
            ndarray::arr2(&[[4.0, 2.0, 1.0], [2.0, 6.0, 3.0], [1.0, 3.0, 3.004]]);
        assert!(f.all_close(&res, 1e-9));
        // let dsqrt = d.map(|x| x.sqrt());
        // let m = l.dot(&dsqrt);
        // println!("l: {:?}", l);
        // println!("d: {:?}", d);
        // println!("f: {:?}", f);
        // println!("m: {:?}", m);
    }
}
