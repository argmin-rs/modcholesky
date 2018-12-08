// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Benches

#![feature(test)]

extern crate modcholesky;
extern crate test;

#[cfg(test)]
mod tests {
    use modcholesky::*;
    use ndarray;
    use test::{black_box, Bencher};

    #[bench]
    fn gmw81_3x3_nd(b: &mut Bencher) {
        let mut a: ndarray::Array2<f64> =
            ndarray::arr2(&[[1.0, 1.0, 2.0], [1.0, 1.0, 3.0], [2.0, 3.0, 1.0]]);
        b.iter(|| {
            black_box(a.mod_cholesky_gmw81().unwrap());
        });
    }

    #[bench]
    fn gmw81_4x4_nd(b: &mut Bencher) {
        let mut a: ndarray::Array2<f64> = ndarray::arr2(&[
            [1890.3, -1705.6, -315.8, 3000.3],
            [-1705.6, 1538.3, 284.9, -2706.6],
            [-315.8, 284.9, 52.5, -501.2],
            [3000.3, -2706.6, -501.2, 4760.8],
        ]);
        b.iter(|| {
            black_box(a.mod_cholesky_gmw81().unwrap());
        });
    }

    #[bench]
    fn se90_3x3_nd(b: &mut Bencher) {
        let mut a: ndarray::Array2<f64> =
            ndarray::arr2(&[[1.0, 1.0, 2.0], [1.0, 1.0, 3.0], [2.0, 3.0, 1.0]]);
        b.iter(|| {
            black_box(a.mod_cholesky_se90().unwrap());
        });
    }

    #[bench]
    fn se90_4x4_nd(b: &mut Bencher) {
        let mut a: ndarray::Array2<f64> = ndarray::arr2(&[
            [1890.3, -1705.6, -315.8, 3000.3],
            [-1705.6, 1538.3, 284.9, -2706.6],
            [-315.8, 284.9, 52.5, -501.2],
            [3000.3, -2706.6, -501.2, 4760.8],
        ]);
        b.iter(|| {
            black_box(a.mod_cholesky_se90().unwrap());
        });
    }

    #[bench]
    fn se99_3x3_nd(b: &mut Bencher) {
        let mut a: ndarray::Array2<f64> =
            ndarray::arr2(&[[1.0, 1.0, 2.0], [1.0, 1.0, 3.0], [2.0, 3.0, 1.0]]);
        b.iter(|| {
            black_box(a.mod_cholesky_se99().unwrap());
        });
    }

    #[bench]
    fn se99_4x4_nd(b: &mut Bencher) {
        let mut a: ndarray::Array2<f64> = ndarray::arr2(&[
            [1890.3, -1705.6, -315.8, 3000.3],
            [-1705.6, 1538.3, 284.9, -2706.6],
            [-315.8, 284.9, 52.5, -501.2],
            [3000.3, -2706.6, -501.2, 4760.8],
        ]);
        b.iter(|| {
            black_box(a.mod_cholesky_se99().unwrap());
        });
    }
}
