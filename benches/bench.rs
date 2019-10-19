// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Benches

#![feature(test)]

extern crate modcholesky;
extern crate openblas_src;
extern crate test;

#[cfg(test)]
mod tests {
    use modcholesky::utils::{random_diagonal, random_householder};
    use modcholesky::*;
    use ndarray;
    use test::{black_box, Bencher};

    #[bench]
    fn gmw81_3x3_nd(b: &mut Bencher) {
        let a: ndarray::Array2<f64> =
            ndarray::arr2(&[[1.0, 1.0, 2.0], [1.0, 1.0, 3.0], [2.0, 3.0, 1.0]]);
        b.iter(|| {
            black_box(a.mod_cholesky_gmw81());
        });
    }

    #[bench]
    fn gmw81_4x4_nd(b: &mut Bencher) {
        let a: ndarray::Array2<f64> = ndarray::arr2(&[
            [1890.3, -1705.6, -315.8, 3000.3],
            [-1705.6, 1538.3, 284.9, -2706.6],
            [-315.8, 284.9, 52.5, -501.2],
            [3000.3, -2706.6, -501.2, 4760.8],
        ]);
        b.iter(|| {
            black_box(a.mod_cholesky_gmw81());
        });
    }

    #[bench]
    fn gmw81_12x12_nd(b: &mut Bencher) {
        let dim = 12;
        let q1: ndarray::Array2<f64> = random_householder(dim, 2);
        let q2: ndarray::Array2<f64> = random_householder(dim, 9);
        let q3: ndarray::Array2<f64> = random_householder(dim, 90);
        let d: ndarray::Array2<f64> = random_diagonal(dim, (-1.0, 1000.0), 1, 23);
        let tmp = q1.dot(&q2.dot(&q3));
        let a = tmp.dot(&d.dot(&tmp.t()));
        b.iter(|| {
            black_box(a.mod_cholesky_gmw81());
        });
    }

    #[bench]
    fn gmw81_128x128_nd(b: &mut Bencher) {
        let dim = 128;
        let q1: ndarray::Array2<f64> = random_householder(dim, 2);
        let q2: ndarray::Array2<f64> = random_householder(dim, 9);
        let q3: ndarray::Array2<f64> = random_householder(dim, 90);
        let d: ndarray::Array2<f64> = random_diagonal(dim, (-1.0, 1000.0), 1, 23);
        let tmp = q1.dot(&q2.dot(&q3));
        let a = tmp.dot(&d.dot(&tmp.t()));
        b.iter(|| {
            black_box(a.mod_cholesky_gmw81());
        });
    }

    #[bench]
    fn gmw81_256x256_nd(b: &mut Bencher) {
        let dim = 256;
        let q1: ndarray::Array2<f64> = random_householder(dim, 2);
        let q2: ndarray::Array2<f64> = random_householder(dim, 9);
        let q3: ndarray::Array2<f64> = random_householder(dim, 90);
        let d: ndarray::Array2<f64> = random_diagonal(dim, (-1.0, 1000.0), 1, 23);
        let tmp = q1.dot(&q2.dot(&q3));
        let a = tmp.dot(&d.dot(&tmp.t()));
        b.iter(|| {
            black_box(a.mod_cholesky_gmw81());
        });
    }

    #[bench]
    fn se90_3x3_nd(b: &mut Bencher) {
        let a: ndarray::Array2<f64> =
            ndarray::arr2(&[[1.0, 1.0, 2.0], [1.0, 1.0, 3.0], [2.0, 3.0, 1.0]]);
        b.iter(|| {
            black_box(a.mod_cholesky_se90());
        });
    }

    #[bench]
    fn se90_4x4_nd(b: &mut Bencher) {
        let a: ndarray::Array2<f64> = ndarray::arr2(&[
            [1890.3, -1705.6, -315.8, 3000.3],
            [-1705.6, 1538.3, 284.9, -2706.6],
            [-315.8, 284.9, 52.5, -501.2],
            [3000.3, -2706.6, -501.2, 4760.8],
        ]);
        b.iter(|| {
            black_box(a.mod_cholesky_se90());
        });
    }

    #[bench]
    fn se90_12x12_nd(b: &mut Bencher) {
        let dim = 12;
        let q1: ndarray::Array2<f64> = random_householder(dim, 2);
        let q2: ndarray::Array2<f64> = random_householder(dim, 9);
        let q3: ndarray::Array2<f64> = random_householder(dim, 90);
        let d: ndarray::Array2<f64> = random_diagonal(dim, (-1.0, 1000.0), 1, 23);
        let tmp = q1.dot(&q2.dot(&q3));
        let a = tmp.dot(&d.dot(&tmp.t()));
        b.iter(|| {
            black_box(a.mod_cholesky_se90());
        });
    }

    #[bench]
    fn se90_128x128_nd(b: &mut Bencher) {
        let dim = 128;
        let q1: ndarray::Array2<f64> = random_householder(dim, 2);
        let q2: ndarray::Array2<f64> = random_householder(dim, 9);
        let q3: ndarray::Array2<f64> = random_householder(dim, 90);
        let d: ndarray::Array2<f64> = random_diagonal(dim, (-1.0, 1000.0), 1, 23);
        let tmp = q1.dot(&q2.dot(&q3));
        let a = tmp.dot(&d.dot(&tmp.t()));
        b.iter(|| {
            black_box(a.mod_cholesky_se90());
        });
    }

    #[bench]
    fn se90_256x256_nd(b: &mut Bencher) {
        let dim = 256;
        let q1: ndarray::Array2<f64> = random_householder(dim, 2);
        let q2: ndarray::Array2<f64> = random_householder(dim, 9);
        let q3: ndarray::Array2<f64> = random_householder(dim, 90);
        let d: ndarray::Array2<f64> = random_diagonal(dim, (-1.0, 1000.0), 1, 23);
        let tmp = q1.dot(&q2.dot(&q3));
        let a = tmp.dot(&d.dot(&tmp.t()));
        b.iter(|| {
            black_box(a.mod_cholesky_se90());
        });
    }

    #[bench]
    fn se99_3x3_nd(b: &mut Bencher) {
        let a: ndarray::Array2<f64> =
            ndarray::arr2(&[[1.0, 1.0, 2.0], [1.0, 1.0, 3.0], [2.0, 3.0, 1.0]]);
        b.iter(|| {
            black_box(a.mod_cholesky_se99());
        });
    }

    #[bench]
    fn se99_4x4_nd(b: &mut Bencher) {
        let a: ndarray::Array2<f64> = ndarray::arr2(&[
            [1890.3, -1705.6, -315.8, 3000.3],
            [-1705.6, 1538.3, 284.9, -2706.6],
            [-315.8, 284.9, 52.5, -501.2],
            [3000.3, -2706.6, -501.2, 4760.8],
        ]);
        b.iter(|| {
            black_box(a.mod_cholesky_se99());
        });
    }

    #[bench]
    fn se99_12x12_nd(b: &mut Bencher) {
        let dim = 12;
        let q1: ndarray::Array2<f64> = random_householder(dim, 2);
        let q2: ndarray::Array2<f64> = random_householder(dim, 9);
        let q3: ndarray::Array2<f64> = random_householder(dim, 90);
        let d: ndarray::Array2<f64> = random_diagonal(dim, (-1.0, 1000.0), 1, 23);
        let tmp = q1.dot(&q2.dot(&q3));
        let a = tmp.dot(&d.dot(&tmp.t()));
        b.iter(|| {
            black_box(a.mod_cholesky_se99());
        });
    }

    #[bench]
    fn se99_128x128_nd(b: &mut Bencher) {
        let dim = 128;
        let q1: ndarray::Array2<f64> = random_householder(dim, 2);
        let q2: ndarray::Array2<f64> = random_householder(dim, 9);
        let q3: ndarray::Array2<f64> = random_householder(dim, 90);
        let d: ndarray::Array2<f64> = random_diagonal(dim, (-1.0, 1000.0), 1, 23);
        let tmp = q1.dot(&q2.dot(&q3));
        let a = tmp.dot(&d.dot(&tmp.t()));
        b.iter(|| {
            black_box(a.mod_cholesky_se99());
        });
    }

    #[bench]
    fn se99_256x256_nd(b: &mut Bencher) {
        let dim = 256;
        let q1: ndarray::Array2<f64> = random_householder(dim, 2);
        let q2: ndarray::Array2<f64> = random_householder(dim, 9);
        let q3: ndarray::Array2<f64> = random_householder(dim, 90);
        let d: ndarray::Array2<f64> = random_diagonal(dim, (-1.0, 1000.0), 1, 23);
        let tmp = q1.dot(&q2.dot(&q3));
        let a = tmp.dot(&d.dot(&tmp.t()));
        b.iter(|| {
            black_box(a.mod_cholesky_se99());
        });
    }
}
