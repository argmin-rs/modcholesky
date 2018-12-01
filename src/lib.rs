// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Modified Cholesky decompositions
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

// #![feature(uniform_paths)]
// #![feature(use_extern_macros)]

mod gershgorin;

use failure::Error;
// use gershgorin::*;

pub trait ModCholeskySchnabel1
where
    Self: Sized,
{
    type L;
    fn mod_cholesky_schnabel1(&self) -> Result<Self::L, Error>;
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
    let mut max = num::abs(c[0].clone());
    let mut max_idx = 0;
    c.iter()
        .enumerate()
        .map(|(i, ci)| {
            let ci = num::abs(ci.clone());
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
