// Copyright 2018-2020 argmin developers
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

//! Utility functions
use ndarray_rand::RandomExt;
use rand::distributions::Uniform;
use rand::prelude::*;

/// Computes the Eigenvalues of a 2x2 matrix
pub fn eigenvalues_2x2(mat: &ndarray::ArrayView2<f64>) -> (f64, f64) {
    let a = mat[(0, 0)];
    let b = mat[(0, 1)];
    let c = mat[(1, 0)];
    let d = mat[(1, 1)];
    let tmp = ((-(a + d) / 2.0).powi(2) - a * d + b * c).sqrt();
    let l1 = (a + d) / 2.0 + tmp;
    let l2 = (a + d) / 2.0 - tmp;
    if l1 > l2 {
        (l1, l2)
    } else {
        (l2, l1)
    }
}

/// Swaps columns `idx1` and `idx2` of matrix `mat`
pub fn swap_columns<T>(mat: &mut ndarray::Array2<T>, idx1: usize, idx2: usize)
where
    ndarray::OwnedRepr<T>: ndarray::Data,
{
    let s = mat.raw_dim();
    for i in 0..s[0] {
        mat.swap((i, idx1), (i, idx2));
    }
}

/// Swaps rows `idx1` and `idx2` of matrix `mat`
pub fn swap_rows<T>(mat: &mut ndarray::Array2<T>, idx1: usize, idx2: usize)
where
    ndarray::OwnedRepr<T>: ndarray::Data,
{
    let s = mat.raw_dim();
    for i in 0..s[1] {
        mat.swap((idx1, i), (idx2, i));
    }
}

/// Returns the index of the largest element in a 1D array
pub fn index_of_largest<'a, T>(c: &ndarray::ArrayView1<T>) -> usize
where
    <ndarray::ViewRepr<&'a T> as ndarray::RawData>::Elem:
        std::cmp::PartialOrd + num::traits::Signed + Clone,
{
    let mut max = c[0].clone();
    let mut max_idx = 0;
    c.iter()
        .enumerate()
        .map(|(i, ci)| {
            let ci = ci.clone();
            if ci > max {
                max = ci;
                max_idx = i
            }
        })
        .count();
    max_idx
}

/// Returns the index of the element with the largest absolute value in a 1D array
pub fn index_of_largest_abs<'a, T>(c: &ndarray::ArrayView1<T>) -> usize
where
    <ndarray::ViewRepr<&'a T> as ndarray::RawData>::Elem:
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

/// Returns the permutation matrix for a vector of permuted indices
pub fn index_to_permutation_mat(idxs: &[usize]) -> ndarray::Array2<f64> {
    let n = idxs.len();
    let mut mat = ndarray::Array2::zeros((n, n));
    for i in 0..n {
        mat[(i, idxs[i])] = 1.0;
    }
    mat
}

/// Builds a diagonal matrix from a 1D slice
pub fn diag_mat_from_arr<T>(arr: &[T]) -> ndarray::Array2<T>
where
    T: Clone + num::traits::identities::Zero,
{
    let n = arr.len();
    let mut mat: ndarray::Array2<T> = ndarray::Array2::zeros((n, n));
    let diag: ndarray::Array1<T> = ndarray::Array::from_iter(arr.iter().cloned());
    mat.diag_mut().assign(&diag);
    mat
}

/// Returns a random Householder matrix of dimension `dim` and with seed `seed`.
pub fn random_householder(dim: usize, seed: u8) -> ndarray::Array2<f64> {
    let mut rng = rand_xorshift::XorShiftRng::from_seed([seed; 16]);
    let w = ndarray::Array::random_using((dim, 1), Uniform::new_inclusive(-1.0, 1.0), &mut rng);
    let denom = w.fold(0.0, |acc, &x: &f64| acc + x.powi(2));
    ndarray::Array::eye(dim) - 2.0 / denom * w.dot(&w.t())
}

/// Returns a random diagonal matrix with Eigenvalues inbetween `eig_range_min` and
/// `eig_range_max`. The minimum number of negative eigenvalues can be specified with
/// `min_neg_eigenvalues`. Takes a `seed for random values`.
pub fn random_diagonal(
    dim: usize,
    (eig_range_min, eig_range_max): (f64, f64),
    min_neg_eigenvalues: usize,
    seed: u8,
) -> ndarray::Array2<f64> {
    let mut rng = rand_xorshift::XorShiftRng::from_seed([seed; 16]);
    let mut w = ndarray::Array::random_using(
        dim,
        Uniform::new_inclusive(eig_range_min, eig_range_max),
        &mut rng,
    );
    let mut idxs: Vec<usize> = (0..dim).collect();
    for _ in 0..min_neg_eigenvalues {
        let idxidx: usize = (rng.gen::<f64>() * (idxs.len() - 1) as f64).floor() as usize;
        let idx = idxs.remove(idxidx);
        w[idx] = rng.gen::<f64>() - 1.0;
    }
    let mut out = ndarray::Array::eye(dim);
    out.diag_mut().assign(&w);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    #[test]
    fn test_biggest_index_abs() {
        use ndarray::s;
        let j = 1;
        let a: ndarray::Array2<f64> = ndarray::arr2(&[
            [1.0, 2.0, 3.0, 0.0],
            [4.0, 2.0, 6.0, 0.0],
            [7.0, 8.0, 3.0, 0.0],
            [3.0, 4.0, 2.0, -8.0],
        ]);
        let idx = super::index_of_largest_abs(&a.diag().slice(s![j..]));
        assert_eq!(idx + j, 3);
    }

    #[test]
    fn test_rand_householder() {
        let q = random_householder(2, 84);
        let res: ndarray::Array2<f64> = ndarray::arr2(&[
            [-0.0000034067079459632055, -0.9999999999941971],
            [-0.9999999999941971, 0.0000034067079460742278],
        ]);
        assert!(q.abs_diff_eq(&res, 1e-19));
    }

    #[test]
    fn test_random_diagonal_all_pos() {
        let d = random_diagonal(2, (-1.0, 1.0), 0, 128);
        let res: ndarray::Array2<f64> =
            ndarray::arr2(&[[0.003923416145884984, 0.0], [0.0, 0.0039215684018965025]]);
        assert!(d.abs_diff_eq(&res, 1e-19));
    }

    #[test]
    fn test_random_diagonal_one_neg() {
        let d = random_diagonal(2, (-1.0, 1.0), 1, 128);
        let res: ndarray::Array2<f64> =
            ndarray::arr2(&[[-0.49803921207376156, 0.0], [0.0, 0.0039215684018965025]]);
        assert!(d.abs_diff_eq(&res, 1e-19));
    }
}
