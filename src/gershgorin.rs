// Copyright 2018 Stefan Kroboth
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://apache.org/licenses/LICENSE-2.0> or the MIT license <LICENSE-MIT or
// http://opensource.org/licenses/MIT>, at your option. This file may not be
// copied, modified, or distributed except according to those terms.

/// Gershgorin circles
///
/// # References
///
/// * Semyon Aranovich Gerschgorin.
///   Über die Abgrenzung der Eigenwerte einer Matrix.
///   Izv. Akad. Nauk. USSR Otd. Fiz.-Mat. Nauk, 6: 749–754, 1931.
pub trait GershgorinCircles {
    /// Computes the Gershgorin Circles of a matrix
    fn gershgorin_circles(&self) -> Vec<(f64, f64)>;
}

impl GershgorinCircles for ndarray::Array2<f64> {
    /// Computes the Gershgorin Circles of a matrix
    fn gershgorin_circles(&self) -> Vec<(f64, f64)> {
        debug_assert!(self.is_square());
        // use ndarray::s;
        let n = self.raw_dim()[0];
        let mut out: Vec<(f64, f64)> = Vec::with_capacity(n);
        for i in 0..n {
            // TODO: do this with slices instead of loops
            let aii = self[(i, i)];
            let mut ri = 0.0;
            let mut ci = 0.0;
            for j in 0..n {
                if i == j {
                    continue;
                }
                ri += self[(i, j)].abs();
                ci += self[(j, i)].abs();
            }
            out.push((aii, ri.min(ci)));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_gershgorin_circles() {
        use super::GershgorinCircles;
        let a: ndarray::Array2<f64> = ndarray::arr2(&[
            [10.0, -1.0, 0.0, 1.0],
            [0.2, 8.0, 0.2, 0.2],
            [1.0, 1.0, 2.0, 1.0],
            [-1.0, -1.0, -1.0, -11.0],
        ]);
        // without considering the columns as well
        // let b: Vec<(f64, f64)> = vec![(10.0, 2.0), (8.0, 0.6), (2.0, 3.0), (-11.0, 3.0)];
        // with considering the columns
        let b: Vec<(f64, f64)> = vec![(10.0, 2.0), (8.0, 0.6), (2.0, 1.2), (-11.0, 2.2)];
        let res = a.gershgorin_circles();
        b.iter()
            .zip(res.iter())
            .map(|((x1, y1), (x2, y2))| {
                assert!((x1 - x2).abs() < 2.0 * std::f64::EPSILON);
                assert!((y1 - y2).abs() < 2.0 * std::f64::EPSILON);
            })
            .count();
    }
}
