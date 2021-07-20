[![Build Status](https://travis-ci.org/argmin-rs/modcholesky.svg?branch=master)](https://travis-ci.org/argmin-rs/modcholesky)
[![modcholesky CI](https://github.com/argmin-rs/modcholesky/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/argmin-rs/modcholesky/actions/workflows/ci.yml)
[![Gitter chat](https://badges.gitter.im/argmin-rs/community.png)](https://gitter.im/argmin-rs/community)

# Modified Cholesky decompositions

Given a symmetric matrix A which is potentially not positive definite, a modified Cholesky algorithm obtains the Cholesky decomposition `LL^T` of the positive definite matrix `P(A + E)P^T` where `E` is symmetric and `>= 0`, `P` is a permutation matrix and `L` is lower triangular.
If `A` is already positive definite, then `E = 0`.
The perturbation `E` should be as small as possible for `A + E` to be "sufficiently positive definite".
This is used in optimization methods where indefinite Hessians can be problematic.

This crate implements the algorithms by Gill, Murray and Wright (GMW81) and Schnabel and Eskow (SE90 and SE99).
All algorithms are currently based on `ndarray` but will also be implemented for `nalgebra` in the future.

[Documentation](https://argmin-rs.github.io/modcholesky/modcholesky/)


## Usage

Add this to your `Cargo.toml`:

```
[dependencies]
modcholesky = "0.1.3"
```

## References

* Philip E. Gill, Walter Murray and Margaret H. Wright.
  Practical Optimization.
  Emerald Group Publishing Limited. ISBN 978-0122839528. 1982
* Semyon Aranovich Gershgorin.
  Über die Abgrenzung der Eigenwerte einer Matrix.
  Izv. Akad. Nauk. USSR Otd. Fiz.-Mat. Nauk, 6: 749–754, 1931.
* Robert B. Schnabel and Elizabeth Eskow.
  A new modified Cholesky factorization.
  SIAM J. Sci. Stat. Comput. Vol. 11, No. 6, pp. 1136-1158, November 1990
* Elizabeth Eskow and Robert B. Schnabel.
  Algorithm 695: Software for a new modified Cholesky factorization.
  ACM Trans. Math. Softw. Vol. 17, p. 306-312, 1991
* Sheung Hun Cheng and Nicholas J. Higham.
  A modified Cholesky algorithm based on a symmetric indefinite factorization.
  SIAM J. Matrix Anal. Appl. Vol. 19, No. 4, pp. 1097-1110, October 1998
* Robert B. Schnabel and Elizabeth Eskow.
  A revised modified Cholesky factorization.
  SIAM J. Optim. Vol. 9, No. 4, pp. 1135-1148, 1999
* Jorge Nocedal and Stephen J. Wright.
  Numerical Optimization.
  Springer. ISBN 0-387-30303-0. 2006.

## License

Licensed under either of

  * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
  * MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.


### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
