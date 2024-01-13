use std::ops::{Add, Div, Mul, Neg, Sub};
use crate::ndarray::ndarray::{DType, NDArray};
use crate::ndarray::shape::{Const, Rank2, Shape};

macro_rules! implement_unary_op {
    ($trait: ident, $method:ident, $op:tt) => {
        impl<T: DType, S: Shape> $trait for &NDArray<T, S> {
            type Output = NDArray<T, S>;

            fn $method(self) -> Self::Output {
                let n_elements = self.shape.n_elements();
                let mut res_data: Vec<T> = Vec::with_capacity(n_elements);

                for i in 0..n_elements {
                    res_data.push($op self.data[i])
                }

                NDArray {
                    data: res_data,
                    shape: self.shape
                }
            }
        }
    };
}

implement_unary_op!(Neg, neg, -);

macro_rules! implement_binary_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T: DType, S: Shape> $trait for &NDArray<T, S> {
            type Output = NDArray<T, S>;

            fn $method(self, rhs: &NDArray<T, S>) -> Self::Output {
                let mut res_data: Vec<T> = Vec::with_capacity(self.shape.n_elements());

                for i in 0..self.shape.n_elements() {
                    res_data.push(self.data[i] $op rhs.data[i]);
                }

                NDArray {
                    data: res_data,
                    shape: self.shape
                }
            }
        }
    };
}

implement_binary_op!(Add, add, +);
implement_binary_op!(Sub, sub, -);
implement_binary_op!(Mul, mul, *);
implement_binary_op!(Div, div, /);

pub fn mat_mul<T: DType, const M: usize, const K: usize, const N: usize>(
    lhs: &NDArray<T, Rank2<M, K>>,
    rhs: &NDArray<T, Rank2<K, N>>
) -> NDArray<T, Rank2<M, N>>
    where T: Copy + std::iter::Sum + Mul<Output = T> {
    let result_shape = (Const::<M>, Const::<N>);
    let mut result_array = Vec::new();

    for i in 0..M {
        for j in 0..N {
            let element = (0..K).map(|k| lhs[[i, k]] * rhs[[k, j]]).sum();
            result_array.push(element);
        }
    }

    NDArray {
        data: result_array,
        shape: result_shape
    }
}