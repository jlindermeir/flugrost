use std::ops::{Add, Div, Mul, Neg, Sub};
use crate::ndarray::ndarray::{DType, NDArray};
use crate::ndarray::shape::{Const, Rank2, Shape};

pub fn unary_op<T: DType, S: Shape>(a: &NDArray<T, S>, op: fn(T) -> T) -> NDArray<T, S> {
    let mut res_data: Vec<T> = Vec::with_capacity(a.shape.n_elements());

    for i in 0..a.shape.n_elements() {
        res_data.push(op(a.data[i]))
    }

    NDArray {
        data: res_data,
        shape: a.shape
    }
}

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

impl<T: DType, S: Shape> Neg for &NDArray<T, S> {
    type Output = NDArray<T, S>;

    fn neg(self) -> Self::Output {
        unary_op(&self, |a| -a)
    }
}

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