use std::ops::{Add, Index, Neg, Sub};
use crate::ndarray::shape::{Const, Rank1, Shape};

pub struct NDArray<T, S: Shape> {
    pub shape: S,
    pub data: Vec<T>
}

fn index_to_i<S: Shape>(shape: &S, strides: &S::Indices, index: S::Indices) -> usize {
    let sizes = shape.shape();
    for (i, idx) in index.into_iter().enumerate() {
        if idx >= sizes[i] {
            panic!("Index {i} out of bounds: index={index:?} shape={shape:?}");
        }
    }
    strides.into_iter().zip(index).map(|(a, b)| a * b).sum()
}

impl<T: Copy, S: Shape> Index<S::Indices> for NDArray<T, S> {
    type Output = T;
    fn index(&self, index: S::Indices) -> &Self::Output {
        let idx: usize = index_to_i(&self.shape, &self.shape.strides(), index);
        &self.data[idx]
    }
}

pub trait IntoNDArray<T, S: Shape> {
    fn into_tensor(self) -> NDArray<T, S>;
}

impl<T: Clone, const M: usize> IntoNDArray<T, Rank1<M>> for [T; M] {
    fn into_tensor(self) -> NDArray<T, Rank1<M>> {
        let shape: Rank1<M> = (Const::<M>, );
        NDArray {
            data: self.to_vec(),
            shape
        }
    }
}

pub fn unary_op<T: Copy, S: Shape>(a: &NDArray<T, S>, op: fn(T) -> T) -> NDArray<T, S> {
    let mut res_data: Vec<T> = Vec::with_capacity(a.shape.n_elements());

    for i in 0..a.shape.n_elements() {
        res_data.push(op(a.data[i]))
    }

    NDArray {
        data: res_data,
        shape: a.shape
    }
}

fn binary_op<T: Copy, S: Shape>(a: &NDArray<T, S>, b: &NDArray<T, S>, op: fn(T, T) -> T) -> NDArray<T, S> {
    let mut res_data: Vec<T> = Vec::with_capacity(a.shape.n_elements());

    for i in 0..a.shape.n_elements() {
        res_data.push(op(a.data[i], b.data[i]))
    }

    NDArray {
        data: res_data,
        shape: a.shape
    }
}

impl<T: Copy + Add<Output = T>, S: Shape> Add for &NDArray<T, S> {
    type Output = NDArray<T, S>;

    fn add(self, rhs: Self) -> Self::Output {
        binary_op(&self, &rhs, |a, b| a + b)
    }
}

impl<T: Copy + Sub<Output = T>, S: Shape> Sub for &NDArray<T, S> {
    type Output = NDArray<T, S>;

    fn sub(self, rhs: Self) -> Self::Output {
        binary_op(&self, &rhs, |a, b| a - b)
    }
}

impl<T: Copy + Neg<Output = T>, S: Shape> Neg for &NDArray<T, S> {
    type Output = NDArray<T, S>;

    fn neg(self) -> Self::Output {
        unary_op(&self, |a| -a)
    }
}

// pub fn mat_mul<T, M: Dim, const K: usize, N: Dim>(
//     lhs: &NDArray<T, (M, Const<K>)>,
//     rhs: &NDArray<T, (Const<K>, N)>
// ) -> NDArray<T, (M, N)>
// where T: Copy + AddAssign + Mul<Output = T> {
//     let result_shape: (M, N) = (Const::<M>, Const::<N>);
//     let result_array = Vec::with_capacity(result_shape.n_elements());
//
//     for i in 0..result_shape.shape()[0] {
//         for j in 0..result_shape.shape()[1] {
//             let mut element: T;
//             for k in 0..lhs.shape.shape()[1] {
//                 element += lhs[[i, k]] * rhs[[k, j]]
//             }
//             result_array[index_to_i(&result_shape, &result_shape.strides(), [i, j])] = element;
//         }
//     }
//
//     NDArray {
//         data: result_array,
//         shape: result_shape
//     }
// }