use std::ops::{Add, Div, Index, Mul, Neg, Sub};
use crate::ndarray::shape::{Const, Rank0, Rank1, Rank2, Shape};

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
    fn into_array(self) -> NDArray<T, S>;
}

impl<T: Clone + Div> IntoNDArray<T, Rank0> for T {
    fn into_array(self) -> NDArray<T, Rank0> {
        NDArray {
            data: [self].to_vec(),
            shape: ()
        }
    }
}

impl<T: Copy + Div, const M: usize> IntoNDArray<T, Rank1<M>> for [T; M] {
    fn into_array(self) -> NDArray<T, Rank1<M>> {
        let shape: Rank1<M> = (Const::<M>, );
        NDArray {
            data: self.to_vec(),
            shape
        }
    }
}

impl<T: Copy + Div, const M: usize, const N: usize> IntoNDArray<T, Rank2<M, N>> for [[T; N]; M] {
    fn into_array(self) -> NDArray<T, Rank2<M, N>> {
        let shape = (Const::<M>, Const::<N>);
        let data = self.iter().flat_map(|v| v.iter().copied()).collect();
        NDArray {
            data,
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

impl<T: Copy + Mul<Output = T>, S: Shape> Mul for &NDArray<T, S> {
    type Output = NDArray<T, S>;

    fn mul(self, rhs: Self) -> Self::Output {
        binary_op(&self, &rhs, |a, b| a * b)
    }
}

pub fn mat_mul<T, const M: usize, const K: usize, const N: usize>(
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