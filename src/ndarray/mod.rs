use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Index, Mul, Neg, Sub};

#[derive(Debug, Copy, Clone)]
pub struct Const<const N: usize>;

trait Dim: Debug + Copy {
    fn size(&self) -> usize;
}

impl<const N: usize> Dim for Const<N> {
    fn size(&self) -> usize {
        N
    }
}

trait Shape: Debug + Copy {
    const N_DIMS: usize;
    type Indices: IntoIterator<Item = usize> + Index<usize, Output = usize> + Debug + Copy;
    fn shape(&self) -> Self::Indices;
    fn strides(&self) -> Self::Indices;
    fn n_elements(&self) -> usize {
        self.shape().into_iter().product()
    }
}

pub type Rank0 = ();
pub type Rank1<const M: usize> = (Const<M>,);
pub type Rank2<const M: usize, const N: usize> = (Const<M>, Const<N>);

impl Shape for Rank0 {
    const N_DIMS: usize = 0;
    type Indices = [usize; 0];
    fn shape(&self) -> [usize; 0] {
        []
    }
    fn strides(&self) -> [usize; 0] {
        []
    }
}

impl<const M: usize> Shape for Rank1<M> {
    const N_DIMS: usize = 1;
    type Indices = [usize; 1];
    fn shape(&self) -> [usize; 1] {
        [self.0.size()]
    }
    fn strides(&self) -> [usize; 1] {
        [1]
    }
}

impl<const M: usize, const N: usize> Shape for Rank2<M, N> {
    const N_DIMS: usize = 2;
    type Indices = [usize; 2];
    fn shape(&self) -> [usize; 2] {
        [self.0.size(), self.1.size()]
    }
    fn strides(&self) -> [usize; 2] {
        [self.1.size(), 1]
    }
}

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
