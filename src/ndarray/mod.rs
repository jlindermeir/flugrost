use std::fmt::{Debug, Display};
use std::ops::{Add, Index};

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

impl Dim for usize {
    fn size(&self) -> usize {
        *self
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

impl Shape for () {
    const N_DIMS: usize = 0;
    type Indices = [usize; 0];
    fn shape(&self) -> [usize; 0] {
        []
    }
    fn strides(&self) -> [usize; 0] {
        []
    }
}

impl<D1: Dim> Shape for (D1, ) {
    const N_DIMS: usize = 1;
    type Indices = [usize; 1];
    fn shape(&self) -> [usize; 1] {
        [self.0.size()]
    }
    fn strides(&self) -> [usize; 1] {
        [1]
    }
}

impl<D1: Dim, D2: Dim> Shape for (D1, D2) {
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

pub fn add<T: Copy + Add<Output = T>, S: Shape>(a: &NDArray<T, S>, b: &NDArray<T, S>) -> NDArray<T, S> {
    let mut res_data: Vec<T> = Vec::with_capacity(a.shape.n_elements());

    for i in 0..a.shape.n_elements() {
        res_data.push(a.data[i] + b.data[i])
    }

    NDArray {
        data: res_data,
        shape: a.shape
    }
}