use std::fmt::{Display, Formatter};
use std::ops::{Add, Div, Index, Mul, Neg, Sub};
use crate::ndarray::shape::{Const, Rank0, Rank1, Rank2, Shape};

pub trait DType: Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Display {

    fn zero() -> Self;
    fn one() -> Self;

}
impl DType for i32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}
impl DType for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}
impl DType for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

pub struct NDArray<T: DType, S: Shape> {
    pub shape: S,
    pub data: Vec<T>
}

impl <T: DType> Display for NDArray<T, Rank0> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\n", self.data[0])
    }
}

impl<T: DType, const M: usize> Display for NDArray<T, Rank1<M>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        s.push_str("[");
        for i in 0..M {
            s.push_str(&format!("{}", self[[i]]));
            if i != M - 1 {
                s.push_str(", ");
            }
        }
        s.push_str("]\n");
        write!(f, "{}", s)
    }
}

impl<T: DType, const M: usize, const N: usize> Display for NDArray<T, Rank2<M, N>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut s = String::new();
        s.push_str("[");
        for i in 0..M {
            s.push_str("[");
            for j in 0..N {
                s.push_str(&format!("{}", self[[i, j]]));
                if j != N - 1 {
                    s.push_str(", ");
                }
            }
            s.push_str("]");
            if i != M - 1 {
                s.push_str(",\n");
            }
        }
        s.push_str("]\n");
        write!(f, "{}", s)
    }
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

impl<T: DType, S: Shape> Index<S::Indices> for NDArray<T, S> {
    type Output = T;
    fn index(&self, index: S::Indices) -> &Self::Output {
        let idx: usize = index_to_i(&self.shape, &self.shape.strides(), index);
        &self.data[idx]
    }
}

impl<T: DType, S: Shape> Clone for NDArray<T, S> {
    fn clone(&self) -> Self {
        NDArray {
            shape: self.shape.clone(),
            data: self.data.clone()
        }
    }
}

pub trait IntoNDArray<T: DType, S: Shape> {
    fn into_array(self) -> NDArray<T, S>;
}

impl<T: DType> IntoNDArray<T, Rank0> for T {
    fn into_array(self) -> NDArray<T, Rank0> {
        NDArray {
            data: [self].to_vec(),
            shape: ()
        }
    }
}

impl<T:DType, const M: usize> IntoNDArray<T, Rank1<M>> for [T; M] {
    fn into_array(self) -> NDArray<T, Rank1<M>> {
        let shape: Rank1<M> = (Const::<M>, );
        NDArray {
            data: self.to_vec(),
            shape
        }
    }
}

impl<T: DType, const M: usize, const N: usize> IntoNDArray<T, Rank2<M, N>> for [[T; N]; M] {
    fn into_array(self) -> NDArray<T, Rank2<M, N>> {
        let shape = (Const::<M>, Const::<N>);
        let data = self.iter().flat_map(|v| v.iter().copied()).collect();
        NDArray {
            data,
            shape
        }
    }
}
