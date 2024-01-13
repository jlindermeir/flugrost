use std::fmt::{Display, Formatter};
use std::ops::{Div, Index};
use crate::ndarray::shape::{Const, Rank0, Rank1, Rank2, Shape};

pub struct NDArray<T, S: Shape> {
    pub shape: S,
    pub data: Vec<T>
}

impl <T: Display + Copy> Display for NDArray<T, Rank0> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\n", self.data[0])
    }
}

impl<T: Display + Copy, const M: usize> Display for NDArray<T, Rank1<M>> {
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

impl<T: Display + Copy, const M: usize, const N: usize> Display for NDArray<T, Rank2<M, N>> {
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
