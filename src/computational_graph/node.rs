use crate::ndarray::ndarray::{DType, NDArray};
use crate::ndarray::shape::Shape;

pub trait NodeOutput {
    type Output;
    fn output(&mut self) -> &Self::Output;
}

pub struct Constant<S, T>
where S: Shape, T: DType {
    pub array: NDArray<T, S>
}

impl<S, T> NodeOutput for Constant<S, T>
where S: Shape, T: DType {
    type Output = NDArray<T, S>;

    fn output(&mut self) -> &Self::Output {
        &self.array
    }
}

pub struct Node<N>(pub N)
where N: NodeOutput;

impl<N> NodeOutput for Node<N>
where N: NodeOutput {
    type Output = N::Output;

    fn output(&mut self) -> &Self::Output {
        self.0.output()
    }
}
