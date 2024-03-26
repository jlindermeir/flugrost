use std::sync::atomic::{AtomicUsize, Ordering};
use crate::ndarray::ndarray::{DType, NDArray};
use crate::ndarray::shape::Shape;

pub trait NodeOutput {
    type Output;
    fn output(&self) -> Self::Output;
}

static ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

pub struct Constant<S, T>
    where
        S: Shape,
        T: DType,
{
    pub id: usize,
    pub array: NDArray<T, S>,
}

impl<S, T> Constant<S, T>
    where
        S: Shape,
        T: DType,
{
    pub fn new(array: NDArray<T, S>) -> Self {
        let id = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        Self { id, array }
    }
}

impl<S, T> PartialEq for Constant<S, T>
    where
        S: Shape,
        T: DType,
{
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<S, T> NodeOutput for Constant<S, T>
where S: Shape, T: DType {
    type Output = NDArray<T, S>;

    fn output(&self) -> Self::Output {
        self.array.clone()
    }
}

pub struct Node<N>(pub N)
where N: NodeOutput;

impl<N> NodeOutput for Node<N>
where N: NodeOutput {
    type Output = N::Output;

    fn output(&self) -> Self::Output {
        self.0.output()
    }
}
