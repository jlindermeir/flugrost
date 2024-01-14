use std::ops::Add;
use crate::ndarray::ndarray::{DType, NDArray};
use crate::ndarray::shape::Shape;

pub trait Node {
    type Output;
    fn output(&mut self) -> &Self::Output;
}

pub struct Constant<S, T>
where S: Shape, T: DType {
    pub array: NDArray<T, S>
}

impl<S, T> Node for Constant<S, T>
where S: Shape, T: DType {
    type Output = NDArray<T, S>;

    fn output(&mut self) -> &Self::Output {
        &self.array
    }
}

pub struct BinaryElementwiseOp<S, T, L, R>
where S: Shape,
      T: DType,
      L: Node<Output = NDArray<T, S>>,
      R: Node<Output = NDArray<T, S>>,
{
    pub lhs: L,
    pub rhs: R,
    pub op: fn(&NDArray<T, S>, &NDArray<T, S>) -> NDArray<T, S>,
    pub result: Option<NDArray<T, S>>
}

impl<S, T, L, R> Node for BinaryElementwiseOp<S, T, L, R>
where S: Shape,
      T: DType,
      L: Node<Output = NDArray<T, S>>,
      R: Node<Output = NDArray<T, S>>
{
    type Output = NDArray<T, S>;

    fn output(&mut self) -> &Self::Output {
        if self.result.is_none() {
            self.result = Some((self.op)(self.lhs.output(), self.rhs.output()));
        }
        self.result.as_ref().unwrap()
    }
}
