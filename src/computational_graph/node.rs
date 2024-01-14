use log::{debug, info};
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

pub struct BinaryElementwiseOp<S, T, L, R>
where S: Shape,
      T: DType,
      L: NodeOutput<Output = NDArray<T, S>>,
      R: NodeOutput<Output = NDArray<T, S>>,
{
    pub lhs: L,
    pub rhs: R,
    pub op: fn(&NDArray<T, S>, &NDArray<T, S>) -> NDArray<T, S>,
    pub result: Option<NDArray<T, S>>
}

impl<S, T, L, R> NodeOutput for BinaryElementwiseOp<S, T, L, R>
where S: Shape,
      T: DType,
      L: NodeOutput<Output = NDArray<T, S>>,
      R: NodeOutput<Output = NDArray<T, S>>
{
    type Output = NDArray<T, S>;

    fn output(&mut self) -> &Self::Output {
        if self.result.is_none() {
            println!("Computing binary elementwise op");
            self.result = Some((self.op)(self.lhs.output(), self.rhs.output()));
        } else {
            println!("Binary elementwise op already computed");
        }
        self.result.as_ref().unwrap()
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
