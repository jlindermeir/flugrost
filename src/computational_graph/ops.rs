use crate::computational_graph::node::{BinaryElementwiseOp, Node};
use crate::ndarray::ndarray::{DType, NDArray};
use crate::ndarray::shape::Shape;

pub fn add<T, S, L, R>(lhs: L, rhs: R) -> BinaryElementwiseOp<S, T, L, R>
where T: DType,
      S: Shape,
      L: Node<Output = NDArray<T, S>>,
      R: Node<Output = NDArray<T, S>>,
{
    BinaryElementwiseOp {
        lhs,
        rhs,
        op: |a, b| a + b,
        result: None
    }
}