use std::ops::Add;
use crate::computational_graph::node::{BinaryElementwiseOp, Node, NodeOutput};
use crate::ndarray::ndarray::{DType, NDArray};
use crate::ndarray::shape::Shape;

impl<S, T, L, R> Add<Node<R>> for Node<L>
    where S: Shape,
          T: DType,
          L: NodeOutput<Output = NDArray<T, S>>,
          R: NodeOutput<Output = NDArray<T, S>>
{
    type Output = Node<BinaryElementwiseOp<S, T, L, R>>;

    fn add(self, rhs: Node<R>) -> Self::Output {
        let output = BinaryElementwiseOp {
            lhs: self.0,
            rhs: rhs.0,
            op: |a, b| a + b,
            result: None
        };
        Node(output)
    }
}