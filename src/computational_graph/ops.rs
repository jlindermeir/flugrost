use std::ops::{Add, Sub, Mul, Div};
use crate::computational_graph::node::{BinaryElementwiseOp, Node, NodeOutput};
use crate::ndarray::ndarray::{DType, NDArray};
use crate::ndarray::shape::Shape;

macro_rules! implement_binary_op {
    ($trait:ident, $method:ident, $op:tt) => {
        /// Implementation for array op array
        impl<S, T, L, R> $trait<Node<R>> for Node<L>
            where S: Shape,
                  T: DType,
                  L: NodeOutput<Output = NDArray<T, S>>,
                  R: NodeOutput<Output = NDArray<T, S>>
        {
            type Output = Node<BinaryElementwiseOp<S, T, L, R>>;

            fn $method(self, rhs: Node<R>) -> Self::Output {
                let output = BinaryElementwiseOp {
                    lhs: self.0,
                    rhs: rhs.0,
                    op: |a, b| a $op b,
                    result: None
                };
                Node(output)
            }
        }
    };
}

implement_binary_op!(Add, add, +);
implement_binary_op!(Sub, sub, -);
implement_binary_op!(Mul, mul, *);
implement_binary_op!(Div, div, /);
