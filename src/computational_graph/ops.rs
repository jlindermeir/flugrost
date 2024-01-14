use std::ops::{Add, Sub, Mul, Div};
use crate::computational_graph::node::{Node, NodeOutput};
use crate::ndarray::ndarray::{DType, NDArray};
use crate::ndarray::shape::Shape;

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
            self.result = Some((self.op)(self.lhs.output(), self.rhs.output()));
        }
        self.result.as_ref().unwrap()
    }
}

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
