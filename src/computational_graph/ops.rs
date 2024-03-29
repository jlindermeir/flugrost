use std::ops::{Add, Sub, Neg, Mul, Div};
use crate::computational_graph::node::{Node, NodeOutput};
use crate::ndarray::ndarray::{DType, NDArray};
use crate::ndarray::shape::Shape;

macro_rules! implement_binary_op_node {
    ($node_name:ident, $trait:ident, $method:ident, $op:tt) => {
        pub struct $node_name<'a, S, T, L, R>
            where S: Shape,
                  T: DType,
                  L: NodeOutput<Output = NDArray<T, S>>,
                  R: NodeOutput<Output = NDArray<T, S>>,
        {
            pub lhs: &'a L,
            pub rhs: &'a R,
        }

        impl<'a, S, T, L, R> NodeOutput for $node_name<'a, S, T, L, R>
            where S: Shape,
                  T: DType,
                  L: NodeOutput<Output = NDArray<T, S>>,
                  R: NodeOutput<Output = NDArray<T, S>>
        {
            type Output = NDArray<T, S>;

            fn output(&self) -> Self::Output {
                &self.lhs.output() $op &self.rhs.output()
            }
        }

        impl<'a, S, T, L, R> $trait<&'a Node<R>> for &'a Node<L>
            where S: Shape,
                  T: DType,
                  L: NodeOutput<Output = NDArray<T, S>> + 'a,
                  R: NodeOutput<Output = NDArray<T, S>> + 'a
        {
            type Output = Node<$node_name<'a, S, T, L, R>>;

            fn $method(self, rhs: &'a Node<R>) -> Self::Output {
                let output = $node_name {
                    lhs: &self.0,
                    rhs: &rhs.0,
                };
                Node(output)
            }
        }
    };
}

implement_binary_op_node!(AddNode, Add, add, +);
implement_binary_op_node!(SubNode, Sub, sub, -);
implement_binary_op_node!(MulNode, Mul, mul, *);
implement_binary_op_node!(DivNode, Div, div, /);

pub struct NegNode<'a, S, T, N>
    where S: Shape,
          T: DType,
          N: NodeOutput<Output = NDArray<T, S>>,
{
    pub node: &'a N,
}

impl<'a, S, T, N> NodeOutput for NegNode<'a, S, T, N>
    where S: Shape,
          T: DType,
          N: NodeOutput<Output = NDArray<T, S>> + 'a
{
    type Output = NDArray<T, S>;

    fn output(&self) -> Self::Output {
        -&self.node.output()
    }
}

impl<'a, S, T, N> Neg for &'a Node<N>
    where S: Shape,
          T: DType,
          N: NodeOutput<Output = NDArray<T, S>> + 'a
{
    type Output = Node<NegNode<'a, S, T, N>>;

    fn neg(self) -> Self::Output {
        let output = NegNode {
            node: &self.0,
        };
        Node(output)
    }
}
