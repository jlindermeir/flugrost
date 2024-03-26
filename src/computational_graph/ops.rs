use std::ops::{Add, Sub, Neg, Mul};
use crate::computational_graph::node::{Node, NodeOutput};
use crate::ndarray::ndarray::{DType, NDArray};
use crate::ndarray::shape::Shape;

pub struct AddNode<S, T, L, R>
    where S: Shape,
          T: DType,
          L: NodeOutput<Output = NDArray<T, S>>,
          R: NodeOutput<Output = NDArray<T, S>>,
{
    pub lhs: L,
    pub rhs: R,
}

impl<S, T, L, R> NodeOutput for AddNode<S, T, L, R>
    where S: Shape,
          T: DType,
          L: NodeOutput<Output = NDArray<T, S>>,
          R: NodeOutput<Output = NDArray<T, S>>
{
    type Output = NDArray<T, S>;

    fn output(&self) -> Self::Output {
        &self.lhs.output() + &self.rhs.output()
    }
}


impl<S, T, L, R> Add<Node<R>> for Node<L>
    where S: Shape,
          T: DType,
          L: NodeOutput<Output = NDArray<T, S>>,
          R: NodeOutput<Output = NDArray<T, S>>
{
    type Output = Node<AddNode<S, T, L, R>>;

    fn add(self, rhs: Node<R>) -> Self::Output {
        let output = AddNode {
            lhs: self.0,
            rhs: rhs.0,
        };
        Node(output)
    }
}


pub struct NegNode<S, T, N>
    where S: Shape,
          T: DType,
          N: NodeOutput<Output = NDArray<T, S>>,
{
    pub node: N,
}

impl<S, T, N> NodeOutput for NegNode<S, T, N>
    where S: Shape,
          T: DType,
          N: NodeOutput<Output = NDArray<T, S>>
{
    type Output = NDArray<T, S>;

    fn output(&self) -> Self::Output {
        -&self.node.output()
    }
}

impl<S, T, N> Neg for Node<N>
    where S: Shape,
          T: DType,
          N: NodeOutput<Output = NDArray<T, S>>
{
    type Output = Node<NegNode<S, T, N>>;

    fn neg(self) -> Self::Output {
        let output = NegNode {
            node: self.0,
        };
        Node(output)
    }
}

impl <S, T, L, R> Sub<Node<R>> for Node<L>
    where S: Shape,
          T: DType,
          L: NodeOutput<Output = NDArray<T, S>>,
          R: NodeOutput<Output = NDArray<T, S>>
{
    type Output = Node<AddNode<S, T, L, NegNode<S, T, R>>>;

    fn sub(self, rhs: Node<R>) -> Self::Output {
        let output = AddNode {
            lhs: self.0,
            rhs: NegNode {
                node: rhs.0,
            },
        };
        Node(output)
    }
}

pub struct MulNode<S, T, L, R>
    where S: Shape,
          T: DType,
          L: NodeOutput<Output = NDArray<T, S>>,
          R: NodeOutput<Output = NDArray<T, S>>,
{
    pub lhs: L,
    pub rhs: R,
}

impl<S, T, L, R> NodeOutput for MulNode<S, T, L, R>
    where S: Shape,
          T: DType,
          L: NodeOutput<Output = NDArray<T, S>>,
          R: NodeOutput<Output = NDArray<T, S>>
{
    type Output = NDArray<T, S>;

    fn output(&self) -> Self::Output {
        &self.lhs.output() * &self.rhs.output()
    }
}

impl<S, T, L, R> Mul<Node<R>> for Node<L>
    where S: Shape,
          T: DType,
          L: NodeOutput<Output = NDArray<T, S>>,
          R: NodeOutput<Output = NDArray<T, S>>
{
    type Output = Node<MulNode<S, T, L, R>>;

    fn mul(self, rhs: Node<R>) -> Self::Output {
        let output = MulNode {
            lhs: self.0,
            rhs: rhs.0,
        };
        Node(output)
    }
}
