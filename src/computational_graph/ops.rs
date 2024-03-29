use std::ops::{Add, Sub, Neg, Mul};
use crate::computational_graph::node::{Node, NodeOutput};
use crate::ndarray::ndarray::{DType, NDArray};
use crate::ndarray::shape::Shape;

pub struct AddNode<'a, S, T, L, R>
    where S: Shape,
          T: DType,
          L: NodeOutput<Output = NDArray<T, S>>,
          R: NodeOutput<Output = NDArray<T, S>>,
{
    pub lhs: &'a L,
    pub rhs: &'a R,
}

impl<'a, S, T, L, R> NodeOutput for AddNode<'a, S, T, L, R>
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


impl<'a, S, T, L, R> Add<&'a Node<R>> for &'a Node<L>
    where S: Shape,
          T: DType,
          L: NodeOutput<Output = NDArray<T, S>> + 'a,
          R: NodeOutput<Output = NDArray<T, S>> + 'a
{
    type Output = Node<AddNode<'a, S, T, L, R>>;

    fn add(self, rhs: &'a Node<R>) -> Self::Output {
        let output = AddNode {
            lhs: &self.0,
            rhs: &rhs.0,
        };
        Node(output)
    }
}

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

pub struct SubNode<'a, S, T, L, R>
    where S: Shape,
          T: DType,
          L: NodeOutput<Output = NDArray<T, S>>,
          R: NodeOutput<Output = NDArray<T, S>>,
{
    pub lhs: &'a L,
    pub rhs: &'a R,
}

impl<'a, S, T, L, R> NodeOutput for SubNode<'a, S, T, L, R>
    where S: Shape,
          T: DType,
          L: NodeOutput<Output = NDArray<T, S>>,
          R: NodeOutput<Output = NDArray<T, S>>
{
    type Output = NDArray<T, S>;

    fn output(&self) -> Self::Output {
        &self.lhs.output() - &self.rhs.output()
    }
}

impl<'a, S, T, L, R> Sub<&'a Node<R>> for &'a Node<L>
    where S: Shape,
          T: DType,
          L: NodeOutput<Output = NDArray<T, S>> + 'a,
          R: NodeOutput<Output = NDArray<T, S>> + 'a
{
    type Output = Node<SubNode<'a, S, T, L, R>>;

    fn sub(self, rhs: &'a Node<R>) -> Self::Output {
        let output = SubNode {
            lhs: &self.0,
            rhs: &rhs.0,
        };
        Node(output)
    }
}

pub struct MulNode<'a, S, T, L, R>
    where S: Shape,
          T: DType,
          L: NodeOutput<Output = NDArray<T, S>>,
          R: NodeOutput<Output = NDArray<T, S>>,
{
    pub lhs: &'a L,
    pub rhs: &'a R,
}

impl<'a, S, T, L, R> NodeOutput for MulNode<'a, S, T, L, R>
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

impl<'a, S, T, L, R> Mul<&'a Node<R>> for &'a Node<L>
    where S: Shape,
          T: DType,
          L: NodeOutput<Output = NDArray<T, S>> + 'a,
          R: NodeOutput<Output = NDArray<T, S>> + 'a
{
    type Output = Node<MulNode<'a, S, T, L, R>>;

    fn mul(self, rhs: &'a Node<R>) -> Self::Output {
        let output = MulNode {
            lhs: &self.0,
            rhs: &rhs.0,
        };
        Node(output)
    }
}
