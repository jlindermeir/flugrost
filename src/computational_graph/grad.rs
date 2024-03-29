use crate::computational_graph::node::{Constant, Node, NodeOutput};
use crate::ndarray::ndarray::{DType, IntoNDArray};
use crate::ndarray::shape::{Rank0, Shape};

pub trait Grad<S: Shape, T: DType> {
    type GradOutput: NodeOutput;
    fn grad(&self, target: &Node<Constant<S, T>>) -> Self::GradOutput;
}

impl<N, S, T> Grad<S, T> for Node<N>
where N: NodeOutput + Grad<S, T>,
      S: Shape,
      T: DType,
{
    type GradOutput = Node<N::GradOutput>;
    fn grad(&self, target: &Node<Constant<S, T>>) -> Self::GradOutput {
        Node(self.0.grad(target))
    }
}

impl<T: DType> Grad<Rank0, T> for Constant<Rank0, T> {
    type GradOutput = Constant<Rank0, T>;

    fn grad(&self, target: &Node<Constant<Rank0, T>>) -> Self::GradOutput {
        if self == &target.0 {
            Constant::new(T::one().into_array())
        } else {
            Constant::new(T::zero().into_array())
        }
    }
}

