use crate::computational_graph::node::{Constant, Node, NodeOutput};
use crate::computational_graph::ops::AddNode;
use crate::ndarray::ndarray::{DType, IntoNDArray, NDArray};
use crate::ndarray::shape::{Rank0, Shape};

pub trait Grad<S: Shape, T: DType, GS: Shape> {
    type GradOutput: NodeOutput<Output = NDArray<T, S>>;
    fn grad(&self, target: &Node<Constant<S, T>>) -> Self::GradOutput;
}

impl<N, S, T, GS> Grad<S, T, GS> for Node<N>
where N: NodeOutput + Grad<S, T, GS>,
      S: Shape,
      T: DType,
      GS: Shape,
{
    type GradOutput = N::GradOutput;
    fn grad(&self, target: &Node<Constant<S, T>>) -> Self::GradOutput {
        self.0.grad(target)
    }
}

impl<T: DType> Grad<Rank0, T, Rank0> for Constant<Rank0, T> {
    type GradOutput = Node<Constant<Rank0, T>>;

    fn grad(&self, target: &Node<Constant<Rank0, T>>) -> Self::GradOutput {
        let grad_result = {
            if self == &target.0 {
                Constant::new(T::one().into_array())
            } else {
                Constant::new(T::zero().into_array())
            }
        };
        Node(grad_result)
    }
}
