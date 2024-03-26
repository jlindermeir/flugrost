use crate::computational_graph::node::{Constant, NodeOutput};
use crate::ndarray::ndarray::{DType, NDArray};
use crate::ndarray::shape::{Const, Rank1, Rank2, Shape};

pub trait Grad<S: Shape, T: DType> {
    type GradOutput: NodeOutput;
    fn grad(&self, target: &Constant<S, T>) -> Self::GradOutput;
}

impl <const N: usize> Grad<Rank1<N>, f64> for Constant<Rank1<N>, f64>
{
    type GradOutput = Constant<Rank2<N, N>, f64>;

    fn grad(&self, target: &Constant<Rank1<N>, f64>) -> Self::GradOutput {
        if self == target {
            let array: NDArray<f64, Rank2<N, N>> = NDArray::eye();
            Constant::new(array)
        } else {
            let array: NDArray<f64, Rank2<N, N>> = NDArray::zero((Const::<N>, Const::<N>));
            Constant::new(array)
        }
    }
}
