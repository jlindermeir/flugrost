use std::fmt::Debug;
use std::ops::Index;

#[derive(Debug, Copy, Clone)]
pub struct Const<const N: usize>;

pub trait Dim: Debug + Copy {
    fn size(&self) -> usize;
}

impl<const N: usize> Dim for Const<N> {
    fn size(&self) -> usize {
        N
    }
}

pub trait Shape: Debug + Copy {
    const N_DIMS: usize;
    type Indices: IntoIterator<Item = usize> + Index<usize, Output = usize> + Debug + Copy;
    fn shape(&self) -> Self::Indices;
    fn strides(&self) -> Self::Indices;
    fn n_elements(&self) -> usize {
        self.shape().into_iter().product()
    }
}

pub type Rank0 = ();
pub type Rank1<const M: usize> = (Const<M>,);
pub type Rank2<const M: usize, const N: usize> = (Const<M>, Const<N>);

impl Shape for Rank0 {
    const N_DIMS: usize = 0;
    type Indices = [usize; 0];
    fn shape(&self) -> [usize; 0] {
        []
    }
    fn strides(&self) -> [usize; 0] {
        []
    }
}

impl<const M: usize> Shape for Rank1<M> {
    const N_DIMS: usize = 1;
    type Indices = [usize; 1];
    fn shape(&self) -> [usize; 1] {
        [self.0.size()]
    }
    fn strides(&self) -> [usize; 1] {
        [1]
    }
}

impl<const M: usize, const N: usize> Shape for Rank2<M, N> {
    const N_DIMS: usize = 2;
    type Indices = [usize; 2];
    fn shape(&self) -> [usize; 2] {
        [self.0.size(), self.1.size()]
    }
    fn strides(&self) -> [usize; 2] {
        [self.1.size(), 1]
    }
}