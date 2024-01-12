use std::fmt::{Display, Formatter};

struct Const<const N: usize>;

trait Dim {
    fn size(&self) -> usize;
}

impl<const N: usize> Dim for Const<N> {
    fn size(&self) -> usize {
        N
    }
}

pub(crate) trait Shape {
    const N_DIMS: usize;
    fn shape(&self) -> [usize; Self::N_DIMS];
    fn strides(&self) -> [usize; Self::N_DIMS];
}

impl Shape for () {
    const N_DIMS: usize = 0;
    fn shape(&self) -> [usize; Self::N_DIMS] {
        []
    }
    fn strides(&self) -> [usize; Self::N_DIMS] {
        []
    }
}

impl<D1: Dim> Shape for (D1, ) {
    const N_DIMS: usize = 1;
    fn shape(&self) -> [usize; Self::N_DIMS] {
        [self.0.size()]
    }
    fn strides(&self) -> [usize; Self::N_DIMS] {
        [1]
    }
}

impl<D1: Dim, D2: Dim> Shape for (D1, D2) {
    const N_DIMS: usize = 2;
    fn shape(&self) -> [usize; Self::N_DIMS] {
        [self.1.size(), self.0.size()]
    }
    fn strides(&self) -> [usize; Self::N_DIMS] {
        [self.1.size(), 1]
    }
}
