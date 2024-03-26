use crate::ndarray::ndarray::{NDArray};
use crate::ndarray::shape::{Const, Rank2, Shape};

impl<const N: usize> NDArray<f64, Rank2<N, N>> {
    pub fn eye() -> Self {
        let mut data = vec![0.0; N * N];
        for i in 0..N {
            data[i * N + i] = 1.0;
        }
        NDArray { shape: (Const::<N>, Const::<N>), data }
    }
}

impl <S: Shape> NDArray<f64, S> {
    pub fn zero(shape: S) -> Self {
        let data = vec![0.0; shape.n_elements()];
        NDArray { shape, data }
    }

    pub fn one(shape: S) -> Self {
        let data = vec![1.0; shape.n_elements()];
        NDArray { shape, data }
    }
}
