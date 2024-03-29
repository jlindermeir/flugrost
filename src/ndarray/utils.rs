use crate::ndarray::ndarray::{DType, NDArray};
use crate::ndarray::shape::{Const, Rank2, Shape};

impl<const N: usize, T: DType> NDArray<T, Rank2<N, N>> {
    pub fn eye() -> Self {
        let mut data = vec![T::zero(); N * N];
        for i in 0..N {
            data[i * N + i] = T::one();
        }
        NDArray { shape: (Const::<N>, Const::<N>), data }
    }
}

impl <T: DType, S: Shape> NDArray<T, S> {
    pub fn zeros(shape: S) -> Self {
        let data = vec![T::zero(); shape.n_elements()];
        NDArray { shape, data }
    }

    pub fn ones(shape: S) -> Self {
        let data = vec![T::one(); shape.n_elements()];
        NDArray { shape, data }
    }
}
