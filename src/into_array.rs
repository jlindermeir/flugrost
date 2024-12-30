use crate::dtype::DType;
use crate::ndarray::NDArray;
use crate::shape::{Rank0, Shape};

trait IntoNDArray<S: Shape, T: DType> {
    fn into_ndarray(self) -> NDArray<S, T>;
}

impl<T: DType> IntoNDArray<Rank0, T> for T {
    fn into_ndarray(self) -> NDArray<Rank0, T> {
        NDArray::new(vec![self])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_into_ndarray() {
        let array = 42.into_ndarray();
        assert_eq!(array.get(&[]), Ok(42));
    }
}