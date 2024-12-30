use crate::dtype::DType;
use crate::ndarray::NDArray;
use crate::shape::{Rank0, Shape};

trait IntoNDArray<S: Shape, T: DType> {
    fn into_array(self) -> NDArray<S, T>;
}

impl<T: DType> IntoNDArray<Rank0, T> for T {
    fn into_array(self) -> NDArray<Rank0, T> {
        NDArray::new(vec![self])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_into_array() {
        let array = 42.into_array();
        assert_eq!(array.get(&[]), Ok(42));
    }
}