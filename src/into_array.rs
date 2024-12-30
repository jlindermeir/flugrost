use crate::dtype::DType;
use crate::ndarray::NDArray;
use crate::shape::{Rank0, Rank1, Rank2, Shape};

trait IntoNDArray<S: Shape, T: DType> {
    fn into_array(self) -> NDArray<S, T>;
}

impl<T: DType> IntoNDArray<Rank0, T> for T {
    fn into_array(self) -> NDArray<Rank0, T> {
        NDArray::new(vec![self])
    }
}

impl<const D0: usize, T: DType> IntoNDArray<Rank1<D0>, T> for [T; D0] {
    fn into_array(self) -> NDArray<Rank1<D0>, T> {
        NDArray::new(self.to_vec())
    }
}

impl<const D0: usize, const D1: usize, T: DType> IntoNDArray<Rank2<D0, D1>, T> for [[T; D1]; D0] {
    fn into_array(self) -> NDArray<Rank2<D0, D1>, T> {
        let mut data = Vec::with_capacity(D0 * D1);

        for j in 0..D1 {
            for i in 0..D0 {
                data.push(self[i][j]);
            }
        }

        NDArray::new(data)
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

    #[test]
    fn test_into_array_1d() {
        let array = [1, 2, 3].into_array();
        assert_eq!(array.get(&[0]), Ok(1));
        assert_eq!(array.get(&[1]), Ok(2));
        assert_eq!(array.get(&[2]), Ok(3));
    }

    #[test]
    fn test_into_array_2d() {
        let array = [[1, 2, 3], [4, 5, 6]].into_array();
        assert_eq!(array.get(&[0, 0]), Ok(1));
        assert_eq!(array.get(&[0, 1]), Ok(2));
        assert_eq!(array.get(&[0, 2]), Ok(3));
        assert_eq!(array.get(&[1, 0]), Ok(4));
        assert_eq!(array.get(&[1, 1]), Ok(5));
        assert_eq!(array.get(&[1, 2]), Ok(6));
    }
}