use std::ops::Add;
use crate::broadcast::Broadcast;
use crate::dtype::DType;
use crate::ndarray::NDArray;
use crate::shape::Shape;

impl<S1: Shape, S2: Shape, T: DType> Add<NDArray<S2, T>> for NDArray<S1, T>
where
    S1: Broadcast<S2>,
    S2: Broadcast<S1>,
{
    type Output = NDArray<S1::Output, T>;

    fn add(self, rhs: NDArray<S2, T>) -> Self::Output {
        // Broadcast the arrays
        let lhs = self.broadcast::<S2>();
        let rhs = rhs.broadcast::<S1>();

        let data = lhs
            .data
            .into_iter()
            .zip(rhs.data.into_iter())
            .map(|(a, b)| a + b)
            .collect();

        NDArray::new(data)
    }
}

#[cfg(test)]
mod tests {
    use crate::number::{One, Three, Two};
    use crate::shape::{Rank1, Rank2};
    use crate::ndarray::NDArray;

    #[test]
    fn test_add() {
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let data2 = vec![1, 2, 3, 4, 5, 6];
        let array1 = NDArray::<Rank2<Two, Three>, i32>::new(data1);
        let array2 = NDArray::<Rank2<Two, Three>, i32>::new(data2);

        let result = array1 + array2;

        assert_eq!(result.get(&[0, 0]), Ok(2));
        assert_eq!(result.get(&[1, 0]), Ok(4));
        assert_eq!(result.get(&[0, 1]), Ok(6));
        assert_eq!(result.get(&[1, 1]), Ok(8));
        assert_eq!(result.get(&[0, 2]), Ok(10));
        assert_eq!(result.get(&[1, 2]), Ok(12));
    }

    #[test]
    fn test_add_broadcast() {
        let data1 = vec![1, 2, 3];
        let data2 = vec![6, 7];
        let array1 = NDArray::<Rank1<Three>, i32>::new(data1);
        let array2 = NDArray::<Rank2<Two, One>, i32>::new(data2);

        let result = array1 + array2;

        assert_eq!(result.shape(), vec![2, 3]);
        assert_eq!(result.get(&[0, 0]), Ok(7));
        assert_eq!(result.get(&[1, 0]), Ok(8));
        assert_eq!(result.get(&[0, 1]), Ok(8));
        assert_eq!(result.get(&[1, 1]), Ok(9));
        assert_eq!(result.get(&[0, 2]), Ok(9));
        assert_eq!(result.get(&[1, 2]), Ok(10));
    }
}