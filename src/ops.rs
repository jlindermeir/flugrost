use std::ops::{Add, Div, Mul, Sub};
use crate::broadcast::Broadcast;
use crate::dtype::DType;
use crate::ndarray::NDArray;
use crate::number::Nat;
use crate::shape::{DimCons, Shape};

macro_rules! impl_broadcast_op {
    ($Op:ident, $op_fn:ident, $op_symbol:tt) => {
        impl<S1, S2, T> $Op<&NDArray<S2, T>> for &NDArray<S1, T>
        where
            S1: Shape + Broadcast<S2>,
            S2: Shape + Broadcast<S1>,
            T: DType,
        {
            type Output = NDArray<S1::Output, T>;

            fn $op_fn(self, rhs: &NDArray<S2, T>) -> Self::Output {
                let lhs = self.broadcast::<S2>();
                let rhs = rhs.broadcast::<S1>();

                let data = lhs
                    .data
                    .into_iter()
                    .zip(rhs.data.into_iter())
                    .map(|(a, b)| a $op_symbol b)
                    .collect();

                NDArray::new(data)
            }
        }
    };
}

// Now invoke the macro for each operation you want.
impl_broadcast_op!(Add, add, +);
impl_broadcast_op!(Sub, sub, -);
impl_broadcast_op!(Mul, mul, *);
impl_broadcast_op!(Div, div, /);

#[cfg(test)]
mod tests {
    use crate::number::{One, Three, Two};
    use crate::shape::{Rank0, Rank1, Rank2};
    use crate::ndarray::NDArray;

    #[test]
    fn test_add() {
        let data1 = vec![1, 2, 3, 4, 5, 6];
        let data2 = vec![1, 2, 3, 4, 5, 6];
        let array1 = NDArray::<Rank2<Two, Three>, i32>::new(data1);
        let array2 = NDArray::<Rank2<Two, Three>, i32>::new(data2);

        let result = &array1 + &array2;

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

        let result = &array1 + &array2;

        assert_eq!(result.shape(), vec![2, 3]);
        assert_eq!(result.get(&[0, 0]), Ok(7));
        assert_eq!(result.get(&[1, 0]), Ok(8));
        assert_eq!(result.get(&[0, 1]), Ok(8));
        assert_eq!(result.get(&[1, 1]), Ok(9));
        assert_eq!(result.get(&[0, 2]), Ok(9));
        assert_eq!(result.get(&[1, 2]), Ok(10));
    }

    #[test]
    fn test_other_ops() {
        let data1 = vec![1, 2, 3];
        let data2 = vec![2];

        let array1 = NDArray::<Rank1<Three>, i32>::new(data1);
        let array2 = NDArray::<Rank0, i32>::new(data2);

        let add_result = &array1 + &array2;
        let sub_result = &array1 - &array2;
        let mul_result = &array1 * &array2;
        let div_result = &array1 / &array2;

        assert_eq!(add_result.get(&[0]), Ok(3));
        assert_eq!(add_result.get(&[1]), Ok(4));
        assert_eq!(add_result.get(&[2]), Ok(5));

        assert_eq!(sub_result.get(&[0]), Ok(-1));
        assert_eq!(sub_result.get(&[1]), Ok(0));
        assert_eq!(sub_result.get(&[2]), Ok(1));

        assert_eq!(mul_result.get(&[0]), Ok(2));
        assert_eq!(mul_result.get(&[1]), Ok(4));
        assert_eq!(mul_result.get(&[2]), Ok(6));

        assert_eq!(div_result.get(&[0]), Ok(0));
        assert_eq!(div_result.get(&[1]), Ok(1));
        assert_eq!(div_result.get(&[2]), Ok(1));
    }



}