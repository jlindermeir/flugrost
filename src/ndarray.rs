use crate::broadcast::Broadcast;
use crate::dtype::DType;
use crate::shape::{Shape};

pub struct NDArray<S: Shape, T> {
    pub(crate) data: Vec<T>,
    shape_marker: std::marker::PhantomData<S>,
}

impl<S: Shape, T: DType> NDArray<S, T> {
    pub fn new(data: Vec<T>) -> Self {
        if data.len() != S::N_ELEMENTS {
            panic!(
                "Expected data of size {}, got size {}",
                S::N_ELEMENTS,
                data.len()
            );
        }

        Self { data, shape_marker: std::marker::PhantomData }
    }

    pub fn shape(&self) -> Vec<usize> {
        S::shape()
    }

    pub fn get<'a>(&'a self, indices: &'a [usize]) -> Result<T, &str> {
        let offset = S::compute_offset(indices)?;
        Ok(self.data[offset])
    }

    pub fn broadcast<SN: Broadcast<S>>(self) -> NDArray<SN::Output, T> {
        let old_shape = S::shape();

        let mut new_data = Vec::with_capacity(SN::Output::N_ELEMENTS);

        for i in 0..SN::Output::N_ELEMENTS {
            let mut new_array_indices = SN::Output::compute_indices(i);
            let mut indices = Vec::new();

            // If the new shape is larger than the old shape, we need to discard the leading dimensions
            if SN::Output::RANK > S::RANK {
                new_array_indices = new_array_indices.split_off(SN::Output::RANK - S::RANK);
            }

            for (j, index) in new_array_indices.iter().enumerate() {
                // If the index is for a broadcast dimension, we set it to 0
                if old_shape[j] == 1 {
                    indices.push(0);
                } else {
                    indices.push(*index);
                }
            }

            if let Ok(value) = self.get(&indices) {
                new_data.push(value);
            } else {
                panic!("Error while broadcasting");
            }
        }

        NDArray::new(new_data)
    }

    pub fn full(value: T) -> Self {
        let data = vec![value; S::N_ELEMENTS];
        Self::new(data)
    }

    pub fn ones() -> Self {
        Self::full(T::one())
    }

    pub fn zeros() -> Self {
        Self::full(T::zero())
    }
}


#[cfg(test)]
mod tests {
    use crate::number::{One, Three, Two};
    use super::*;
    use crate::shape::{Rank1, Rank2};

    #[test]
    fn test_ndarray() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let array = NDArray::<Rank2<Two, Three>, i32>::new(data);

        assert_eq!(array.get(&[0, 0]), Ok(1));
        assert_eq!(array.get(&[1, 0]), Ok(2));
        assert_eq!(array.get(&[0, 1]), Ok(3));
        assert_eq!(array.get(&[1, 1]), Ok(4));
        assert_eq!(array.get(&[0, 2]), Ok(5));
        assert_eq!(array.get(&[1, 2]), Ok(6));
    }

    #[test]
    fn test_broadcast_leading_dim() {
        let data = vec![1, 2, 3];
        let array = NDArray::<Rank1<Three>, i32>::new(data);
        let broadcasted_array = array.broadcast::<Rank2<Two, Three>>();

        print!("{:?}", broadcasted_array.shape());

        assert_eq!(broadcasted_array.get(&[0, 0]), Ok(1));
        assert_eq!(broadcasted_array.get(&[1, 0]), Ok(1));
        assert_eq!(broadcasted_array.get(&[0, 1]), Ok(2));
        assert_eq!(broadcasted_array.get(&[1, 1]), Ok(2));
        assert_eq!(broadcasted_array.get(&[0, 2]), Ok(3));
        assert_eq!(broadcasted_array.get(&[1, 2]), Ok(3));
    }

    #[test]
    fn test_broadcast_inner_dim() {
        let data = vec![1, 2, 3];
        let array = NDArray::<Rank2<Three, One>, i32>::new(data);
        let broadcasted_array = array.broadcast::<Rank2<Three, Two>>();

        assert_eq!(broadcasted_array.get(&[0, 0]), Ok(1));
        assert_eq!(broadcasted_array.get(&[0, 1]), Ok(1));
        assert_eq!(broadcasted_array.get(&[1, 0]), Ok(2));
        assert_eq!(broadcasted_array.get(&[1, 1]), Ok(2));
        assert_eq!(broadcasted_array.get(&[2, 0]), Ok(3));
        assert_eq!(broadcasted_array.get(&[2, 1]), Ok(3));
    }
}