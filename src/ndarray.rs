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
    use crate::number::{Three, Two};
    use super::*;
    use crate::shape::{Rank2};

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
}