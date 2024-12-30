use crate::dtype::DType;
use crate::shape::{Shape};

pub struct NDArray<S: Shape, T> {
    data: Vec<T>,
    shape_marker: std::marker::PhantomData<S>,
}

impl<S: Shape, T: DType> NDArray<S, T> {
    pub fn new(data: Vec<T>) -> Self {
        if data.len() != S::SIZE {
            panic!(
                "Expected data of size {}, got size {}",
                S::SIZE,
                data.len()
            );
        }

        Self { data, shape_marker: std::marker::PhantomData }
    }

    pub fn get<'a>(&'a self, indices: &'a [usize]) -> Result<T, &str> {
        let offset = S::compute_offset(indices)?;
        Ok(self.data[offset])
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::{Rank2};

    #[test]
    fn test_ndarray() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let array = NDArray::<Rank2<2, 3>, i32>::new(data);

        assert_eq!(array.get(&[0, 0]), Ok(1));
        assert_eq!(array.get(&[1, 0]), Ok(2));
        assert_eq!(array.get(&[0, 1]), Ok(3));
        assert_eq!(array.get(&[1, 1]), Ok(4));
        assert_eq!(array.get(&[0, 2]), Ok(5));
        assert_eq!(array.get(&[1, 2]), Ok(6));
    }
}