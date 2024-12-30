use crate::shape::Shape;

pub struct NDArray<S: Shape, T> {
    data: Vec<T>,
    pub shape: S,
}

impl<S: Shape, T> NDArray<S, T> {
    pub fn new(data: Vec<T>, shape: S) -> Self {
        if data.len() != S::SIZE {
            panic!(
                "Expected data of size {}, got size {}",
                S::SIZE,
                data.len()
            );
        }

        Self { data, shape }
    }
}