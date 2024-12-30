use crate::shape::Shape;

pub struct NDArray<S: Shape, T> {
    data: Vec<T>,
    pub shape: S,
}

impl<S: Shape, T> NDArray<S, T> {
    pub fn new(data: Vec<T>, shape: S) -> Self {
        let expected_size = shape.size();
        if data.len() != expected_size {
            panic!(
                "Expected data of size {}, got size {}",
                expected_size,
                data.len()
            );
        }

        Self { data, shape }
    }
}