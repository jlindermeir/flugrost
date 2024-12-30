pub trait DType: Copy + PartialEq + std::fmt::Debug {
    fn zero() -> Self;
    fn one() -> Self;
}

impl DType for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
}

impl DType for i32 {
    fn zero() -> Self {
        0
    }
    fn one() -> Self {
        1
    }
}