use std::ops::Add;

pub trait Node {
    type Output;
    fn output(&self) -> Self::Output;
}

pub struct Constant<T> {
    pub value: T
}

impl<T: Copy> Node for Constant<T> {
    type Output = T;
    fn output(&self) -> Self::Output {
        self.value
    }
}

pub struct BinaryOp<'a, T, N>
where N: Node<Output = T>
{
    pub op: fn(&T, &T) -> T,
    pub rhs: &'a N,
    pub lhs: &'a N
}

impl<'a, T: Add<Output = T>, N: Node<Output = T>> Node for BinaryOp<'a, T, N> {
    type Output = T;

    fn output(&self) -> Self::Output {
        (self.op)(&self.lhs.output(), &self.rhs.output())
    }
}
