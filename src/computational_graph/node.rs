use std::ops::Add;
use std::process::Output;

pub trait Node {
    type OUTPUT;
    fn output(&self) -> Self::OUTPUT;
}

pub struct Constant<T> {
    pub value: T
}

impl<T: Copy> Node for Constant<T> {
    type OUTPUT = T;
    fn output(&self) -> Self::OUTPUT {
        self.value
    }
}

pub struct Sum<'a, T, N: Node<OUTPUT = T>> {
    pub lhs: &'a N,
    pub rhs: &'a N
}

impl <'a, T: Add<Output = T>, N: Node<OUTPUT = T>> Node for Sum<'a, T, N> {
    type OUTPUT = T;
    fn output(&self) -> Self::OUTPUT {
        self.lhs.output() + self.rhs.output()
    }
}
