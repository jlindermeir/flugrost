pub struct Zero;
pub struct Succ<N>(core::marker::PhantomData<N>);

pub trait Nat {
    const VALUE: usize;
}

impl Nat for Zero {
    const VALUE: usize = 0;
}

impl<N: Nat> Nat for Succ<N> {
    const VALUE: usize = 1 + N::VALUE;
}

pub trait GreaterThanZero: Nat {}
impl GreaterThanZero for Succ<One> {}
impl<N: GreaterThanZero> GreaterThanZero for Succ<N> {}


pub type One = Succ<Zero>;
pub type Two = Succ<One>;
pub type Three = Succ<Two>;
pub type Four = Succ<Three>;
pub type Five = Succ<Four>;


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nat() {
        assert_eq!(Zero::VALUE, 0);
        assert_eq!(One::VALUE, 1);
        assert_eq!(Two::VALUE, 2);
        assert_eq!(Three::VALUE, 3);
        assert_eq!(Four::VALUE, 4);
    }
}