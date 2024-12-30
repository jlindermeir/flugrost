// Define a type-level list of dimensions
// The convention here is that it's right to left, so the first dimension is the last type parameter
pub struct DimNil;

pub struct DimCons<const D: usize, Tail>(core::marker::PhantomData<Tail>);

// For convenience, define some type aliases
pub type Rank0 = DimNil;
pub type Rank1<const D0: usize> = DimCons<D0, Rank0>;
pub type Rank2<const D0: usize, const D1: usize> = DimCons<D1, Rank1<D0>>;


pub trait Shape {
    const RANK: usize;
    const SIZE: usize;
}

impl Shape for DimNil {
    const RANK: usize = 0;
    const SIZE: usize = 1;
}

impl<const D: usize, Tail: Shape> Shape for DimCons<D, Tail> {
    const RANK: usize = 1 + Tail::RANK;
    const SIZE: usize = D * Tail::SIZE;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank() {
        // The rank of a 0-D shape (DimNil) should be 0
        assert_eq!(<Rank0 as Shape>::RANK, 0);

        // The rank of (D0) should be 1
        assert_eq!(<Rank1<4> as Shape>::RANK, 1);

        // The rank of (D1, D0) should be 2
        assert_eq!(<Rank2<2, 3> as Shape>::RANK, 2);
    }

    #[test]
    fn test_size() {
        // The size of DimNil is 1 by convention
        assert_eq!(<Rank0 as Shape>::SIZE, 1);

        // The size of (D0) is D0
        assert_eq!(<Rank1<4> as Shape>::SIZE, 4);

        // The size of (D1, D0) is D1 * D0
        assert_eq!(<Rank2<2, 3> as Shape>::SIZE, 6);

        // Another quick example
        assert_eq!(<Rank2<4, 5> as Shape>::SIZE, 20);
    }
}
