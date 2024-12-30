// Define a type-level list of dimensions
// The convention here is that it's right to left, so the first dimension is the last type parameter
pub struct DimNil;

pub struct DimCons<const D: usize, Tail>(core::marker::PhantomData<Tail>);

// For convenience, define some type aliases
pub type Rank0 = DimNil;
pub type Rank1<const D0: usize> = DimCons<D0, Rank0>;
pub type Rank2<const D0: usize, const D1: usize> = DimCons<D1, Rank1<D0>>;


// Implement a trait to get the number of dimensions
pub trait Rank {
    const RANK: usize;
}

impl Rank for DimNil {
    const RANK: usize = 0;
}

impl<const D: usize, Tail: Rank> Rank for DimCons<D, Tail> {
    const RANK: usize = 1 + Tail::RANK;
}

// Implement a trait to get the size of all dimensions
pub trait Size {
    const SIZE: usize;
}

impl Size for DimNil {
    const SIZE: usize = 1;
}

impl<const D: usize, Tail: Size> Size for DimCons<D, Tail> {
    const SIZE: usize = D * Tail::SIZE;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank() {
        // The rank of a 0-D shape (DimNil) should be 0
        assert_eq!(<Rank0 as Rank>::RANK, 0);

        // The rank of (D0) should be 1
        assert_eq!(<Rank1<4> as Rank>::RANK, 1);

        // The rank of (D1, D0) should be 2
        assert_eq!(<Rank2<2, 3> as Rank>::RANK, 2);
    }

    #[test]
    fn test_size() {
        // The size of DimNil is 1 by convention
        assert_eq!(<Rank0 as Size>::SIZE, 1);

        // The size of (D0) is D0
        assert_eq!(<Rank1<4> as Size>::SIZE, 4);

        // The size of (D1, D0) is D1 * D0
        assert_eq!(<Rank2<2, 3> as Size>::SIZE, 6);

        // Another quick example
        assert_eq!(<Rank2<4, 5> as Size>::SIZE, 20);
    }
}
