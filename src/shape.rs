use crate::number::Nat;

// Define a type-level list of dimensions
// The convention here is that it's right to left, so the first dimension is the last type parameter
pub struct DimNil;

pub struct DimCons<Head, Tail>(core::marker::PhantomData<(Head, Tail)>);

// For convenience, define some type aliases
pub type Rank0 = DimNil;
pub type Rank1<D0: Nat> = DimCons<D0, Rank0>;
pub type Rank2<D0: Nat, D1: Nat> = DimCons<D1, Rank1<D0>>;


pub trait Shape {
    const RANK: usize;
    const N_ELEMENTS: usize;
    fn compute_offset(indices: &[usize]) -> Result<usize, &str>;
    fn shape() -> Vec<usize>;
}

impl Shape for DimNil {
    const RANK: usize = 0;
    const N_ELEMENTS: usize = 1;
    fn compute_offset(indices: &[usize]) -> Result<usize, &str> {
        if indices.is_empty() {
            Ok(0)
        } else {
            Err("Cannot index into a 0-D shape")
        }
    }
    fn shape() -> Vec<usize> {
        vec![]
    }
}

impl<D: Nat, Tail: Shape> Shape for DimCons<D, Tail> {
    const RANK: usize = 1 + Tail::RANK;
    const N_ELEMENTS: usize = D::VALUE * Tail::N_ELEMENTS;
    fn compute_offset(indices: &[usize]) -> Result<usize, &str> {
        if indices.len() != Self::RANK {
            return Err("Incorrect number of indices");
        }

        if let Some((head, tail)) = indices.split_last() {
            if *head >= D::VALUE {
                return Err("Index out of bounds");
            }

            let tail_offset = Tail::compute_offset(tail)?;
            Ok(head * Tail::N_ELEMENTS + tail_offset)
        } else {
            Err("Incorrect number of indices")
        }
    }
    fn shape() -> Vec<usize> {
        let mut shape = Tail::shape();
        shape.push(D::VALUE);
        shape
    }
}

#[cfg(test)]
mod tests {
    use crate::number::{Five, Four, Three, Two};
    use super::*;

    #[test]
    fn test_rank() {
        // The rank of a 0-D shape (DimNil) should be 0
        assert_eq!(<Rank0 as Shape>::RANK, 0);

        // The rank of (D0) should be 1
        assert_eq!(<Rank1<Four> as Shape>::RANK, 1);

        // The rank of (D1, D0) should be 2
        assert_eq!(<Rank2<Two, Three> as Shape>::RANK, 2);
    }

    #[test]
    fn test_size() {
        // The size of DimNil is 1 by convention
        assert_eq!(<Rank0 as Shape>::N_ELEMENTS, 1);

        // The size of (D0) is D0
        assert_eq!(<Rank1<Four> as Shape>::N_ELEMENTS, 4);

        // The size of (D1, D0) is D1 * D0
        assert_eq!(<Rank2<Two, Three> as Shape>::N_ELEMENTS, 6);

        // Another quick example
        assert_eq!(<Rank2<Four, Five> as Shape>::N_ELEMENTS, 20);
    }

    #[test]
    fn test_compute_offset() {
        // Test a 0-D shape
        assert_eq!(<Rank0 as Shape>::compute_offset(&[]), Ok(0));

        // Test a 1-D shape
        assert_eq!(<Rank1<Four> as Shape>::compute_offset(&[2]), Ok(2));
        assert_eq!(<Rank1<Four> as Shape>::compute_offset(&[4]), Err("Index out of bounds"));
        assert_eq!(<Rank1<Four> as Shape>::compute_offset(&[2, 3]), Err("Incorrect number of indices"));

        // Test a 2-D shape
        assert_eq!(<Rank2<Two, Three> as Shape>::compute_offset(&[0, 0]), Ok(0));
        assert_eq!(<Rank2<Two, Three> as Shape>::compute_offset(&[0, 2]), Ok(4));
        assert_eq!(<Rank2<Two, Three> as Shape>::compute_offset(&[1, 2]), Ok(5));
        assert_eq!(<Rank2<Two, Three> as Shape>::compute_offset(&[2, 4]), Err("Index out of bounds"));
        assert_eq!(<Rank2<Two, Three> as Shape>::compute_offset(&[1]), Err("Incorrect number of indices"));
    }

    #[test]
    fn test_shape() {
        // Test a 0-D shape
        assert_eq!(<Rank0 as Shape>::shape(), vec![]);

        // Test a 1-D shape
        assert_eq!(<Rank1<Four> as Shape>::shape(), vec![4]);

        // Test a 2-D shape
        assert_eq!(<Rank2<Two, Three> as Shape>::shape(), vec![2, 3]);
    }
}
