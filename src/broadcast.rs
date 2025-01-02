use crate::number::{GreaterThanZero, Nat, One};
use crate::shape::{DimCons, DimNil};

pub trait BroadcastOneDim<Rhs> {
    type Output;
}

impl<D: GreaterThanZero> BroadcastOneDim<D> for One {
    type Output = D;
}

impl<D: GreaterThanZero> BroadcastOneDim<One> for D {
    type Output = D;

}

impl<D: Nat> BroadcastOneDim<D> for D {
    type Output = D;

}

pub trait Broadcast<Rhs> {
    type Output;
}

impl<Head1, Tail1, Head2, Tail2> Broadcast<DimCons<Head2, Tail2>> for DimCons<Head1, Tail1>
where
    Head1: BroadcastOneDim<Head2>,
    Tail1: Broadcast<Tail2>,
{
    type Output = DimCons<<Head1 as BroadcastOneDim<Head2>>::Output, <Tail1 as Broadcast<Tail2>>::Output>;
}

impl Broadcast<DimNil> for DimNil {
    type Output = DimNil;
}

impl<Head, Tail> Broadcast<DimCons<Head, Tail>> for DimNil
where
    Head: BroadcastOneDim<One>,
    Tail: Broadcast<DimNil>,
{
    type Output = DimCons<Head::Output, Tail::Output>;
}

impl<Head, Tail> Broadcast<DimNil> for DimCons<Head, Tail>
where
    Head: BroadcastOneDim<One>,
    Tail: Broadcast<DimNil>,
{
    type Output = DimCons<Head::Output, Tail::Output>;
}

#[cfg(test)]
mod tests {
    use crate::number::{Three, Two};
    use crate::shape::{Rank1, Rank2, Rank3, Shape};
    use super::*;


    #[test]
    fn test_broadcast_one_dim() {
        type A = One;
        type B = Two;
        type C = Three;

        type AB = <A as BroadcastOneDim<B>>::Output;
        type AC = <A as BroadcastOneDim<C>>::Output;

        assert_eq!(AB::VALUE, 2);
        assert_eq!(AC::VALUE, 3);
    }

    #[test]
    fn test_broadcast() {
        // Shape (3,)
        type A = Rank1<Three>;

        // Shape (2, 1)
        type B = Rank2<Two, One>;

        // Shape (3, 1, 3)
        type C = Rank3<Three, One, Three>;

        // Shape (2, 3)
        type AB = <A as Broadcast<B>>::Output;
        // Shape (3, 1, 3)
        type AC = <A as Broadcast<C>>::Output;
        // Shape (3, 2, 3)
        type BC = <B as Broadcast<C>>::Output;

        assert_eq!(AB::RANK, 2);
        assert_eq!(AC::RANK, 3);
        assert_eq!(BC::RANK, 3);

        assert_eq!(AB::N_ELEMENTS, 6);
        assert_eq!(AC::N_ELEMENTS, 9);
        assert_eq!(BC::N_ELEMENTS, 18);

        assert_eq!(AB::shape(), vec![2, 3]);
        assert_eq!(AC::shape(), vec![3, 1, 3]);
        assert_eq!(BC::shape(), vec![3, 2, 3]);
    }
}