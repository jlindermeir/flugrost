pub trait ShapeSize {
    const SIZE: usize;
}

pub struct Rank0;
impl ShapeSize for Rank0 {
    const SIZE: usize = 0;
}

pub struct Rank1<const D0: usize>;
impl<const D0: usize> ShapeSize for Rank1<D0> {
    const SIZE: usize = D0;
}

pub struct Rank2<const D0: usize, const D1: usize>;
impl<const D0: usize, const D1: usize> ShapeSize for Rank2<D0, D1> {
    const SIZE: usize = D0 * D1;
}

pub struct Rank3<const D0: usize, const D1: usize, const D2: usize>;
impl<const D0: usize, const D1: usize, const D2: usize> ShapeSize for Rank3<D0, D1, D2> {
    const SIZE: usize = D0 * D1 * D2;
}
