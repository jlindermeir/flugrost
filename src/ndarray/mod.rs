struct Const<const N: usize>;

trait Dim {
    fn size(&self) -> usize;
}

impl<const N: usize> Dim for Const<N> {
    fn size(&self) -> usize {
        N
    }
}

trait Shape {
    const N_DIMS: usize;
    fn shape(&self) -> [usize; Self::N_DIMS];
}
