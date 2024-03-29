pub trait Grad<T> {
    type GradOutput;
    fn grad(&self, target: &T) -> Self::GradOutput;
}