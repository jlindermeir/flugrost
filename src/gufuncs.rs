use crate::dtype::DType;
use crate::ndarray::NDArray;
use crate::number::{Nat, Two};
use crate::shape::{DimCons, Rank1, Shape};

fn norm<Head, Tail>(array: &NDArray<DimCons<Head, Tail>, f32>) -> NDArray<Tail, f32>
where
    Head: Nat,
    Tail: Shape,
{
    let mut data = Vec::with_capacity(Tail::N_ELEMENTS);

    for i in 0..Tail::N_ELEMENTS {
        let indices = Tail::compute_indices(i);
        let mut sum: f32 = 0.0;

        for j in 0..Head::VALUE {
            let mut index = indices.clone();
            index.push(j);
            sum += array.get(&index).unwrap().powi(2);
        }

        data.push(sum.sqrt());
    }

    NDArray::new(data)
}

#[cfg(test)]
mod tests {
    use crate::shape::Rank2;
    use super::*;

    #[test]
    fn test_norm_1d() {
        let data = vec![3.0, 4.0];
        let array = NDArray::<Rank1<Two>, f32>::new(data);
        let result = norm(&array);

        assert_eq!(result.get(&[]), Ok(5.0));
    }

    #[test]
    fn test_norm_2d() {
        let data = vec![5.0, 6.0, 12.0, 8.0];
        let array = NDArray::<Rank2<Two, Two>, f32>::new(data);
        let result = norm(&array);

        assert_eq!(result.get(&[0]), Ok(13.0));
        assert_eq!(result.get(&[1]), Ok(10.0));
    }
}