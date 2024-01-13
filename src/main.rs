use crate::ndarray::ndarray::{IntoNDArray, mat_mul};

mod ndarray;

fn main() {
    let arr1 = [
        [0, 1, 2],
        [3, 4, 5]
    ].into_array();
    let arr2 = [
        [1, 0],
        [0, 1],
        [0, 0]
    ].into_array();
    let prod = mat_mul(&arr1, &arr2);
    print!("{}", prod[[0, 1]]);
}
