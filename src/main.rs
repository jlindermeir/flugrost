use crate::ndarray::ndarray::{IntoNDArray};
use crate::ndarray::ops::mat_mul;

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
    print!("{}", prod);
    let arr3 = [0, 1, 3].into_array();
    println!("{}", arr3)
}
