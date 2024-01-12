use crate::ndarray::ndarray::IntoNDArray;

mod ndarray;

fn main() {
    let arr1 = [
        [0, 1, 2],
        [3, 4, 5]
    ].into_array();
    print!("{}", arr1[[0, 1]]);
}
