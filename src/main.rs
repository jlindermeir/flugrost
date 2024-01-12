use crate::ndarray::{add, NDArray};

mod ndarray;

fn main() {
    let arr1 = NDArray {
        data: Vec::from([0, 1, 2, 3, 4, 5]),
        shape: (2, 3)
    };
    let arr2 = NDArray {
        data: Vec::from([0, 1, 2, 3, 4, 5]),
        shape: (2, 3)
    };
    let sum = add(&arr1, &arr2);
    println!("{}", sum[[1, 2]])
}
