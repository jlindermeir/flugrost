use crate::ndarray::{NDArray};

mod ndarray;

fn main() {
    let arr = NDArray {
        _data: Vec::from([0, 1, 2, 3, 4, 5]),
        shape: (2, 3)
    };
    println!("{}", arr[[1, 0]])
}
