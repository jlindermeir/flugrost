use crate::ndarray::{IntoNDArray, unary_op};

mod ndarray;

fn main() {
    let arr1 = [1., 2., 3.].into_tensor();
    let arr2= [3, 4, 4, 5].into_tensor();
    let neg = -&arr1;
    print!("{}", neg[[0]]);
}
