use crate::computational_graph::node::{Constant, Node};
use crate::computational_graph::ops::add;
use crate::ndarray::ndarray::IntoNDArray;

#[test]
fn test_simple_addition() {
    let a = Constant {
        array: [1, 2].into_array()
    };
    let b = Constant {
        array: [3, 4].into_array()
    };
    let mut sum = add(a, b);
    assert_eq!(sum.output()[[0]], 4);
    assert_eq!(sum.output()[[1]], 6);
}
