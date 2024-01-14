use crate::computational_graph::node::{Constant, Node, NodeOutput};
use crate::ndarray::ndarray::IntoNDArray;

#[test]
fn test_simple_addition() {
    let a = Node(Constant {
        array: [1, 2, 3].into_array()
    });
    let b = Node(Constant {
        array: [3, 4, 5].into_array()
    });
    let mut sum = a + b;
    assert_eq!(sum.output()[[0]], 4);
    assert_eq!(sum.output()[[1]], 6);
}

#[test]
fn test_more_complicated_addition() {
    let a = Node(Constant {
        array: [1, 2, 3].into_array()
    });
    let b = Node(Constant {
        array: [3, 4, 5].into_array()
    });
    let c = Node(Constant {
        array: [5, 6, 7].into_array()
    });
    let sum1 = a + b;
    let mut sum2 = sum1 + c;
    assert_eq!(sum2.output()[[0]], 9);
    assert_eq!(sum2.output()[[1]], 12);
    assert_eq!(sum2.output()[[2]], 15);
}
