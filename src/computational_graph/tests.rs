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
    let sum = a + b;
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
    let sum2 = sum1 + c;
    assert_eq!(sum2.output()[[0]], 9);
    assert_eq!(sum2.output()[[1]], 12);
    assert_eq!(sum2.output()[[2]], 15);
}

#[test]
fn test_negation() {
    let a = Node(Constant {
        array: [1, 2, 3].into_array()
    });
    let neg = -a;
    assert_eq!(neg.output()[[0]], -1);
    assert_eq!(neg.output()[[1]], -2);
    assert_eq!(neg.output()[[2]], -3);
}

#[test]
fn test_subtraction() {
    let a = Node(Constant {
        array: [1, 2, 3].into_array()
    });
    let b = Node(Constant {
        array: [3, 4, 5].into_array()
    });
    let diff = a - b;
    assert_eq!(diff.output()[[0]], -2);
    assert_eq!(diff.output()[[1]], -2);
    assert_eq!(diff.output()[[2]], -2);
}

#[test]
fn test_multiplication() {
    let a = Node(Constant {
        array: [1, 2, 3].into_array()
    });
    let b = Node(Constant {
        array: [3, 4, 5].into_array()
    });
    let prod = a * b;
    assert_eq!(prod.output()[[0]], 3);
    assert_eq!(prod.output()[[1]], 8);
    assert_eq!(prod.output()[[2]], 15);
}

#[test]
fn test_scalar_multiplication() {
    let a = Node(Constant {
        array: [1, 2, 3].into_array()
    });
    let prod = a * 2;
    assert_eq!(prod.output()[[0]], 2);
    assert_eq!(prod.output()[[1]], 4);
    assert_eq!(prod.output()[[2]], 6);
}
