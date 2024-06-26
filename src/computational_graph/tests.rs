use crate::computational_graph::grad::Grad;
use crate::computational_graph::node::{Constant, Node, NodeOutput};
use crate::ndarray::ndarray::IntoNDArray;

#[test]
fn test_simple_addition() {
    let a = Node(Constant::new([1, 2].into_array()));
    let b = Node(Constant::new([3, 4].into_array()));
    let sum = &a + &b;
    assert_eq!(sum.output()[[0]], 4);
    assert_eq!(sum.output()[[1]], 6);
}

#[test]
fn test_more_complicated_addition() {
    let a = Node(Constant::new([1, 2, 3].into_array()));
    let b = Node(Constant::new([4, 5, 6].into_array()));
    let c = Node(Constant::new([4, 5, 6].into_array()));
    let sum1 = &a + &b;
    let sum2 = &sum1 + &c;
    assert_eq!(sum2.output()[[0]], 9);
    assert_eq!(sum2.output()[[1]], 12);
    assert_eq!(sum2.output()[[2]], 15);
}

#[test]
fn test_negation() {
    let a = Node(Constant::new([1, 2, 3].into_array()));
    let neg = -&a;
    assert_eq!(neg.output()[[0]], -1);
    assert_eq!(neg.output()[[1]], -2);
    assert_eq!(neg.output()[[2]], -3);
}

#[test]
fn test_subtraction() {
    let a = Node(Constant::new([1, 2, 3].into_array()));
    let b = Node(Constant::new([4, 5, 6].into_array()));
    let diff = &a - &b;
    assert_eq!(diff.output()[[0]], -3);
    assert_eq!(diff.output()[[1]], -3);
    assert_eq!(diff.output()[[2]], -3);
}

#[test]
fn test_multiplication() {
    let a = Node(Constant::new([1, 2, 3].into_array()));
    let b = Node(Constant::new([4, 5, 6].into_array()));
    let prod = &a * &b;
    assert_eq!(prod.output()[[0]], 4);
    assert_eq!(prod.output()[[1]], 10);
    assert_eq!(prod.output()[[2]], 18);
}

#[test]
fn test_division() {
    let a = Node(Constant::new([1, 2, 3].into_array()));
    let b = Node(Constant::new([4, 5, 6].into_array()));
    let div = &a / &b;
    assert_eq!(div.output()[[0]], 0);
    assert_eq!(div.output()[[1]], 0);
    assert_eq!(div.output()[[2]], 0);
}

#[test]
fn test_node_reuse() {
    let a = Node(Constant::new([1, 2, 3].into_array()));
    let b = Node(Constant::new([4, 5, 6].into_array()));
    let c = Node(Constant::new([4, 5, 6].into_array()));
    let sum1 = &a + &b;
    let sum2 = &a + &c;
    assert_eq!(sum1.output()[[0]], 5);
    assert_eq!(sum1.output()[[1]], 7);
    assert_eq!(sum1.output()[[2]], 9);
    assert_eq!(sum2.output()[[0]], 5);
    assert_eq!(sum2.output()[[1]], 7);
    assert_eq!(sum2.output()[[2]], 9);
}

#[test]
fn test_rank0_constant_grad() {
    let a = Node(Constant::new(1.0.into_array()));
    let b = Node(Constant::new(2.0.into_array()));
    let da_da = a.grad(&a);
    let da_db = a.grad(&b);

    assert_eq!(da_da.output()[[]], 1.0);
    assert_eq!(da_db.output()[[]], 0.0);
}
