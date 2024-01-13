use crate::computational_graph::node::{BinaryOp, Constant, Node};

#[test]
fn test_simple_addition() {
    let a = Constant {
        value: 1
    };
    let b = Constant {
        value: 2
    };
    let sum = BinaryOp {
        op: |a, b| a + b,
        lhs: &a,
        rhs: &b
    };
    assert_eq!(sum.output(), 3)
}