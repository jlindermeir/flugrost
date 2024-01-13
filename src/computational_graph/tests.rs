use crate::computational_graph::node::{Constant, Node, Sum};

#[test]
fn test_simple_addition() {
    let a = Constant {
        value: 1
    };
    let b = Constant {
        value: 2
    };
    let sum = Sum {
        lhs: &a,
        rhs: &b
    };
    assert_eq!(sum.output(), 3)
}