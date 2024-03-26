use crate::computational_graph::node::{Constant, Node, NodeOutput};
use crate::ndarray::ndarray::IntoNDArray;

mod ndarray;
mod computational_graph;

fn main() {
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
