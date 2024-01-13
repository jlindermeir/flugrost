use crate::ndarray::ndarray::IntoNDArray;

#[test]
fn test_rank0() {
    let arr = [1].into_array();
    assert_eq!(arr[[0]], 1);
}

#[test]
fn test_rank1() {
    let arr = [1, 2, 3].into_array();
    assert_eq!(arr[[0]], 1);
    assert_eq!(arr[[1]], 2);
    assert_eq!(arr[[2]], 3);
}

#[test]
fn test_rank2() {
    let arr = [[1, 2, 3], [4, 5, 6]].into_array();
    assert_eq!(arr[[0, 0]], 1);
    assert_eq!(arr[[0, 1]], 2);
    assert_eq!(arr[[0, 2]], 3);
    assert_eq!(arr[[1, 0]], 4);
    assert_eq!(arr[[1, 1]], 5);
    assert_eq!(arr[[1, 2]], 6);
}
