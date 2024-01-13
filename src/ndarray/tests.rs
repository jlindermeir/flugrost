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

#[test]
fn test_addition() {
    let arr1 = [1, 2, 3].into_array();
    let arr2 = [4, 5, 6].into_array();
    let sum = &arr1 + &arr2;
    assert_eq!(sum[[0]], 5);
    assert_eq!(sum[[1]], 7);
    assert_eq!(sum[[2]], 9);
}

#[test]
fn test_subtraction() {
    let arr1 = [1, 2, 3].into_array();
    let arr2 = [4, 5, 6].into_array();
    let diff = &arr1 - &arr2;
    assert_eq!(diff[[0]], -3);
    assert_eq!(diff[[1]], -3);
    assert_eq!(diff[[2]], -3);
}

#[test]
fn test_multiplication() {
    let arr1 = [1, 2, 3].into_array();
    let arr2 = [4, 5, 6].into_array();
    let prod = &arr1 * &arr2;
    assert_eq!(prod[[0]], 4);
    assert_eq!(prod[[1]], 10);
    assert_eq!(prod[[2]], 18);
}
