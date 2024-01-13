use crate::ndarray::ndarray::{IntoNDArray};
use crate::ndarray::ops::mat_mul;

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

    let scalar_sum = &arr1 + 1;
    assert_eq!(scalar_sum[[0]], 2);
    assert_eq!(scalar_sum[[1]], 3);
    assert_eq!(scalar_sum[[2]], 4);
}

#[test]
fn test_subtraction() {
    let arr1 = [1, 2, 3].into_array();
    let arr2 = [4, 5, 6].into_array();
    let diff = &arr1 - &arr2;
    assert_eq!(diff[[0]], -3);
    assert_eq!(diff[[1]], -3);
    assert_eq!(diff[[2]], -3);

    let scalar_diff = &arr1 - 1;
    assert_eq!(scalar_diff[[0]], 0);
    assert_eq!(scalar_diff[[1]], 1);
    assert_eq!(scalar_diff[[2]], 2);
}

#[test]
fn test_multiplication() {
    let arr1 = [1, 2, 3].into_array();
    let arr2 = [4, 5, 6].into_array();
    let prod = &arr1 * &arr2;
    assert_eq!(prod[[0]], 4);
    assert_eq!(prod[[1]], 10);
    assert_eq!(prod[[2]], 18);

    let scalar_prod = &arr1 * 2;
    assert_eq!(scalar_prod[[0]], 2);
    assert_eq!(scalar_prod[[1]], 4);
    assert_eq!(scalar_prod[[2]], 6);
}

#[test]
fn test_division() {
    let arr1 = [1.0, 2.0, 3.0].into_array();
    let arr2 = [4.0, 5.0, 6.0].into_array();
    let quot = &arr1 / &arr2;
    assert_eq!(quot[[0]], 0.25);
    assert_eq!(quot[[1]], 0.4);
    assert_eq!(quot[[2]], 0.5);

    let scalar_quot = &arr1 / 2.0;
    assert_eq!(scalar_quot[[0]], 0.5);
    assert_eq!(scalar_quot[[1]], 1.0);
    assert_eq!(scalar_quot[[2]], 1.5);
}

#[test]
fn test_mat_mul() {
    let arr1 = [
        [0, 1, 2],
        [3, 4, 5]
    ].into_array();
    let arr2 = [
        [1, 0],
        [0, 1],
        [0, 0]
    ].into_array();
    let prod = mat_mul(&arr1, &arr2);
    assert_eq!(prod[[0, 0]], 0);
    assert_eq!(prod[[0, 1]], 1);
    assert_eq!(prod[[1, 0]], 3);
    assert_eq!(prod[[1, 1]], 4);
}
