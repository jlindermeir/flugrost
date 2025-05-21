# Flugrost

Flugrost is a small proof of concept that explores how ndarray like data
structures can be expressed with shapes that live entirely in the Rust type
system.  All shape information is encoded in types which means that common
operations such as broadcasting or index calculations can be verified at compile
time.

## Type level numbers

Dimensions are represented by a tiny type level arithmetic.  Natural numbers are
constructed from `Zero` and the successor type `Succ<N>` and expose their value
through the `Nat` trait:

```rust
pub struct Zero;
pub struct Succ<N>(core::marker::PhantomData<N>);

pub trait Nat {
    const VALUE: usize;
}
```

Higher numbers are created via type aliases such as `Two` or `Three` which are
just nested applications of `Succ`.

## Shapes as linked lists

An array shape is a list of these numbers.  The project uses a right to left
linked list where the last type parameter denotes the leading dimension:

```rust
pub struct DimNil;

pub struct DimCons<Head, Tail>(core::marker::PhantomData<(Head, Tail)>);

pub type Rank1<D0> = DimCons<D0, DimNil>;
pub type Rank2<D0, D1> = DimCons<D1, Rank1<D0>>;
```

The `Shape` trait then computes meta information like `RANK` and
`N_ELEMENTS` and provides helpers to translate between multi dimensional indices
and flat offsets.

## Compile time broadcasting

Numpy style broadcasting is expressed purely in the type system.  Two shapes can
be combined through the `Broadcast` trait which recursively evaluates the rules
for each dimension.  Single dimensions are handled by `BroadcastOneDim` and only
allow either equality or broadcasting from `1`:

```rust
pub trait BroadcastOneDim<Rhs> {
    type Output: Nat;
}

impl<D: GreaterThanZero> BroadcastOneDim<D> for One { type Output = D; }
impl<D: GreaterThanZero> BroadcastOneDim<One> for D { type Output = D; }
impl<D: Nat>              BroadcastOneDim<D> for D   { type Output = D; }
```

Larger shapes build on this to compute a new `Output` shape at compile time.

## The `NDArray` type

`NDArray<S, T>` stores the actual data alongside a phantom marker for its shape.
The constructor checks that the provided buffer matches the expected compile time
size.  Indexing and broadcasting rely on the `Shape` and `Broadcast` machinery:

```rust
pub struct NDArray<S: Shape, T> {
    data: Vec<T>,
    shape_marker: std::marker::PhantomData<S>,
}
```

The `broadcast` method uses the compile time result of `Broadcast` to create a
new array with the desired shape.

Element wise operations such as addition or multiplication are provided via a
macro that performs broadcasting of both operands before applying the operator.

## Why all the type level machinery?

Rust currently only supports arrays indexed by constant integers.  To express
arbitrary shapes and to evaluate broadcasting rules within the type system we
need a representation that can be manipulated at compile time.  The linked list
based shapes together with the small natural number hierarchy make it possible
for the compiler to reason about ranks, element counts and resulting broadcast
shapes.  Errors like mismatched dimensions therefore become compile time errors
instead of runtime panics.

This repository is merely a starting point but demonstrates that a surprising
amount of ndarray functionality can be expressed with the tools available in
stable Rust today.

## Inspiration and status

This crate was inspired by the [dxdy](https://github.com/coreylowman/dfdx)
project by Corey Lowman.  Flugrost is an experiment in type-level ndarray
construction and should not be used for production code.
