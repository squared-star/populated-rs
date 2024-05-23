# Populated collections for Rust

Non-empty collection types with guaranteed 1 or more elements.

A library for working with non-empty collections i.e. populated collections.
Mirrors std collections in its populated versions that guarantee always containing at least one
element in the collection.

These collections are useful when you want to ensure that a collection is never empty. Any attempt
to make it empty will result in a compile-time error. Also, these collections provide additional
guarantees about their length and capacity, which are always non-zero. This means that you do not
need to deal with `Option` in cases where you know that the collection will always have at least
one element. For example, you call `first()` on a `PopulatedVec` and you are guaranteed to get a
reference to the first element without having to deal with `Option`.

## Safe transition to `std` collections when emptying a populated collection

If you invoke an operation that empties a populated collection, the library provides a safe way to
transition to the corresponding `std` collection. For example, if you call `clear()` on a
`PopulatedVec`, it will return the underlying `Vec`. `clear()` on a `PopulatedVec` will take
ownership of the `PopulatedVec` and return the underlying `Vec`. This way any attempt to use
a cleared `PopulatedVec` will result in a compile-time error. At the same time, you can safely
and efficiently transition to the `Vec` when you need to.

## Collections

The following `std` collections have been mirrored in this library:

- `Vec` → `PopulatedVec`
- `Slice` → `PopulatedSlice`
- `BinaryHeap` → `PopulatedBinaryHeap`
- `HashMap` → `PopulatedHashMap`
- `HashSet` → `PopulatedHashSet`
- `BTreeMap` → `PopulatedBTreeMap`
- `BTreeSet` → `PopulatedBTreeSet`
- `VecDeque` → `PopulatedVecDeque`

## Examples

`first()` on a `PopulatedVec` and `PopulatedSlice`:

```rust
use populated::{PopulatedVec, PopulatedSlice};

let vec = PopulatedVec::new(1);
assert_eq!(vec.len().get(), 1);
assert_eq!(vec.first(), &1);

let slice = vec.as_slice();
assert_eq!(slice.len().get(), 1);
assert_eq!(slice.first(), &1);
```

`clear()` on a `BTreeMap`:

```rust
use populated::PopulatedBTreeMap;

let mut map = PopulatedBTreeMap::new("a", 1);
map.insert("b", 2);
assert_eq!(map.len().get(), 2);

// Safe transition to std BTreeMap on clear
let map = map.clear();
assert_eq!(map.len(), 0);
```
