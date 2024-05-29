//! A library for working with non-empty collections i.e. populated collections.
//! Mirrors std collections in its populated versions that guarantee always containing at least one
//! element in the collection.
//!
//! These collections are useful when you want to ensure that a collection is never empty. Any attempt
//! to make it empty will result in a compile-time error. Also, these collections provide additional
//! guarantees about their length and capacity, which are always non-zero. This means that you do not
//! need to deal with `Option` in cases where you know that the collection will always have at least
//! one element. For example, you call `first()` on a `PopulatedVec` and you are guaranteed to get a
//! reference to the first element without having to deal with `Option`.
//!
//! # Safe transition to `std` collections when emptying a populated collection
//!
//! If you invoke an operation that empties a populated collection, the library provides a safe way to
//! transition to the corresponding `std` collection. For example, if you call `clear()` on a
//! `PopulatedVec`, it will return the underlying `Vec`. `clear()` on a `PopulatedVec` will take
//! ownership of the `PopulatedVec` and return the underlying `Vec`. This way any attempt to use
//! a cleared `PopulatedVec` will result in a compile-time error. At the same time, you can safely
//! and efficiently transition to the `Vec` when you need to.
//!
//! # Collections
//!
//! The following `std` collections have been mirrored in this library:
//!
//! - `Vec` → `PopulatedVec`
//! - `Slice` → `PopulatedSlice`
//! - `BinaryHeap` → `PopulatedBinaryHeap`
//! - `HashMap` → `PopulatedHashMap`
//! - `HashSet` → `PopulatedHashSet`
//! - `BTreeMap` → `PopulatedBTreeMap`
//! - `BTreeSet` → `PopulatedBTreeSet`
//! - `VecDeque` → `PopulatedVecDeque`
//!
//! # Examples
//!
//! `first()` on a `PopulatedVec` and `PopulatedSlice`:
//!
//! ```
//! use populated::{PopulatedVec, PopulatedSlice};
//!
//! let vec = PopulatedVec::new(1);
//! assert_eq!(vec.len().get(), 1);
//! assert_eq!(vec.first(), &1);
//!
//! let slice = vec.as_slice();
//! assert_eq!(slice.len().get(), 1);
//! assert_eq!(slice.first(), &1);
//! ```
//!
//! `clear()` on a `BTreeMap`:
//!
//! ```
//! use populated::PopulatedBTreeMap;
//!
//! let mut map = PopulatedBTreeMap::new("a", 1);
//! map.insert("b", 2);
//! assert_eq!(map.len().get(), 2);
//!
//! // Safe transition to std BTreeMap on clear
//! let map = map.clear();
//! assert_eq!(map.len(), 0);
//! ```

use std::{
    borrow::{Borrow, BorrowMut, Cow},
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, TryReserveError, VecDeque},
    hash::{BuildHasher, Hash, RandomState},
    num::NonZeroUsize,
    ops::{BitOr, Deref, DerefMut, Index, IndexMut, RangeBounds, RangeFull},
};

/// A non-empty `Vec` with at least one element.
#[derive(PartialEq, PartialOrd, Eq, Ord, Clone, Debug)]
pub struct PopulatedVec<T>(Vec<T>);

impl<T> From<PopulatedVec<T>> for Vec<T> {
    fn from(populated_vec: PopulatedVec<T>) -> Vec<T> {
        populated_vec.0
    }
}

impl<T> TryFrom<Vec<T>> for PopulatedVec<T> {
    type Error = Vec<T>;

    /// Attempts to create a `PopulatedVec` from a `Vec`. If the `Vec` is empty, it returns the original `Vec` as an error.
    /// Otherwise, it returns a `PopulatedVec` with `Ok`.
    ///
    /// ```
    /// use populated::PopulatedVec;
    ///
    /// let vec = vec![1];
    /// let vec = PopulatedVec::try_from(vec).unwrap();
    /// assert_eq!(vec.len().get(), 1);
    /// assert_eq!(vec.first(), &1);
    /// ```
    ///
    /// If the `Vec` is empty, it returns the original `Vec` as an error.
    ///
    /// ```
    /// use populated::PopulatedVec;
    ///
    /// let vec: Vec<u64> = vec![];
    /// let vec = PopulatedVec::try_from(vec);
    /// assert_eq!(vec, Err(vec![]));
    /// ```
    fn try_from(vec: Vec<T>) -> Result<PopulatedVec<T>, Self::Error> {
        if vec.is_empty() {
            Err(vec)
        } else {
            Ok(PopulatedVec(vec))
        }
    }
}

impl<T> PopulatedVec<T> {
    /// Constructs a `Vec<T>` populated with a single element.
    pub fn new(value: T) -> PopulatedVec<T> {
        PopulatedVec(vec![value])
    }

    /// Creates a new `PopulatedVec` with the specified capacity.
    /// The capacity must be non-zero. This creates a `PopulatedVec` with the specified capacity and the first element initialized
    /// with the provided value.
    pub fn with_capacity(capacity: NonZeroUsize, value: T) -> PopulatedVec<T> {
        let vec = Vec::with_capacity(capacity.get());
        PopulatedVec::pushed(vec, value)
    }

    /// Constructs a `PopulatedVec` from a `Vec<T>` by pushing a single element to it.
    ///
    /// ```
    /// use populated::PopulatedVec;
    ///
    /// let vec = vec![1];
    /// let vec = PopulatedVec::pushed(vec, 2);
    /// assert_eq!(vec.len().get(), 2);
    /// assert_eq!(vec.first(), &1);
    /// assert_eq!(vec.last(), &2);
    /// ```
    pub fn pushed(mut vec: Vec<T>, value: T) -> PopulatedVec<T> {
        vec.push(value);
        PopulatedVec(vec)
    }

    /// Constructs a `PopulatedVec` from a `Vec<T>` by inserting a single element at the specified index.
    /// If the index is out of bounds, it will panic.
    /// If the index is within bounds, it will insert the element at the specified index.
    ///
    /// ```
    /// use populated::PopulatedVec;
    ///
    /// let vec = vec![1, 3];
    /// let vec = PopulatedVec::inserted(vec, 1, 2);
    /// assert_eq!(vec.len().get(), 3);
    /// assert_eq!(vec.first(), &1);
    /// assert_eq!(vec.last(), &3);
    /// ```
    pub fn inserted(mut vec: Vec<T>, index: usize, value: T) -> PopulatedVec<T> {
        vec.insert(index, value);
        PopulatedVec(vec)
    }

    /// Constructs a `PopulatedVec` from a `Vec<T>` by providing the first element and the rest of the elements from an array.
    ///
    /// ```
    /// use populated::PopulatedVec;
    ///
    /// let vec = PopulatedVec::from_first_and_rest_array(1, [2, 3]);
    /// assert_eq!(vec.len().get(), 3);
    /// assert_eq!(vec.first(), &1);
    /// assert_eq!(vec.last(), &3);
    /// ```
    pub fn from_first_and_rest_array<const N: usize>(first: T, rest: [T; N]) -> PopulatedVec<T> {
        let mut vec = PopulatedVec::with_capacity(NonZeroUsize::MIN.saturating_add(N), first);
        vec.extend_from_array(rest);
        vec
    }

    /// Returns the underlying `Vec<T>`.
    pub fn into_inner(self) -> Vec<T> {
        self.0
    }

    /// Appends an element to the back of a collection.
    pub fn push(&mut self, value: T) {
        self.0.push(value);
    }

    /// Appends elements from a slice to the back of a collection.
    /// This is a safe operation because this is a `PopulatedVec`.
    ///
    /// ```
    /// use populated::PopulatedVec;
    ///
    /// let mut vec = PopulatedVec::new(1);
    /// vec.extend_from_slice(&[2, 3]);
    /// assert_eq!(vec.len().get(), 3);
    /// assert_eq!(vec.last(), &3);
    /// ```
    pub fn extend_from_array<const N: usize>(&mut self, array: [T; N]) {
        self.reserve(N);
        for value in array {
            self.push(value);
        }
    }

    /// Removes the last element from the vector and returns it, along with
    /// the remaining vector. Note that this is a safe operation because
    /// this is a `PopulatedVec`.
    ///
    /// ```
    /// use populated::PopulatedVec;
    ///
    /// let vec = PopulatedVec::new(1);
    /// let (vec, last) = vec.pop();
    /// assert_eq!(last, 1);
    /// assert_eq!(vec.len(), 0);
    /// ```
    ///
    /// # Returns
    /// A tuple containing `Vec` (with the last element removed) and the last element of the vector
    /// The returned `Vec` has no additional guarantees about its length.
    pub fn pop(self) -> (Vec<T>, T) {
        let mut vec = self.0;
        let last = vec.pop().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedVec
        (vec, last)
    }

    /// Returns the first element of the slice. This is a safe operation
    /// because this is a `PopulatedVec`.
    ///
    /// ```
    /// use populated::PopulatedVec;
    ///
    /// let vec = PopulatedVec::new(1);
    /// assert_eq!(vec.first(), &1);
    /// ```
    ///
    /// # Returns
    /// A reference to the first element of the vector.
    pub fn first(&self) -> &T {
        self.0.first().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedVec
    }

    /// Returns the last element of the slice. This is a safe operation
    /// because this is a `PopulatedVec`.
    ///
    /// ```
    /// use populated::PopulatedVec;
    ///
    /// let vec = PopulatedVec::new(1);
    /// assert_eq!(vec.last(), &1);
    /// ```
    ///
    /// # Returns
    /// A reference to the last element of the vector.
    pub fn last(&self) -> &T {
        self.0.last().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedVec
    }

    /// Returns the number of elements in the vector, also referred to as its
    /// ‘length’. It is non-zero because this is a `PopulatedVec`.
    ///
    /// ```
    /// use populated::PopulatedVec;
    ///
    /// let vec = PopulatedVec::new(1);
    /// assert_eq!(vec.len().get(), 1);
    /// ```
    ///
    /// # Returns
    /// The number of elements in the vector.
    pub fn len(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.0.len()).unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedVec
    }

    /// Returns the total number of elements the vector can hold without
    /// reallocating. Since this is a `PopulatedVec`, it is guaranteed to
    /// be non-zero.
    pub fn capacity(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.0.capacity()).unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedVec
    }

    /// Reserves capacity for at least `additional` more elements to be inserted.
    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    /// Reserves the minimum capacity for exactly `additional` more elements to be inserted.
    pub fn reserve_exact(&mut self, additional: usize) {
        self.0.reserve_exact(additional);
    }

    /// Tries to reserve capacity for at least `additional` more elements to be inserted.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.0.try_reserve(additional)
    }

    /// Tries to reserve the minimum capacity for exactly `additional` more elements to be inserted.
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.0.try_reserve_exact(additional)
    }

    /// Shrinks the capacity of the vector as much as possible.
    pub fn shrink_to_fit(&mut self) {
        self.0.shrink_to_fit();
    }

    /// Shrinks the capacity of the vector to the minimum allowed capacity.
    /// Note that this is a safe operation because capacity is guaranteed to be non-zero.
    pub fn shrink_to(&mut self, min_capacity: NonZeroUsize) {
        self.0.shrink_to(min_capacity.get());
    }

    /// Shortens the vector, keeping the first `len` elements and dropping the rest.
    /// Note that this is a safe operation because length is guaranteed to be non-zero.
    pub fn truncate(&mut self, len: NonZeroUsize) {
        self.0.truncate(len.get());
    }

    /// Shortens the vector, keeping the first `len` elements and dropping the rest.
    /// The truncated `Vec` is returned.
    ///
    /// ```
    /// use populated::PopulatedVec;
    ///
    /// let vec = PopulatedVec::new(1);
    /// let vec = vec.truncate_into(0);
    /// assert_eq!(vec.len(), 0);
    /// ```
    ///
    /// # Returns
    /// A new `Vec` with the first `len` elements of the original vector.
    /// The returned `Vec` has no additional guarantees about its length.
    pub fn truncate_into(self, len: usize) -> Vec<T> {
        let mut vec = self.0;
        vec.truncate(len);
        vec
    }

    /// Extracts a slice containing the entire vector.
    ///
    /// ```
    /// use populated::PopulatedVec;
    ///
    /// let vec = PopulatedVec::new(1);
    /// let slice = vec.as_slice();
    /// assert_eq!(slice.len().get(), 1);
    /// ```
    ///
    /// # Returns
    /// A `PopulatedSlice` containing the entire vector. The returned `PopulatedSlice` guarantees that it is non-empty.
    pub fn as_slice(&self) -> &PopulatedSlice<T> {
        self.0.deref().try_into().unwrap()
    }

    /// Extracts a mutable slice containing the entire vector.
    ///
    /// ```
    /// use populated::PopulatedVec;
    ///
    /// let mut vec = PopulatedVec::new(1);
    /// let slice = vec.as_mut_slice();
    /// assert_eq!(slice.len().get(), 1);
    /// ```
    ///
    /// # Returns
    /// A mutable `PopulatedSlice` containing the entire vector. The returned `PopulatedSlice` guarantees that it is non-empty.
    pub fn as_mut_slice(&mut self) -> &mut PopulatedSlice<T> {
        self.0.deref_mut().try_into().unwrap()
    }

    /// Removes an element from the vector and returns it along with the
    /// remaining `Vec`.
    ///
    /// The removed element is replaced by the last element of the vector.
    ///
    /// This does not preserve ordering, but is O(1). If you need to
    /// preserve the element order, use remove instead.
    ///
    /// # Panics
    ///
    /// Panics if the `index` is out of bounds.
    pub fn swap_remove(self, index: usize) -> (T, Vec<T>) {
        let mut vec = self.0;
        let removed = vec.swap_remove(index);
        (removed, vec)
    }

    /// Inserts an element at position index within the vector, shifting
    /// all elements after it to the right.
    pub fn insert(&mut self, index: usize, element: T) {
        self.0.insert(index, element);
    }

    /// Removes and returns the element at position index within the
    /// populated vector, shifting all elements after it to the left and
    /// returns the resulting `Vec`.
    ///
    /// Note: Because this shifts over the remaining elements, it has a
    /// worst-case performance of O(n). If you don’t need the order of
    /// elements to be preserved, use swap_remove instead. If you’d like
    /// to remove elements from the beginning of the Vec, consider using
    /// `VecDeque::pop_front` instead.
    pub fn remove(self, index: usize) -> (T, Vec<T>) {
        let mut vec = self.0;
        let removed = vec.remove(index);
        (removed, vec)
    }

    /// Retains only the elements specified by the predicate in the
    /// returned `Vec`.
    ///
    /// In other words, remove all elements `e` for which `f(&e)` returns
    /// false. This method operates in place, visiting each element
    /// exactly once in the original order, and preserves the order of
    /// the retained elements.
    pub fn retain(self, f: impl FnMut(&T) -> bool) -> Vec<T> {
        let mut vec = self.0;
        vec.retain(f);
        vec
    }

    /// Retains only the elements specified by the predicate in the returned
    /// `Vec`, passing a mutable reference to it.
    ///
    /// In other words, remove all elements `e` such that `f(&mut e)`
    /// returns false. This method operates in place, visiting each
    /// element exactly once in the original order, and preserves the
    /// order of the retained elements.
    pub fn retain_mut(self, f: impl FnMut(&mut T) -> bool) -> Vec<T> {
        let mut vec = self.0;
        vec.retain_mut(f);
        vec
    }

    /// Removes all but the first of consecutive elements in the vector that resolve to the same key.
    ///
    /// If the vector is sorted, this removes all duplicates.
    pub fn dedup_by_key<K: PartialEq>(&mut self, key: impl FnMut(&mut T) -> K) {
        self.0.dedup_by_key(key);
    }

    /// Removes all but the first of consecutive elements in the vector satisfying a given equality relation.
    ///
    /// The `same_bucket` function is passed references to two elements from the
    /// vector and must determine if the elements compare equal. The elements
    /// are passed in opposite order from their order in the slice, so if
    /// `same_bucket(a, b)` returns `true`, `a` is removed.
    ///
    /// If the vector is sorted, this removes all duplicates.
    pub fn dedup_by(&mut self, same_bucket: impl FnMut(&mut T, &mut T) -> bool) {
        self.0.dedup_by(same_bucket);
    }

    /// Moves all the elements of other into self, leaving other empty.
    pub fn append(&mut self, other: &mut Vec<T>) {
        self.0.append(other);
    }

    /// Unpopulates the vector, returning the underlying `Vec`.
    pub fn clear(self) -> Vec<T> {
        let mut vec = self.0;
        vec.clear();
        vec
    }

    pub fn split_off(&mut self, at: NonZeroUsize) -> Vec<T> {
        self.0.split_off(at.get())
    }

    pub fn split_into(self, at: usize) -> (Vec<T>, Vec<T>) {
        let mut vec = self.0;
        let other = vec.split_off(at);
        (vec, other)
    }

    // pub fn splice<I: IntoIterator<Item = T>>(
    //     &mut self,
    //     range: impl RangeBounds<usize>,
    //     replace_with: I,
    // ) -> Splice<'_, I::IntoIter> {
    //     self.0.splice(range, replace_with)
    // }
}
impl<T: Clone> PopulatedVec<T> {
    /// Resizes the vector in place so that `len` is equal to `new_len`.
    /// If `new_len` is greater than `len`, the vector is extended by the
    /// difference, with each additional slot filled with `value`. If `new_len`
    /// is less than `len`, the vector is simply truncated.
    pub fn resize(&mut self, new_len: NonZeroUsize, value: T) {
        self.0.resize(new_len.get(), value);
    }

    /// Resizes the vector in place so that `len` is equal to `new_len`.
    /// If `new_len` is greater than `len`, the vector is extended by the
    /// difference, with each additional slot filled with `value`. If `new_len`
    /// is less than `len`, the vector is simply truncated.
    ///
    /// This is a safe operation because this it returns the underlying `Vec` that can be empty.
    pub fn resize_into(self, new_len: usize, value: T) -> Vec<T> {
        let mut vec = self.0;
        vec.resize(new_len, value);
        vec
    }

    /// Clones and appends all elements in a slice to the Vec.
    ///
    /// Iterates over the slice other, clones each element, and then appends
    /// it to this Vec. The other slice is traversed in-order.
    ///
    /// Note that this function is same as extend except that it is
    /// specialized to work with slices instead. If and when Rust gets
    /// specialization this function will likely be deprecated (but still
    /// available).
    pub fn extend_from_slice(&mut self, other: &[T]) {
        self.0.extend_from_slice(other);
    }

    /// Copies elements from src range to the end of the vector.
    pub fn extend_from_within(&mut self, src: impl RangeBounds<usize>) {
        self.0.extend_from_within(src);
    }
}

impl<T: PartialEq> PopulatedVec<T> {
    /// Removes consecutive repeated elements in the vector according to the
    /// `PartialEq` trait implementation.
    ///
    /// If the vector is sorted, this removes all duplicates.
    pub fn dedup(&mut self) {
        self.0.dedup();
    }
}

impl<T> Index<usize> for PopulatedVec<T> {
    type Output = T;

    /// Performs the indexing (`container[index]`) operation.
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> Index<NonZeroUsize> for PopulatedVec<T> {
    type Output = T;

    /// Performs the indexing (`container[index]`) operation for a non-zero index.
    fn index(&self, index: NonZeroUsize) -> &Self::Output {
        &self.0[index.get() - 1]
    }
}

impl<T> IndexMut<usize> for PopulatedVec<T> {
    /// Performs the mutable indexing (`container[index]`) operation.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T> Index<RangeFull> for PopulatedVec<T> {
    type Output = PopulatedSlice<T>;

    /// Performs the indexing (`container[..]`) operation to get a slice of the complete vector.
    fn index(&self, _: RangeFull) -> &Self::Output {
        self.deref()
    }
}

impl<T> IndexMut<RangeFull> for PopulatedVec<T> {
    /// Performs the mutable indexing (`container[..]`) operation to get a mutable reference to get a slice of the complete vector.
    fn index_mut(&mut self, _: RangeFull) -> &mut Self::Output {
        self.deref_mut()
    }
}

impl<T> PopulatedVec<T> {
    /// Returns a populated iterator over the elements of the vector.
    pub fn iter(&self) -> crate::slice::PopulatedIter<'_, T> {
        self.into_populated_iter()
    }

    /// Returns a mutable populated iterator over the elements of the vector.
    pub fn iter_mut(&mut self) -> slice::PopulatedIterMut<T> {
        self.into_populated_iter()
    }
}

impl<T> Extend<T> for PopulatedVec<T> {
    /// Extends the vector with the contents of an iterator.
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.0.extend(iter);
    }
}

impl<T, E> FromPopulatedIterator<Result<T, E>> for Result<PopulatedVec<T>, E> {
    fn from_populated_iter(iter: impl IntoPopulatedIterator<Item = Result<T, E>>) -> Self {
        let vec = Result::from_iter(iter.into_iter())?;
        Ok(PopulatedVec(vec))
    }
}

/// A macro pvec! to create a PopulatedVec with a list of elements.
///
/// # Examples
///
/// ```
/// use populated::pvec;
/// let vec = pvec![1, 2, 3];
/// assert_eq!(vec.len().get(), 3);
/// assert_eq!(vec.first(), &1);
/// assert_eq!(vec.last(), &3);
/// ```
///
/// Works with a single element
///
/// ```
/// use populated::pvec;
/// let vec = pvec![1];
/// assert_eq!(vec.len().get(), 1);
/// assert_eq!(vec.first(), &1);
/// assert_eq!(vec.last(), &1);
/// ```
///
/// Works with trailing comma
///
/// ```
/// use populated::pvec;
/// let vec = pvec![1, 2, 3,];
/// assert_eq!(vec.len().get(), 3);
/// assert_eq!(vec.first(), &1);
/// assert_eq!(vec.last(), &3);
/// ```
///
/// Fails to compile if no elements are provided
///
/// ```compile_fail
/// use populated::pvec;
/// // This will result in a compile-time error. Test should pass if the code fails to compile.
/// let vec = pvec![];
/// ```
#[macro_export]
macro_rules! pvec {
    () => {
        compile_error!("PopulatedVec requires at least one element");
    };
    ($first: expr $(,)?) => {
        $crate::PopulatedVec::new($first)
    };
    ($first:expr $(, $item:expr)+ $(,)? ) => ($crate::PopulatedVec::from_first_and_rest_array($first, [$($item),*]));
}

#[derive(PartialEq, PartialOrd, Eq, Ord, Debug)]
#[repr(transparent)]
pub struct PopulatedSlice<T>([T]);

impl<'a, T> From<&'a PopulatedSlice<T>> for &'a [T] {
    /// Convert a PopulatedSlice to a slice.
    fn from(populated_slice: &PopulatedSlice<T>) -> &[T] {
        &populated_slice.0
    }
}

impl<'a, T> From<&'a mut PopulatedSlice<T>> for &'a mut [T] {
    /// Convert a mutable PopulatedSlice to a mutable slice.
    fn from(populated_slice: &mut PopulatedSlice<T>) -> &mut [T] {
        &mut populated_slice.0
    }
}

#[derive(Debug)]
pub struct UnpopulatedError;

impl<'a, T> TryFrom<&'a [T]> for &'a PopulatedSlice<T> {
    type Error = UnpopulatedError;

    /// Convert a slice to a PopulatedSlice. If the slice is empty, an error is returned.
    /// If the slice is not empty, a reference to the PopulatedSlice is returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use populated::PopulatedSlice;
    /// use std::convert::TryFrom;
    ///
    /// let slice: &[u64] = &[1, 2, 3];
    /// let populated_slice: &PopulatedSlice<u64> = TryFrom::try_from(slice).unwrap();
    /// assert_eq!(populated_slice.len().get(), 3);
    /// ```
    ///
    /// ```
    /// use populated::{PopulatedSlice, UnpopulatedError};
    /// use std::convert::TryFrom;
    ///
    /// let slice: &[u64] = &[];
    /// let populated_slice: Result<&PopulatedSlice<u64>, UnpopulatedError> = TryFrom::try_from(slice);
    /// assert!(populated_slice.is_err());
    /// ```
    fn try_from(slice: &'a [T]) -> Result<&'a PopulatedSlice<T>, Self::Error> {
        if slice.is_empty() {
            Err(UnpopulatedError)
        } else {
            Ok(unsafe { &*(slice as *const [T] as *const PopulatedSlice<T>) })
        }
    }
}

impl<'a, T> TryFrom<&'a mut [T]> for &'a mut PopulatedSlice<T> {
    type Error = UnpopulatedError;

    fn try_from(slice: &'a mut [T]) -> Result<&'a mut PopulatedSlice<T>, Self::Error> {
        if slice.is_empty() {
            Err(UnpopulatedError)
        } else {
            Ok(unsafe { &mut *(slice as *mut [T] as *mut PopulatedSlice<T>) })
        }
    }
}

impl<T> Index<usize> for PopulatedSlice<T> {
    type Output = T;

    /// Index into the populated slice.
    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> IndexMut<usize> for PopulatedSlice<T> {
    /// Index into the populated slice mutably.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T> PopulatedSlice<T> {
    /// Returns a populated iterator over the slice.
    pub fn iter(&self) -> slice::PopulatedIter<T> {
        self.into_populated_iter()
    }

    /// Returns a mutable populated iterator over the slice.
    pub fn iter_mut(&mut self) -> slice::PopulatedIterMut<T> {
        self.into_populated_iter()
    }

    /// Returns the length of the populated slice.
    ///
    /// ```
    /// use populated::PopulatedSlice;
    /// use std::num::NonZeroUsize;
    ///
    /// let slice: &[u64] = &[1, 2, 3];
    /// let slice: &PopulatedSlice<u64> = TryFrom::try_from(slice).unwrap();
    /// assert_eq!(slice.len(), NonZeroUsize::new(3).unwrap());
    /// ```
    pub fn len(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.0.len()).unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedSlice
    }

    /// Returns the first element of the populated slice.
    pub fn first(&self) -> &T {
        self.0.first().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedSlice
    }

    /// Returns a mutable pointer to the first element of the populated
    /// slice.
    pub fn first_mut(&mut self) -> &mut T {
        self.0.first_mut().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedSlice
    }

    /// Returns the first and all the rest of the elements of the populated slice.
    /// Note that the returned slice can be empty. It is not a PopulatedSlice.
    pub fn split_first(&self) -> (&T, &[T]) {
        self.0.split_first().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedSlice
    }

    /// Returns the first and all the rest of the elements of the populated slice. The returned
    /// first element and the rest of the elements are mutable.
    /// Note that the returned slice can be empty. It is not a PopulatedSlice.
    pub fn split_first_mut(&mut self) -> (&mut T, &mut [T]) {
        self.0.split_first_mut().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedSlice
    }

    /// Returns the last and all the rest of the elements of the populated slice.
    /// Note that the returned slice can be empty. It is not a PopulatedSlice.
    pub fn split_last(&self) -> (&T, &[T]) {
        self.0.split_last().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedSlice
    }

    /// Returns the mutable last and all the rest of the elements of the populated slice as a mutable slice.
    /// Note that the returned slice can be empty. It is not a PopulatedSlice.
    pub fn split_last_mut(&mut self) -> (&mut T, &mut [T]) {
        self.0.split_last_mut().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedSlice
    }

    /// Returns the last element of the populated slice.
    pub fn last(&self) -> &T {
        self.0.last().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedSlice
    }

    /// Returns a mutable reference to the last item in the populated slice.
    pub fn last_mut(&mut self) -> &mut T {
        self.0.last_mut().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedSlice
    }

    /// Swaps two elements in the slice.
    ///
    /// If a equals to b, it’s guaranteed that elements won’t change
    /// value.
    pub fn swap(&mut self, i: usize, j: usize) {
        self.0.swap(i, j);
    }

    /// Reverses the order of elements in the slice, in place.
    pub fn reverse(&mut self) {
        self.0.reverse();
    }

    /// Returns a reference to an element or subslice depending on the type of
    /// index. Since mid is a NonZeroUsize, it is guaranteed that the first slice is
    /// populated.
    pub fn split_at_populated(&self, mid: NonZeroUsize) -> (&PopulatedSlice<T>, &[T]) {
        let (left, right) = self.0.split_at(mid.get());
        (
            unsafe { &*(left as *const [T] as *const PopulatedSlice<T>) },
            right,
        )
    }

    /// Divides one slice into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding the
    /// index `mid` itself) and the second will contain all indices from
    /// `[mid, len)` (excluding the index `len` itself).
    pub fn split_at(&self, mid: usize) -> (&[T], &[T]) {
        self.0.split_at(mid)
    }

    /// Returns a mutable reference to an element or subslice depending on the
    /// type of index. Since mid is a NonZeroUsize, it is guaranteed that the first slice is
    /// populated.
    pub fn split_at_mut_populated(
        &mut self,
        mid: NonZeroUsize,
    ) -> (&mut PopulatedSlice<T>, &mut [T]) {
        let (left, right) = self.0.split_at_mut(mid.get());
        (
            unsafe { &mut *(left as *mut [T] as *mut PopulatedSlice<T>) },
            right,
        )
    }

    /// Divides one mutable slice into two at an index.
    ///
    /// The first will contain all indices from `[0, mid)` (excluding the
    /// index `mid` itself) and the second will contain all indices from
    /// `[mid, len)` (excluding the index `len` itself).
    pub fn split_at_mut(&mut self, mid: usize) -> (&mut [T], &mut [T]) {
        self.0.split_at_mut(mid)
    }

    /// Binary searches this populated slice with a comparator function.
    ///
    /// The comparator function should return an order code that indicates
    /// whether its argument is `Less`, `Equal` or `Greater` the desired target. If
    /// the slice is not sorted or if the comparator function does not
    /// implement an order consistent with the sort order of the underlying
    /// slice, the returned result is unspecified and meaningless.
    ///
    /// If the value is found then Result::Ok is returned, containing the
    /// index of the matching element. If there are multiple matches, then any
    /// one of the matches could be returned. The index is chosen
    /// deterministically, but is subject to change in future versions of
    /// Rust. If the value is not found then Result::Err is returned,
    /// containing the index where a matching element could be inserted
    /// while maintaining sorted order.
    ///
    /// See also `binary_search`, `binary_search_by_key`, and `partition_point`.
    pub fn binary_search_by(&self, f: impl FnMut(&T) -> Ordering) -> Result<usize, usize> {
        self.0.binary_search_by(f)
    }

    /// Binary searches this populated slice with a key extraction function.
    ///
    /// Assumes that the slice is sorted by the key, for instance with sort_by_key using the same key extraction function. If the slice is
    /// not sorted by the key, the returned result is unspecified and meaningless.
    ///
    /// If the value is found then Result::Ok is returned, containing the index of the matching element. If there are multiple matches, then
    /// any one of the matches could be returned. The index is chosen deterministically, but is subject to change in future versions of Rust.
    /// If the value is not found then Result::Err is returned, containing the index where a matching element could be inserted while
    /// maintaining sorted order.
    ///
    /// See also `binary_search`, `binary_search_by`, and `partition_point`.
    pub fn binary_search_by_key<K: Ord>(
        &self,
        key: &K,
        f: impl FnMut(&T) -> K,
    ) -> Result<usize, usize> {
        self.0.binary_search_by_key(key, f)
    }

    /// Sorts the populated slice with a comparator function, but might not preserve the order of equal elements.
    ///
    /// This sort is unstable (i.e., may reorder equal elements), in-place (i.e., does not allocate), and O(n \* log(n)) worst-case.
    ///
    /// The comparator function must define a total ordering for the elements in the slice. If the ordering is not total, the order of the
    /// elements is unspecified. An order is a total order if it is (for all a, b and c):
    ///
    /// - total and antisymmetric: exactly one of a < b, a == b or a > b is true, and
    /// - transitive, a < b and b < c implies a < c. The same must hold for both == and >.
    ///
    /// For example, while f64 doesn’t implement Ord because NaN != NaN, we can use partial_cmp as our sort function when we know the slice
    /// doesn’t contain a NaN.
    pub fn sort_unstable_by(&mut self, f: impl FnMut(&T, &T) -> Ordering) {
        self.0.sort_unstable_by(f);
    }

    /// Sorts the populated slice with a key extraction function, but might not preserve the order of equal elements.
    ///
    /// This sort is unstable (i.e., may reorder equal elements), in-place (i.e., does not allocate), and O(m \* n \* log(n)) worst-case,
    /// where the key function is O(m)
    pub fn sort_unstable_by_key<K: Ord>(&mut self, f: impl FnMut(&T) -> K) {
        self.0.sort_unstable_by_key(f);
    }

    pub fn select_nth_unstable_by(
        &mut self,
        index: usize,
        compare: impl FnMut(&T, &T) -> Ordering,
    ) -> (&mut [T], &mut T, &mut [T]) {
        self.0.select_nth_unstable_by(index, compare)
    }

    pub fn select_nth_unstable_by_key<K: Ord>(
        &mut self,
        index: usize,
        key: impl FnMut(&T) -> K,
    ) -> (&mut [T], &mut T, &mut [T]) {
        self.0.select_nth_unstable_by_key(index, key)
    }

    /// Rotates the slice in-place such that the first mid elements of
    /// the slice move to the end while the last `self.len() - mid`
    /// elements move to the front. After calling `rotate_left`, the
    /// element previously at index `mid` will become the first element in
    /// the slice.
    pub fn rotate_left(&mut self, mid: usize) {
        self.0.rotate_left(mid);
    }

    /// Rotates the slice in-place such that the first `self.len() - k`
    /// elements of the slice move to the end while the last `k` elements
    /// move to the front. After calling `rotate_right`, the element
    /// previously at index `self.len() - k` will become the first
    /// element in the slice.
    pub fn rotate_right(&mut self, mid: usize) {
        self.0.rotate_right(mid);
    }

    /// Fills self with elements returned by calling a closure repeatedly.
    ///
    /// This method uses a closure to create new values. If you’d rather
    /// `Clone` a given value, use `fill`. If you want to use the
    /// `Default` trait to generate values, you can pass
    /// `Default::default` as the argument.
    pub fn fill_with(&mut self, f: impl FnMut() -> T) {
        self.0.fill_with(f);
    }

    /// Swaps all elements in self with those in other.
    ///
    /// The length of other must be the same as self.
    pub fn swap_with_slice(&mut self, other: &mut PopulatedSlice<T>) {
        self.0.swap_with_slice(&mut other.0);
    }

    pub fn partition_point(&self, mut f: impl FnMut(&T) -> bool) -> usize {
        self.0.partition_point(|x| f(x))
    }

    /// Sorts the populated slice with a comparator function.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and
    /// O(n \* log(n)) worst-case.
    ///
    /// The comparator function must define a total ordering for the
    /// elements in the slice. If the ordering is not total, the order of
    /// the elements is unspecified. An order is a total order if it is
    /// (for all `a`, `b` and `c`):
    ///
    /// - total and antisymmetric: exactly one of `a < b`, `a == b` or
    ///   `a > b` is true, and
    /// - transitive, `a < b` and `b < c` implies `a < c`. The same must
    ///   hold for both `==` and `>`.
    ///
    /// For example, while `f64` doesn’t implement `Ord` because `NaN != NaN`,
    /// we can use `partial_cmp` as our sort function when we know the
    /// slice doesn’t contain a `NaN`.
    pub fn sort_by(&mut self, compare: impl FnMut(&T, &T) -> Ordering) {
        self.0.sort_by(compare);
    }

    /// Sorts the slice with a key extraction function.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and
    /// O(m \* n \* log(n)) worst-case, where the key function is O(m).
    ///
    /// For expensive key functions (e.g. functions that are not simple
    /// property accesses or basic operations), `sort_by_cached_key` is
    /// likely to be significantly faster, as it does not recompute
    /// element keys.
    ///
    /// When applicable, unstable sorting is preferred because it is
    /// generally faster than stable sorting and it doesn’t allocate
    /// auxiliary memory. See `sort_unstable_by_key`.
    pub fn sort_by_key<K: Ord>(&mut self, key: impl FnMut(&T) -> K) {
        self.0.sort_by_key(key);
    }

    /// Sorts the slice with a key extraction function.
    ///
    /// During sorting, the key function is called at most once per
    /// element, by using temporary storage to remember the results of
    /// key evaluation. The order of calls to the key function is
    /// unspecified and may change in future versions of the standard
    /// library.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and
    /// O(m \* n + n \* log(n)) worst-case, where the key function is
    /// O(m).
    ///
    /// For simple key functions (e.g., functions that are property accesses
    /// or basic operations), `sort_by_key` is likely to be faster.
    pub fn sort_by_cached_key<K: Ord>(&mut self, key: impl FnMut(&T) -> K) {
        self.0.sort_by_cached_key(key);
    }
}

impl<T: Clone> PopulatedSlice<T> {
    pub fn fill(&mut self, value: T) {
        self.0.fill(value);
    }

    /// Copies the elements from src into self.
    ///
    /// The length of src must be the same as self.
    pub fn clone_from_slice(&mut self, src: &PopulatedSlice<T>) {
        self.0.clone_from_slice(&src.0);
    }

    /// Copies `self` into a new `Vec`.
    pub fn to_vec(&self) -> PopulatedVec<T> {
        PopulatedVec(self.0.to_vec())
    }
}

impl<T: Copy> PopulatedSlice<T> {
    /// Copies all elements from src into self, using a memcpy.
    ///
    /// The length of src must be the same as self.
    ///
    /// If T does not implement Copy, use clone_from_slice.
    pub fn copy_from_slice(&mut self, src: &PopulatedSlice<T>) {
        self.0.copy_from_slice(&src.0);
    }

    /// Copies elements from one part of the slice to another part of itself, using a memmove.
    ///
    /// `src` is the range within `self` to copy from. `dest` is the starting
    /// index of the range within self to copy to, which will have the same
    /// length as src. The two ranges may overlap. The ends of the two ranges
    /// must be less than or equal to `self.len()`.
    pub fn copy_within(&mut self, src: impl RangeBounds<usize>, dest: usize) {
        self.0.copy_within(src, dest);
    }

    pub fn repeat(&self, n: NonZeroUsize) -> PopulatedVec<T> {
        PopulatedVec(self.0.repeat(n.get()))
    }
}

impl<T: PartialEq> PopulatedSlice<T> {
    /// Returns true if the slice contains an element with the given
    /// value.
    ///
    /// This operation is O(n).
    ///
    /// Note that if you have a sorted slice, `binary_search` may be
    /// faster.
    pub fn contains(&self, x: &T) -> bool {
        self.0.contains(x)
    }

    /// Returns true if needle is a prefix of the slice or equal to the
    /// slice.
    pub fn starts_with(&self, needle: &[T]) -> bool {
        self.0.starts_with(needle)
    }

    /// Returns true if needle is a suffix of the slice or equal to the
    /// slice.
    pub fn ends_with(&self, needle: &[T]) -> bool {
        self.0.ends_with(needle)
    }
}

impl<T: Ord> PopulatedSlice<T> {
    /// Binary searches this slice for a given element. If the slice is not
    /// sorted, the returned result is unspecified and meaningless.
    ///
    /// If the value is found then `Result::Ok` is returned, containing the
    /// index of the matching element. If there are multiple matches, then
    /// any one of the matches could be returned. The index is chosen
    /// deterministically, but is subject to change in future versions of
    /// Rust. If the value is not found then Result::Err is returned,
    /// containing the index where a matching element could be inserted while
    /// maintaining sorted order.
    ///
    /// See also `binary_search_by`, `binary_search_by_key`, and
    /// `partition_point`.
    pub fn binary_search(&self, x: &T) -> Result<usize, usize> {
        self.0.binary_search(x)
    }

    /// Sorts the slice, but might not preserve the order of equal elements.
    ///
    /// This sort is unstable (i.e., may reorder equal elements),
    /// in-place (i.e., does not allocate), and O(n * log(n)) worst-case.
    pub fn sort_unstable(&mut self) {
        self.0.sort_unstable();
    }

    pub fn select_nth_unstable(&mut self, index: usize) {
        self.0.select_nth_unstable(index);
    }

    /// Sorts the slice.
    ///
    /// This sort is stable (i.e., does not reorder equal elements) and
    /// $O(n * log(n))$ worst-case.
    ///
    /// When applicable, unstable sorting is preferred because it is
    /// generally faster than stable sorting and it doesn’t allocate
    /// auxiliary memory. See `sort_unstable`.
    pub fn sort(&mut self) {
        self.0.sort();
    }
}

impl PopulatedSlice<u8> {
    /// Checks if all bytes in this slice are within the ASCII range.
    pub const fn is_ascii(&self) -> bool {
        self.0.is_ascii()
    }
}

impl<T> Borrow<PopulatedSlice<T>> for PopulatedVec<T> {
    fn borrow(&self) -> &PopulatedSlice<T> {
        let slice: &[T] = self.0.deref();
        unsafe { &*(slice as *const [T] as *const PopulatedSlice<T>) }
    }
}

impl<T> BorrowMut<PopulatedSlice<T>> for PopulatedVec<T> {
    fn borrow_mut(&mut self) -> &mut PopulatedSlice<T> {
        let slice: &mut [T] = self.0.deref_mut();
        unsafe { &mut *(slice as *mut [T] as *mut PopulatedSlice<T>) }
    }
}

impl<T: Clone> From<&PopulatedSlice<T>> for PopulatedVec<T> {
    fn from(value: &PopulatedSlice<T>) -> Self {
        PopulatedVec(value.0.to_vec())
    }
}

impl<T: Clone> From<&mut PopulatedSlice<T>> for PopulatedVec<T> {
    fn from(value: &mut PopulatedSlice<T>) -> Self {
        PopulatedVec(value.0.to_vec())
    }
}

impl<T> Deref for PopulatedVec<T> {
    type Target = PopulatedSlice<T>;

    fn deref(&self) -> &Self::Target {
        let slice: &[T] = &self.0;
        unsafe { &*(slice as *const [T] as *const PopulatedSlice<T>) }
    }
}

impl<T> DerefMut for PopulatedVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let slice: &mut [T] = &mut self.0;
        unsafe { &mut *(slice as *mut [T] as *mut PopulatedSlice<T>) }
    }
}

impl<T> AsRef<PopulatedSlice<T>> for PopulatedVec<T> {
    fn as_ref(&self) -> &PopulatedSlice<T> {
        self
    }
}

impl<T> AsMut<PopulatedSlice<T>> for PopulatedVec<T> {
    fn as_mut(&mut self) -> &mut PopulatedSlice<T> {
        self
    }
}

impl<T: Clone> ToOwned for PopulatedSlice<T> {
    type Owned = PopulatedVec<T>;

    fn to_owned(&self) -> Self::Owned {
        self.to_vec()
    }
}

impl<'a, T: Clone> From<&'a PopulatedSlice<T>> for Cow<'a, PopulatedSlice<T>> {
    fn from(slice: &'a PopulatedSlice<T>) -> Self {
        Cow::Borrowed(slice)
    }
}

/// A populated double-ended queue implemented with a growable ring
/// buffer. This type is guaranteed to contain at least one element, so
/// the underlying `VecDeque` is guaranteed to be non-empty.
#[derive(PartialEq, PartialOrd, Eq, Ord, Clone, Debug)]
pub struct PopulatedVecDeque<T>(VecDeque<T>);

impl<T> Extend<T> for PopulatedVecDeque<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.0.extend(iter);
    }
}

impl<T> From<PopulatedVecDeque<T>> for VecDeque<T> {
    fn from(populated_vec_deque: PopulatedVecDeque<T>) -> VecDeque<T> {
        populated_vec_deque.0
    }
}

impl<T> From<PopulatedVec<T>> for PopulatedVecDeque<T> {
    fn from(populated_vec: PopulatedVec<T>) -> PopulatedVecDeque<T> {
        PopulatedVecDeque(populated_vec.0.into())
    }
}

impl<T> TryFrom<VecDeque<T>> for PopulatedVecDeque<T> {
    type Error = VecDeque<T>;

    fn try_from(vec_deque: VecDeque<T>) -> Result<PopulatedVecDeque<T>, Self::Error> {
        if vec_deque.is_empty() {
            Err(vec_deque)
        } else {
            Ok(PopulatedVecDeque(vec_deque))
        }
    }
}

impl<T> PopulatedVecDeque<T> {
    /// Creates a singleton populated deque i.e. a deque with a single element.
    pub fn new(value: T) -> PopulatedVecDeque<T> {
        PopulatedVecDeque(VecDeque::from(vec![value]))
    }

    /// Creates a populated deque with a single element and space for at
    /// least `capacity` elements.
    pub fn with_capacity(capacity: NonZeroUsize, value: T) -> PopulatedVecDeque<T> {
        let mut vec_deque = VecDeque::with_capacity(capacity.get());
        vec_deque.push_back(value);
        PopulatedVecDeque(vec_deque)
    }

    /// Creates a populated deque with the given element inserted at the specified index.
    ///
    /// ```
    /// use std::collections::VecDeque;
    /// use std::num::NonZeroUsize;
    /// use populated::PopulatedVecDeque;
    ///
    /// let vec_deque = vec![1].into();
    /// let vec_deque = PopulatedVecDeque::inserted(vec_deque, 0, 2);
    /// assert_eq!(vec_deque.get(0), Some(&2));
    /// assert_eq!(vec_deque.get(1), Some(&1));
    /// ```
    pub fn inserted(mut vec_deque: VecDeque<T>, index: usize, value: T) -> PopulatedVecDeque<T> {
        vec_deque.insert(index, value);
        PopulatedVecDeque(vec_deque)
    }

    /// Provides a reference to the element at the given index.
    ///
    /// Element at index 0 is the front of the queue.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.0.get(index)
    }

    /// Provides a mutable reference to the element at the given index.
    ///
    /// Element at index 0 is the front of the queue.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.0.get_mut(index)
    }

    /// Swaps elements at indices i and j.
    ///
    /// `i` and `j` may be equal.
    ///
    /// Element at index 0 is the front of the queue.
    pub fn swap(&mut self, i: usize, j: usize) {
        self.0.swap(i, j);
    }

    pub fn into_inner(self) -> VecDeque<T> {
        self.0
    }

    /// Constructs a `PopulatedVecDeque` from a `VecDeque` by pushing back a
    /// single element.
    ///
    /// ```
    /// use std::collections::VecDeque;
    /// use std::num::NonZeroUsize;
    /// use populated::PopulatedVecDeque;
    ///
    /// let vec_deque = VecDeque::from([1, 2, 3]);
    /// let populated_vec_deque = PopulatedVecDeque::pushed_back(vec_deque, 42);
    /// assert_eq!(populated_vec_deque.len(), NonZeroUsize::new(4).unwrap());
    /// assert_eq!(populated_vec_deque.back(), &42);
    /// ```
    pub fn pushed_back(mut vec_deque: VecDeque<T>, value: T) -> PopulatedVecDeque<T> {
        vec_deque.push_back(value);
        PopulatedVecDeque(vec_deque)
    }

    /// Appends an element to the back of the deque.
    pub fn push_back(&mut self, value: T) {
        self.0.push_back(value);
    }

    /// Constructs a `PopulatedVecDeque` from a `VecDeque` by pushing front a
    /// single element.
    ///
    /// ```
    /// use std::collections::VecDeque;
    /// use std::num::NonZeroUsize;
    /// use populated::PopulatedVecDeque;
    ///
    /// let vec_deque = VecDeque::from([1, 2, 3]);
    /// let populated_vec_deque = PopulatedVecDeque::pushed_front(42, vec_deque);
    /// assert_eq!(populated_vec_deque.len(), NonZeroUsize::new(4).unwrap());
    /// assert_eq!(populated_vec_deque.front(), &42);
    /// ```
    pub fn pushed_front(value: T, mut vec_deque: VecDeque<T>) -> PopulatedVecDeque<T> {
        vec_deque.push_front(value);
        PopulatedVecDeque(vec_deque)
    }

    /// Prepends an element to the deque.
    pub fn push_front(&mut self, value: T) {
        self.0.push_front(value);
    }

    /// Removes the last element from the deque and returns it, along
    /// with the remaining deque.
    pub fn pop_back(self) -> (VecDeque<T>, T) {
        let mut vec_deque = self.0;
        let last = vec_deque.pop_back().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedVecDeque
        (vec_deque, last)
    }

    /// Removes the first element and returns it, along with the remaining deque.
    pub fn pop_front(self) -> (T, VecDeque<T>) {
        let mut vec_deque = self.0;
        let first = vec_deque.pop_front().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedVecDeque
        (first, vec_deque)
    }

    /// Provides a reference to the front element.
    pub fn front(&self) -> &T {
        self.0.front().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedVecDeque
    }

    /// Provides a mutable reference to the front element.
    pub fn front_mut(&mut self) -> &mut T {
        self.0.front_mut().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedVecDeque
    }

    /// Provides a reference to the back element.
    pub fn back(&self) -> &T {
        self.0.back().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedVecDeque
    }

    /// Provides a mutable reference to the back element.
    pub fn back_mut(&mut self) -> &mut T {
        self.0.back_mut().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedVecDeque
    }

    /// Returns number of elements in the populated deque.
    pub fn len(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.0.len()).unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedVecDeque
    }

    /// Returns the number of elements the deque can hold without
    /// reallocating.
    pub fn capacity(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.0.capacity()).unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedVecDeque
    }

    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    pub fn reserve_exact(&mut self, additional: usize) {
        self.0.reserve_exact(additional);
    }

    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.0.try_reserve(additional)
    }

    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.0.try_reserve_exact(additional)
    }

    pub fn shrink_to_fit(&mut self) {
        self.0.shrink_to_fit();
    }

    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.0.shrink_to(min_capacity);
    }

    pub fn truncate(&mut self, len: NonZeroUsize) {
        self.0.truncate(len.get());
    }

    /// Truncates the deque to the given length.
    pub fn truncate_into(self, len: usize) -> VecDeque<T> {
        let mut vec = self.0;
        vec.truncate(len);
        vec
    }

    /// Returns an iterator that yields references to the values.
    pub fn range(&self, range: impl RangeBounds<usize>) -> std::collections::vec_deque::Iter<T> {
        self.0.range(range)
    }

    /// Returns an iterator that allows modifying each value.
    /// The iterator yields mutable references to the values.
    pub fn range_mut(
        &mut self,
        range: impl RangeBounds<usize>,
    ) -> std::collections::vec_deque::IterMut<T> {
        self.0.range_mut(range)
    }

    /// Clears the deque, removing all values.
    /// Returns the cleared deque that is not populated.
    pub fn clear(self) -> VecDeque<T> {
        let mut vec_deque = self.0;
        vec_deque.clear();
        vec_deque
    }

    // pub fn swap_remove_front(self, index: usize) -> (Option<T>, VecDeque<T>) {
    //     let mut vec_deque = self.0;
    //     let removed = vec_deque.swap_remove_front(index);
    //     (removed, vec_deque)
    // }

    // pub fn swap_remove_back(self, index: usize) -> (Option<T>, VecDeque<T>) {
    //     let mut vec_deque = self.0;
    //     let removed = vec_deque.swap_remove_back(index);
    //     (removed, vec_deque)
    // }

    /// Inserts an element at `index` within the deque, shifting all elements
    /// with indices greater than or equal to `index` towards the back.
    ///
    /// Element at index 0 is the front of the queue.
    pub fn insert(&mut self, index: usize, element: T) {
        self.0.insert(index, element);
    }

    /// Removes and returns the element at `index` from the deque. Whichever
    /// end is closer to the removal point will be moved to make room, and
    /// all the affected elements will be moved to new positions. Returns
    /// `None` if `index` is out of bounds.
    ///
    /// Element at index 0 is the front of the queue.
    pub fn remove(self, index: usize) -> (Option<T>, VecDeque<T>) {
        let mut vec_deque = self.0;
        let removed = vec_deque.remove(index);
        (removed, vec_deque)
    }

    /// Splits the deque into two at the given index.
    ///
    /// Returns a newly allocated VecDeque. self contains elements
    /// `[0, at)`, and the returned deque contains elements `[at, len)`.
    ///
    /// `at` must be non-zero to ensure that the deque being mutated stays populated.
    ///
    /// Note that the capacity of self does not change.
    ///
    /// Element at index 0 is the front of the queue.
    pub fn split_off(&mut self, at: NonZeroUsize) -> VecDeque<T> {
        self.0.split_off(at.get())
    }

    pub fn split_into(self, at: usize) -> (VecDeque<T>, VecDeque<T>) {
        let mut vec_deque = self.0;
        let other = vec_deque.split_off(at);
        (vec_deque, other)
    }

    /// Moves all the elements of other into self, leaving other empty.
    pub fn append(&mut self, other: &mut VecDeque<T>) {
        self.0.append(other);
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements e for which `f(&e)` returns
    /// false. This method operates in place on the underlying `VecDeque`, visiting each element
    /// exactly once in the original order, and preserves the order of
    /// the retained elements.
    ///
    /// Since there is no guarantee that all elements will not be removed,
    /// the method returns the underlying VecDeque.
    pub fn retain(self, f: impl FnMut(&T) -> bool) -> VecDeque<T> {
        let mut vec_deque = self.0;
        vec_deque.retain(f);
        vec_deque
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements e for which `f(&e)` returns
    /// false. This method operates in place on the underlying `VecDeque`, visiting each element
    /// exactly once in the original order, and preserves the order of
    /// the retained elements.
    ///
    /// Since there is no guarantee that all elements will not be removed,
    /// the method returns the underlying VecDeque.
    pub fn retain_mut(self, f: impl FnMut(&mut T) -> bool) -> VecDeque<T> {
        let mut vec_deque = self.0;
        vec_deque.retain_mut(f);
        vec_deque
    }

    pub fn resize_with(&mut self, new_len: NonZeroUsize, f: impl FnMut() -> T) {
        self.0.resize_with(new_len.get(), f);
    }

    pub fn resize_with_into(self, new_len: usize, f: impl FnMut() -> T) -> VecDeque<T> {
        let mut vec_deque = self.0;
        vec_deque.resize_with(new_len, f);
        vec_deque
    }

    /// Rotates the double-ended queue `n` places to the left.
    ///
    /// Equivalently,
    ///
    /// - Rotates item `n` into the first position.
    /// - Pops the first `n` items and pushes them to the end.
    /// - Rotates `len() - n` places to the right.
    pub fn rotate_left(&mut self, n: usize) {
        self.0.rotate_left(n);
    }

    /// Rotates the double-ended queue `n` places to the right.
    ///
    /// Equivalently,
    ///
    /// - Rotates the first item into position `n`.
    /// - Pops the last `n` items and pushes them to the front.
    /// - Rotates `len() - n` places to the left.
    pub fn rotate_right(&mut self, n: usize) {
        self.0.rotate_right(n);
    }

    /// Binary searches this `PopulatedVecDeque` with a comparator
    /// function.
    ///
    /// The comparator function should return an order code that
    /// indicates whether its argument is `Less`, `Equal` or `Greater` the
    /// desired target. If the `PopulatedVecDeque` is not sorted or if
    /// the comparator function does not implement an order consistent
    /// with the sort order of the underlying `VecDeque`, the returned
    /// result is unspecified and meaningless.
    ///
    /// If the value is found then `Result::Ok` is returned, containing
    /// the index of the matching element. If there are multiple matches,
    /// then any one of the matches could be returned. If the value is
    /// not found then `Result::Err` is returned, containing the index
    /// where a matching element could be inserted while maintaining
    /// sorted order.
    ///
    /// See also `binary_search`, `binary_search_by_key`, and
    /// `partition_point`.
    pub fn binary_search_by(&self, f: impl FnMut(&T) -> Ordering) -> Result<usize, usize> {
        self.0.binary_search_by(f)
    }

    /// Binary searches this `PopulatedVecDeque` with a key extraction
    /// function.
    ///
    /// Assumes that the deque is sorted by the key, for instance with
    /// `make_contiguous().sort_by_key()` using the same key extraction
    /// function. If the deque is not sorted by the key, the returned result
    /// is unspecified and meaningless.
    ///
    /// If the value is found then `Result::Ok` is returned, containing the
    /// index of the matching element. If there are multiple matches, then
    /// any one of the matches could be returned. If the value is not found
    /// then `Result::Err` is returned, containing the index where a matching
    /// element could be inserted while maintaining sorted order.
    ///
    /// See also `binary_search`, `binary_search_by`, and `partition_point`.
    pub fn binary_search_by_key<K: Ord>(
        &self,
        key: &K,
        f: impl FnMut(&T) -> K,
    ) -> Result<usize, usize> {
        self.0.binary_search_by_key(key, f)
    }

    pub fn partition_point(&self, predicate: impl FnMut(&T) -> bool) -> usize {
        self.0.partition_point(predicate)
    }
}

impl<T: PartialEq> PopulatedVecDeque<T> {
    /// Returns true if the deque contains an element equal to the given
    /// value.
    ///
    /// This operation is O(n).
    ///
    /// Note that if you have a sorted `PopulatedVecDeque`, `binary_search` may be
    /// faster.
    pub fn contains(&self, x: &T) -> bool {
        self.0.contains(x)
    }
}

impl<T: Ord> PopulatedVecDeque<T> {
    /// Binary searches this `PopulatedVecDeque` for a given element. If the `PopulatedVecDeque`
    /// is not sorted, the returned result is unspecified and meaningless.
    ///
    /// If the value is found then `Result::Ok` is returned, containing the
    /// index of the matching element. If there are multiple matches, then
    /// any one of the matches could be returned. If the value is not found
    /// then `Result::Err` is returned, containing the index where a matching
    /// element could be inserted while maintaining sorted order.
    ///
    /// See also `binary_search_by`, `binary_search_by_key`, and
    /// `partition_point`.
    pub fn binary_search(&self, x: &T) -> Result<usize, usize> {
        self.0.binary_search(x)
    }
}

impl<T: Clone> PopulatedVecDeque<T> {
    pub fn resize(&mut self, new_len: NonZeroUsize, value: T) {
        self.0.resize(new_len.get(), value);
    }

    pub fn resize_into(self, new_len: usize, value: T) -> VecDeque<T> {
        let mut vec_deque = self.0;
        vec_deque.resize(new_len, value);
        vec_deque
    }
}

impl<T: PartialEq> PartialEq<VecDeque<T>> for PopulatedVecDeque<T> {
    fn eq(&self, other: &VecDeque<T>) -> bool {
        self.0.eq(other)
    }
}

impl<T: PartialEq> PartialEq<PopulatedVecDeque<T>> for VecDeque<T> {
    fn eq(&self, other: &PopulatedVecDeque<T>) -> bool {
        self.eq(&other.0)
    }
}

impl<T> Index<usize> for PopulatedVecDeque<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T> IndexMut<usize> for PopulatedVecDeque<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T> Index<NonZeroUsize> for PopulatedVecDeque<T> {
    type Output = T;

    fn index(&self, index: NonZeroUsize) -> &Self::Output {
        &self.0[index.get() - 1]
    }
}

impl<T> IndexMut<NonZeroUsize> for PopulatedVecDeque<T> {
    fn index_mut(&mut self, index: NonZeroUsize) -> &mut Self::Output {
        &mut self.0[index.get() - 1]
    }
}

impl<T> PopulatedVecDeque<T> {
    pub fn iter(&self) -> vec_deque::PopulatedIter<T> {
        self.into_populated_iter()
    }

    pub fn iter_mut(&mut self) -> vec_deque::PopulatedIterMut<T> {
        self.into_populated_iter()
    }
}

/// An iterator that guaratees that there is at least one element.
pub trait PopulatedIterator: IntoIterator // to enable using in for loops
{
    /// Advances the iterator and returns the first value and a new iterator
    /// to the remaining values.
    fn next(self) -> (Self::Item, Self::IntoIter);

    /// Collects the iterator into a collection. Supports constructing
    /// collections that require a populated iterator.
    fn collect<C: FromPopulatedIterator<Self::Item>>(self) -> C
    where
        Self: Sized,
    {
        C::from_populated_iter(self)
    }

    /// Takes a closure and creates an iterator which calls that closure on
    /// each element. Preserves the populated property.
    fn map<B, F: FnMut(Self::Item) -> B>(self, f: F) -> iter::Map<Self, F>
    where
        Self: Sized,
    {
        iter::Map { iter: self, f }
    }

    /// Creates an iterator giving the current element count along with the
    /// element. Preserves the populated property.
    fn enumerate(self) -> iter::Enumerate<Self>
    where
        Self: Sized,
    {
        iter::Enumerate { iter: self }
    }

    /// Zips this iterator with another iterator to yield a new iterator of
    /// pairs. Preserves the populated property.
    fn zip<J>(self, other: J) -> iter::Zip<Self, J>
    where
        Self: Sized,
    {
        iter::Zip { iter: self, other }
    }

    /// Flattens a populated iterator of populated iteratorables into a single iterator. Preserves
    /// the populated property.
    fn flatten(self) -> iter::Flatten<Self>
    where
        Self: Sized,
        Self::Item: IntoPopulatedIterator,
    {
        iter::Flatten { iter: self }
    }

    /// Creates an iterator taking at most `n` elements from this iterator, where `n` is non-zero.
    /// Preserves the populated property.
    fn take(self, n: NonZeroUsize) -> iter::Take<Self>
    where
        Self: Sized,
    {
        iter::Take::new(self, n)
    }

    /// Creates an iterator that works like `map`, but flattens nested populated structure. Preserves the
    /// populated property.
    fn flat_map<B: IntoPopulatedIterator, F: FnMut(Self::Item) -> B>(
        self,
        f: F,
    ) -> iter::Flatten<iter::Map<Self, F>>
    where
        Self: Sized,
    {
        self.map(f).flatten()
    }

    /// Returns the maximum element of the iterator. Note that unlike the standard library, this
    /// method directly returns the maximum element instead of an `Option`. This is because this
    /// method is only available on populated iterators.
    fn max(self) -> Self::Item
    where
        Self::Item: Ord,
        Self: Sized,
    {
        self.into_iter().max().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedIterator
    }

    /// Returns the minimum element of the iterator.
    fn min(self) -> Self::Item
    where
        Self::Item: Ord,
        Self: Sized,
    {
        self.into_iter().min().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedIterator
    }

    /// Takes self populator and another iteratorable and creates a new iterator over both in sequence. Preserves the populated property.
    fn chain<I: IntoIterator<Item = Self::Item>>(self, other: I) -> iter::Chain<Self, I::IntoIter>
    where
        Self: Sized,
    {
        iter::Chain {
            prefix: self,
            iter: other.into_iter(),
        }
    }

    /// Reduces the iterator to a single value using a closure. Note that unlike the standard library,
    /// this method directly returns the reduced value instead of an `Option`. This is because this
    /// method is only available on populated iterators.
    fn reduce(self, f: impl FnMut(Self::Item, Self::Item) -> Self::Item) -> Self::Item
    where
        Self: Sized,
    {
        let (first, iter) = self.next();
        iter.fold(first, f)
    }

    /// Returns the element that gives the maximum value from the specified function. Note that unlike the standard library,
    /// this method directly returns the maximum element instead of an `Option`. This is because this
    /// method is only available on populated iterators.
    fn max_by_key<K: Ord>(self, f: impl FnMut(&Self::Item) -> K) -> Self::Item
    where
        Self: Sized,
    {
        self.into_iter().max_by_key(f).unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedIterator
    }

    /// Returns the element that gives the maximum value with respect to the specified comparison function. Note that unlike the standard library,
    /// this method directly returns the maximum element instead of an `Option`. This is because this
    /// method is only available on populated iterators.
    fn max_by(self, compare: impl FnMut(&Self::Item, &Self::Item) -> Ordering) -> Self::Item
    where
        Self: Sized,
    {
        self.into_iter().max_by(compare).unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedIterator
    }

    /// Returns the element that gives the minimum value from the specified function. Note that unlike the standard library,
    /// this method directly returns the minimum element instead of an `Option`. This is because this
    /// method is only available on populated iterators.
    fn min_by_key<K: Ord>(self, f: impl FnMut(&Self::Item) -> K) -> Self::Item
    where
        Self: Sized,
    {
        self.into_iter().min_by_key(f).unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedIterator
    }

    /// Returns the element that gives the minimum value with respect to the specified comparison function. Note that unlike the standard library,
    /// this method directly returns the minimum element instead of an `Option`. This is because this
    /// method is only available on populated iterators.
    fn min_by(self, compare: impl FnMut(&Self::Item, &Self::Item) -> Ordering) -> Self::Item
    where
        Self: Sized,
    {
        self.into_iter().min_by(compare).unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedIterator
    }

    fn eq<I: IntoIterator>(self, other: I) -> bool
    where
        Self::Item: PartialEq<I::Item>,
        Self: Sized,
    {
        self.into_iter().eq(other)
    }
}

pub mod iter {

    use std::num::NonZeroUsize;

    use crate::{IntoPopulatedIterator, PopulatedIterator};

    pub struct Map<I, F> {
        pub(crate) iter: I,
        pub(crate) f: F,
    }

    impl<I: PopulatedIterator, B, F: FnMut(I::Item) -> B> IntoIterator for Map<I, F> {
        type Item = B;
        type IntoIter = std::iter::Map<I::IntoIter, F>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter.into_iter().map(self.f)
        }
    }

    impl<I: PopulatedIterator, F: FnMut(I::Item) -> O, O> PopulatedIterator for Map<I, F> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let (item, iter) = self.iter.next();
            ((self.f)(item), iter.map(self.f))
        }
    }

    pub struct Enumerate<I> {
        pub(crate) iter: I,
    }

    impl<I: PopulatedIterator> IntoIterator for Enumerate<I> {
        type Item = (usize, I::Item);
        type IntoIter = std::iter::Enumerate<I::IntoIter>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter.into_iter().enumerate()
        }
    }

    impl<I: PopulatedIterator> PopulatedIterator for Enumerate<I> {
        fn next(self) -> ((usize, I::Item), Self::IntoIter) {
            let mut iter = self.into_iter();
            let (index, item) = iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedVec

            ((index, item), iter)
        }
    }

    pub struct Zip<I, J> {
        pub(crate) iter: I,
        pub(crate) other: J,
    }

    impl<I: PopulatedIterator, J: PopulatedIterator> IntoIterator for Zip<I, J> {
        type Item = (I::Item, J::Item);
        type IntoIter = std::iter::Zip<I::IntoIter, J::IntoIter>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter.into_iter().zip(self.other)
        }
    }

    impl<I: PopulatedIterator, J: PopulatedIterator> PopulatedIterator for Zip<I, J> {
        fn next(self) -> (Self::Item, Self::IntoIter) {
            let (x, iter) = self.iter.next();
            let (y, other) = self.other.next();
            ((x, y), iter.zip(other))
        }
    }

    pub struct Flatten<I> {
        pub(crate) iter: I,
    }

    impl<I: PopulatedIterator> IntoIterator for Flatten<I>
    where
        I::Item: IntoPopulatedIterator,
    {
        type Item = <I::Item as IntoIterator>::Item;
        type IntoIter = std::iter::Flatten<I::IntoIter>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter.into_iter().flatten()
        }
    }

    impl<I: PopulatedIterator> PopulatedIterator for Flatten<I>
    where
        I::Item: IntoPopulatedIterator,
    {
        fn next(self) -> (Self::Item, Self::IntoIter) {
            let mut iter = self.into_iter();
            let item = iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedIterator contain impl IntoPopulatedIterator
            (item, iter)
        }
    }

    pub struct Take<I: PopulatedIterator> {
        iter: std::iter::Take<I::IntoIter>,
    }
    impl<I: PopulatedIterator> Take<I> {
        pub(crate) fn new(iter: I, n: NonZeroUsize) -> Self {
            Take {
                iter: iter.into_iter().take(n.get()),
            }
        }
    }

    impl<I: PopulatedIterator> IntoIterator for Take<I> {
        type Item = I::Item;
        type IntoIter = std::iter::Take<I::IntoIter>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<I: PopulatedIterator> PopulatedIterator for Take<I> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let item = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedIterator
            (item, self.iter)
        }
    }

    // TODO: Cycle

    pub struct Chain<P, I> {
        pub(crate) prefix: P,
        pub(crate) iter: I,
    }

    impl<P: PopulatedIterator, I: Iterator<Item = P::Item>> IntoIterator for Chain<P, I> {
        type Item = P::Item;
        type IntoIter = std::iter::Chain<P::IntoIter, I>;

        fn into_iter(self) -> Self::IntoIter {
            self.prefix.into_iter().chain(self.iter)
        }
    }

    impl<P: PopulatedIterator, I: Iterator<Item = P::Item>> PopulatedIterator for Chain<P, I> {
        fn next(self) -> (Self::Item, Self::IntoIter) {
            let (item, prefix) = self.prefix.next();
            let iter = self.iter;
            (item, prefix.chain(iter))
        }
    }
}

/// Conversion into a `PopulatedIterator`.
///
/// This trait is used to convert a type into a `PopulatedIterator`.
///
/// # Examples
///
/// ```
/// use populated::{PopulatedVecDeque, IntoPopulatedIterator, PopulatedIterator};
///
/// let mut vec_deque = PopulatedVecDeque::new(1);
/// vec_deque.push_back(2);
/// vec_deque.push_back(3);
/// let populated_iter = vec_deque.into_populated_iter();
///
/// let (first, iter) = populated_iter.next();
/// assert_eq!(first, 1);
/// assert_eq!(iter.collect::<Vec<_>>(), [2, 3]);
/// ```
pub trait IntoPopulatedIterator: IntoIterator {
    type PopulatedIntoIter: PopulatedIterator<
        Item = <Self as IntoIterator>::Item,
        IntoIter = <Self as IntoIterator>::IntoIter,
    >;

    /// Converts the type into a `PopulatedIterator`.
    fn into_populated_iter(self) -> Self::PopulatedIntoIter;
}

impl<I: PopulatedIterator> IntoPopulatedIterator for I {
    type PopulatedIntoIter = Self;

    fn into_populated_iter(self) -> Self::PopulatedIntoIter {
        self
    }
}

pub trait PotentiallyUnpopulated {}

impl<T> PotentiallyUnpopulated for std::collections::VecDeque<T> {}
impl<T> PotentiallyUnpopulated for Vec<T> {}
impl<T> PotentiallyUnpopulated for std::collections::LinkedList<T> {}
impl<T> PotentiallyUnpopulated for std::collections::BinaryHeap<T> {}
impl<T> PotentiallyUnpopulated for std::collections::HashSet<T> {}
impl<T> PotentiallyUnpopulated for std::collections::BTreeSet<T> {}
impl<K, V> PotentiallyUnpopulated for std::collections::HashMap<K, V> {}
impl<K, V> PotentiallyUnpopulated for std::collections::BTreeMap<K, V> {}

/// Conversion from a `PopulatedIterator`.
///
/// This trait is used to convert a `PopulatedIterator` into `Self`.
pub trait FromPopulatedIterator<T>: Sized {
    /// Converts a `PopulatedIterator` into `Self`.
    fn from_populated_iter(iter: impl IntoPopulatedIterator<Item = T>) -> Self;
}

impl<E, C: FromIterator<E> + PotentiallyUnpopulated> FromPopulatedIterator<E> for C {
    fn from_populated_iter(iter: impl IntoPopulatedIterator<Item = E>) -> C {
        iter.into_iter().collect()
    }
}

pub mod vec {
    use super::*;
    pub struct PopulatedIntoIter<T> {
        iter: std::vec::IntoIter<T>,
    }
    impl<T> IntoIterator for PopulatedIntoIter<T> {
        type Item = T;
        type IntoIter = std::vec::IntoIter<T>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<T> PopulatedIterator for PopulatedIntoIter<T> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedVec
            (first, self.iter)
        }
    }

    impl<T> IntoIterator for PopulatedVec<T> {
        type Item = T;
        type IntoIter = std::vec::IntoIter<T>;

        fn into_iter(self) -> Self::IntoIter {
            self.0.into_iter()
        }
    }

    impl<T> IntoPopulatedIterator for PopulatedVec<T> {
        type PopulatedIntoIter = vec::PopulatedIntoIter<T>;

        fn into_populated_iter(self) -> vec::PopulatedIntoIter<T> {
            vec::PopulatedIntoIter {
                iter: self.into_iter(),
            }
        }
    }
}
pub mod slice {
    use crate::{IntoPopulatedIterator, PopulatedIterator, PopulatedSlice, PopulatedVec};

    pub struct PopulatedIter<'a, T> {
        iter: std::slice::Iter<'a, T>,
    }

    impl<'a, T> IntoIterator for PopulatedIter<'a, T> {
        type Item = &'a T;
        type IntoIter = std::slice::Iter<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<'a, T> PopulatedIterator for PopulatedIter<'a, T> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedSlice
            (first, self.iter)
        }
    }

    impl<'a, T> IntoIterator for &'a PopulatedSlice<T> {
        type Item = &'a T;
        type IntoIter = std::slice::Iter<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            self.0.iter()
        }
    }

    impl<'a, T> IntoPopulatedIterator for &'a PopulatedSlice<T> {
        type PopulatedIntoIter = PopulatedIter<'a, T>;

        fn into_populated_iter(self) -> PopulatedIter<'a, T> {
            PopulatedIter {
                iter: self.into_iter(),
            }
        }
    }

    impl<'a, T> IntoIterator for &'a PopulatedVec<T> {
        type Item = &'a T;
        type IntoIter = std::slice::Iter<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            self.0.iter()
        }
    }

    impl<'a, T> IntoPopulatedIterator for &'a PopulatedVec<T> {
        type PopulatedIntoIter = PopulatedIter<'a, T>;

        fn into_populated_iter(self) -> PopulatedIter<'a, T> {
            PopulatedIter {
                iter: self.into_iter(),
            }
        }
    }

    pub struct PopulatedIterMut<'a, T> {
        iter: std::slice::IterMut<'a, T>,
    }

    impl<'a, T> IntoIterator for PopulatedIterMut<'a, T> {
        type Item = &'a mut T;
        type IntoIter = std::slice::IterMut<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<'a, T> PopulatedIterator for PopulatedIterMut<'a, T> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedSlice
            (first, self.iter)
        }
    }

    impl<'a, T> IntoIterator for &'a mut PopulatedSlice<T> {
        type Item = &'a mut T;
        type IntoIter = std::slice::IterMut<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            self.0.iter_mut()
        }
    }

    impl<'a, T> IntoPopulatedIterator for &'a mut PopulatedSlice<T> {
        type PopulatedIntoIter = PopulatedIterMut<'a, T>;

        fn into_populated_iter(self) -> PopulatedIterMut<'a, T> {
            PopulatedIterMut {
                iter: self.into_iter(),
            }
        }
    }

    impl<'a, T> IntoIterator for &'a mut PopulatedVec<T> {
        type Item = &'a mut T;
        type IntoIter = std::slice::IterMut<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            self.0.iter_mut()
        }
    }

    impl<'a, T> IntoPopulatedIterator for &'a mut PopulatedVec<T> {
        type PopulatedIntoIter = PopulatedIterMut<'a, T>;

        fn into_populated_iter(self) -> PopulatedIterMut<'a, T> {
            PopulatedIterMut {
                iter: self.into_iter(),
            }
        }
    }
}

pub mod vec_deque {
    use super::*;
    pub struct PopulatedIntoIter<T> {
        iter: std::collections::vec_deque::IntoIter<T>,
    }

    impl<Y> IntoIterator for PopulatedIntoIter<Y> {
        type Item = Y;
        type IntoIter = std::collections::vec_deque::IntoIter<Y>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<T> PopulatedIterator for PopulatedIntoIter<T> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedVecDeque
            (first, self.iter)
        }
    }

    impl<T> IntoIterator for PopulatedVecDeque<T> {
        type Item = T;
        type IntoIter = std::collections::vec_deque::IntoIter<T>;

        fn into_iter(self) -> Self::IntoIter {
            self.0.into_iter()
        }
    }

    impl<T> IntoPopulatedIterator for PopulatedVecDeque<T> {
        type PopulatedIntoIter = vec_deque::PopulatedIntoIter<T>;

        fn into_populated_iter(self) -> vec_deque::PopulatedIntoIter<T> {
            vec_deque::PopulatedIntoIter {
                iter: self.into_iter(),
            }
        }
    }

    pub struct PopulatedIter<'a, T> {
        iter: std::collections::vec_deque::Iter<'a, T>,
    }

    impl<'a, T> IntoIterator for PopulatedIter<'a, T> {
        type Item = &'a T;
        type IntoIter = std::collections::vec_deque::Iter<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<'a, T> PopulatedIterator for PopulatedIter<'a, T> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedVecDeque
            (first, self.iter)
        }
    }

    impl<'a, T> IntoIterator for &'a PopulatedVecDeque<T> {
        type Item = &'a T;
        type IntoIter = std::collections::vec_deque::Iter<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            self.0.iter()
        }
    }

    impl<'a, T> IntoPopulatedIterator for &'a PopulatedVecDeque<T> {
        type PopulatedIntoIter = vec_deque::PopulatedIter<'a, T>;

        fn into_populated_iter(self) -> vec_deque::PopulatedIter<'a, T> {
            vec_deque::PopulatedIter {
                iter: self.into_iter(),
            }
        }
    }

    pub struct PopulatedIterMut<'a, T> {
        iter: std::collections::vec_deque::IterMut<'a, T>,
    }

    impl<'a, T> IntoIterator for PopulatedIterMut<'a, T> {
        type Item = &'a mut T;
        type IntoIter = std::collections::vec_deque::IterMut<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<'a, T> PopulatedIterator for PopulatedIterMut<'a, T> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedVecDeque
            (first, self.iter)
        }
    }

    impl<'a, T> IntoIterator for &'a mut PopulatedVecDeque<T> {
        type Item = &'a mut T;
        type IntoIter = std::collections::vec_deque::IterMut<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            self.0.iter_mut()
        }
    }

    impl<'a, T> IntoPopulatedIterator for &'a mut PopulatedVecDeque<T> {
        type PopulatedIntoIter = vec_deque::PopulatedIterMut<'a, T>;

        fn into_populated_iter(self) -> vec_deque::PopulatedIterMut<'a, T> {
            vec_deque::PopulatedIterMut {
                iter: self.into_iter(),
            }
        }
    }
}

/// PopulatedHashMap is a wrapper around `HashMap` that guarantees that the
/// map is non-empty. This is useful when you want to ensure that a map is
/// populated before using it.
#[derive(Clone, Debug)]
pub struct PopulatedHashMap<K, V, S = RandomState>(HashMap<K, V, S>);

impl<K, V, S> From<PopulatedHashMap<K, V, S>> for HashMap<K, V, S> {
    fn from(populated_hash_map: PopulatedHashMap<K, V, S>) -> HashMap<K, V, S> {
        populated_hash_map.0
    }
}

impl<K, V, S> TryFrom<HashMap<K, V, S>> for PopulatedHashMap<K, V, S> {
    type Error = HashMap<K, V, S>;

    fn try_from(hash_map: HashMap<K, V, S>) -> Result<PopulatedHashMap<K, V, S>, Self::Error> {
        if hash_map.is_empty() {
            Err(hash_map)
        } else {
            Ok(PopulatedHashMap(hash_map))
        }
    }
}

impl<K: Eq + Hash, V> PopulatedHashMap<K, V, RandomState> {
    /// Creates a singleton populated hash map i.e. a hash map with a single key-value pair.
    pub fn new(key: K, value: V) -> PopulatedHashMap<K, V, RandomState> {
        PopulatedHashMap(HashMap::from([(key, value)]))
    }

    /// Creates an empty `PopulatedHashMap` with the specified capacity
    /// and containing the given key-value-pair.
    ///
    /// Note the capacity must be non-zero and a key value pair must be
    /// supplied because this is a populated hash map i.e. non-empty
    /// hash map.
    pub fn with_capacity(
        capacity: NonZeroUsize,
        key: K,
        value: V,
    ) -> PopulatedHashMap<K, V, RandomState> {
        let mut hash_map = HashMap::with_capacity_and_hasher(capacity.get(), RandomState::new());
        hash_map.insert(key, value);
        PopulatedHashMap(hash_map)
    }
}

impl<K: Eq + Hash, V, S: BuildHasher> PopulatedHashMap<K, V, S> {
    pub fn with_hasher(hash_builder: S, key: K, value: V) -> PopulatedHashMap<K, V, S> {
        let mut hash_map = HashMap::with_hasher(hash_builder);
        hash_map.insert(key, value);
        PopulatedHashMap(hash_map)
    }

    pub fn with_capacity_and_hasher(
        capacity: NonZeroUsize,
        hash_builder: S,
        key: K,
        value: V,
    ) -> PopulatedHashMap<K, V, S> {
        let mut hash_map = HashMap::with_capacity_and_hasher(capacity.get(), hash_builder);
        hash_map.insert(key, value);
        PopulatedHashMap(hash_map)
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map’s key type, but `Hash`
    /// and `Eq` on the borrowed form must match those for the key type.
    pub fn get<Q: Hash + Eq + ?Sized>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        self.0.get(k)
    }

    /// Returns the key-value pair corresponding to the supplied key.
    ///
    /// The supplied key may be any borrowed form of the map’s key type,
    /// but `Hash` and `Eq` on the borrowed form must match those for the key
    /// type.
    pub fn get_key_value<Q: Hash + Eq + ?Sized>(&self, k: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
    {
        self.0.get_key_value(k)
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map’s key type, but `Hash`
    /// and `Eq` on the borrowed form must match those for the key type.
    pub fn contains_key<Q: Hash + Eq + ?Sized>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
    {
        self.0.contains_key(k)
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map’s key type, but `Hash`
    /// and `Eq` on the borrowed form must match those for the key type.
    pub fn get_mut<Q: Hash + Eq + ?Sized>(&mut self, k: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
    {
        self.0.get_mut(k)
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, None is returned.
    ///
    /// If the map did have this key present, the value is updated, and
    /// the old value is returned. The key is not updated, though; this
    /// matters for types that can be == without being identical. See the
    /// module-level documentation for more.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.0.insert(key, value)
    }

    /// Inserts a key-value pair into the map return the map as a `PopulatedHashMap` since inserting guarantees a `len()` > 0.
    ///
    /// If the map did not have this key present, None is returned.
    ///
    /// If the map did have this key present, the value is updated, and
    /// the old value is returned. The key is not updated, though; this
    /// matters for types that can be == without being identical. See the
    /// module-level documentation for more.
    ///
    /// This method is useful when you want to ensure that the map is populated after inserting a key-value pair.
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use populated::PopulatedHashMap;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut hash_map = HashMap::from([(42, "the answer")]);
    /// let (old, populated_hash_map) = PopulatedHashMap::inserted(hash_map, 42, "the updated answer");
    /// assert_eq!(populated_hash_map.len(), NonZeroUsize::new(1).unwrap());
    /// assert_eq!(old, Some("the answer"));
    /// ```
    pub fn inserted(
        hash_map: HashMap<K, V, S>,
        key: K,
        value: V,
    ) -> (Option<V>, PopulatedHashMap<K, V, S>) {
        let mut hash_map = hash_map;
        let old = hash_map.insert(key, value);
        (old, PopulatedHashMap(hash_map))
    }

    /// Removes a key from the map, returning the value at the key if the key was previously in the map.
    ///
    /// The key may be any borrowed form of the map’s key type, but `Hash` and
    /// `Eq` on the borrowed form must match those for the key type.
    pub fn remove<Q: Hash + Eq + ?Sized>(&mut self, k: &Q) -> Option<V>
    where
        K: Borrow<Q>,
    {
        self.0.remove(k)
    }

    /// Removes a key from the map, returning the stored key and value if
    /// the key was previously in the map.
    ///
    /// The key may be any borrowed form of the map’s key type, but `Hash`
    /// and Eq on the borrowed form must match those for the key type.
    pub fn remove_entry<Q: Hash + Eq + ?Sized>(&mut self, k: &Q) -> Option<(K, V)>
    where
        K: Borrow<Q>,
    {
        self.0.remove_entry(k)
    }
}

impl<K, V, S> PopulatedHashMap<K, V, S> {
    /// Returns the number of elements the map can hold without reallocating.
    ///
    /// This number is a lower bound; the `PopulatedHashMap<K, V>` might
    /// be able to hold more, but is guaranteed to be able to hold at
    /// least this many.
    pub fn capacity(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.0.capacity()).unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedHashMap
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.0.len()).unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedHashMap
    }

    pub fn retain(self, predicate: impl FnMut(&K, &mut V) -> bool) -> HashMap<K, V, S> {
        let mut hash_map = self.0;
        hash_map.retain(predicate);
        hash_map
    }

    pub fn clear(self) -> HashMap<K, V, S> {
        let mut hash_map = self.0;
        hash_map.clear();
        hash_map
    }

    pub fn hasher(&self) -> &S {
        self.0.hasher()
    }
}

impl<Q: Eq + Hash + ?Sized, K: Eq + Hash + Borrow<Q>, V, S: BuildHasher> Index<&Q>
    for PopulatedHashMap<K, V, S>
{
    type Output = V;

    fn index(&self, index: &Q) -> &Self::Output {
        &self.0[index]
    }
}

impl<K, V, S> PopulatedHashMap<K, V, S> {
    pub fn iter(&self) -> hash_map::PopulatedIter<K, V> {
        self.into_populated_iter()
    }

    pub fn iter_mut(&mut self) -> hash_map::PopulatedIterMut<K, V> {
        self.into_populated_iter()
    }

    pub fn keys(&self) -> hash_map::PopulatedKeys<K, V> {
        hash_map::PopulatedKeys {
            keys: self.0.keys(),
        }
    }

    pub fn values(&self) -> hash_map::PopulatedValues<K, V> {
        hash_map::PopulatedValues {
            values: self.0.values(),
        }
    }

    pub fn values_mut(&mut self) -> hash_map::PopulatedValuesMut<K, V> {
        hash_map::PopulatedValuesMut {
            values: self.0.values_mut(),
        }
    }
}
impl<K, V, S> IntoIterator for PopulatedHashMap<K, V, S> {
    type Item = (K, V);
    type IntoIter = std::collections::hash_map::IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, K, V, S> IntoIterator for &'a PopulatedHashMap<K, V, S> {
    type Item = (&'a K, &'a V);
    type IntoIter = std::collections::hash_map::Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, K, V, S> IntoIterator for &'a mut PopulatedHashMap<K, V, S> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = std::collections::hash_map::IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

pub mod hash_map {
    use super::*;
    pub struct PopulatedIntoIter<K, V> {
        iter: std::collections::hash_map::IntoIter<K, V>,
    }

    impl<K, V> IntoIterator for PopulatedIntoIter<K, V> {
        type Item = (K, V);
        type IntoIter = std::collections::hash_map::IntoIter<K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<K, V> PopulatedIterator for PopulatedIntoIter<K, V> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedHashMap
            (first, self.iter)
        }
    }

    impl<K, V, S> IntoPopulatedIterator for PopulatedHashMap<K, V, S> {
        type PopulatedIntoIter = hash_map::PopulatedIntoIter<K, V>;

        fn into_populated_iter(self) -> hash_map::PopulatedIntoIter<K, V> {
            hash_map::PopulatedIntoIter {
                iter: self.into_iter(),
            }
        }
    }

    pub struct PopulatedIter<'a, K, V> {
        iter: std::collections::hash_map::Iter<'a, K, V>,
    }

    impl<'a, K, V> IntoIterator for PopulatedIter<'a, K, V> {
        type Item = (&'a K, &'a V);
        type IntoIter = std::collections::hash_map::Iter<'a, K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<'a, K, V> PopulatedIterator for PopulatedIter<'a, K, V> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedHashMap
            (first, self.iter)
        }
    }

    impl<'a, K, V, S> IntoPopulatedIterator for &'a PopulatedHashMap<K, V, S> {
        type PopulatedIntoIter = PopulatedIter<'a, K, V>;

        fn into_populated_iter(self) -> PopulatedIter<'a, K, V> {
            PopulatedIter {
                iter: self.into_iter(),
            }
        }
    }

    pub struct PopulatedIterMut<'a, K, V> {
        iter: std::collections::hash_map::IterMut<'a, K, V>,
    }

    impl<'a, K, V> IntoIterator for PopulatedIterMut<'a, K, V> {
        type Item = (&'a K, &'a mut V);
        type IntoIter = std::collections::hash_map::IterMut<'a, K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<'a, K, V> PopulatedIterator for PopulatedIterMut<'a, K, V> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedHashMap
            (first, self.iter)
        }
    }

    impl<'a, K, V, S> IntoPopulatedIterator for &'a mut PopulatedHashMap<K, V, S> {
        type PopulatedIntoIter = PopulatedIterMut<'a, K, V>;

        fn into_populated_iter(self) -> PopulatedIterMut<'a, K, V> {
            PopulatedIterMut {
                iter: self.0.iter_mut(),
            }
        }
    }

    pub struct PopulatedKeys<'a, K, V> {
        pub(crate) keys: std::collections::hash_map::Keys<'a, K, V>,
    }

    impl<'a, K, V> IntoIterator for PopulatedKeys<'a, K, V> {
        type Item = &'a K;
        type IntoIter = std::collections::hash_map::Keys<'a, K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.keys
        }
    }

    impl<'a, K, V> PopulatedIterator for PopulatedKeys<'a, K, V> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.keys.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedHashMap
            (first, self.keys)
        }
    }

    pub struct PopulatedValues<'a, K, V> {
        pub(crate) values: std::collections::hash_map::Values<'a, K, V>,
    }

    impl<'a, K, V> IntoIterator for PopulatedValues<'a, K, V> {
        type Item = &'a V;
        type IntoIter = std::collections::hash_map::Values<'a, K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.values
        }
    }

    impl<'a, K, V> PopulatedIterator for PopulatedValues<'a, K, V> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.values.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedHashMap
            (first, self.values)
        }
    }

    pub struct PopulatedValuesMut<'a, K, V> {
        pub(crate) values: std::collections::hash_map::ValuesMut<'a, K, V>,
    }

    impl<'a, K, V> IntoIterator for PopulatedValuesMut<'a, K, V> {
        type Item = &'a mut V;
        type IntoIter = std::collections::hash_map::ValuesMut<'a, K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.values
        }
    }

    impl<'a, K, V> PopulatedIterator for PopulatedValuesMut<'a, K, V> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.values.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedHashMap
            (first, self.values)
        }
    }

    pub struct PopulatedIntoKeys<K, V> {
        pub(crate) keys: std::collections::hash_map::IntoKeys<K, V>,
    }

    impl<K, V> IntoIterator for PopulatedIntoKeys<K, V> {
        type Item = K;
        type IntoIter = std::collections::hash_map::IntoKeys<K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.keys
        }
    }

    impl<K, V> PopulatedIterator for PopulatedIntoKeys<K, V> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.keys.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedHashMap
            (first, self.keys)
        }
    }

    pub struct PopulatedIntoValues<K, V> {
        pub(crate) values: std::collections::hash_map::IntoValues<K, V>,
    }

    impl<K, V> IntoIterator for PopulatedIntoValues<K, V> {
        type Item = V;
        type IntoIter = std::collections::hash_map::IntoValues<K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.values
        }
    }

    impl<K, V> PopulatedIterator for PopulatedIntoValues<K, V> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.values.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedHashMap
            (first, self.values)
        }
    }
}
/// A hash set that is populated i.e. guaranteed to have at least one element.
#[derive(Clone, Debug)]
pub struct PopulatedHashSet<T, S = RandomState>(HashSet<T, S>);

impl<T: Eq + Hash, S: BuildHasher> PartialEq for PopulatedHashSet<T, S> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl<T: Eq + Hash, S: BuildHasher> PartialEq<HashSet<T, S>> for PopulatedHashSet<T, S> {
    fn eq(&self, other: &HashSet<T, S>) -> bool {
        &self.0 == other
    }
}

impl<T: Eq + Hash, S: BuildHasher> PartialEq<PopulatedHashSet<T, S>> for HashSet<T, S> {
    fn eq(&self, other: &PopulatedHashSet<T, S>) -> bool {
        self == &other.0
    }
}

impl<T: Eq + Hash, S: BuildHasher> Eq for PopulatedHashSet<T, S> {}

impl<T, S> From<PopulatedHashSet<T, S>> for HashSet<T, S> {
    fn from(populated_hash_set: PopulatedHashSet<T, S>) -> HashSet<T, S> {
        populated_hash_set.0
    }
}

impl<T, S> TryFrom<HashSet<T, S>> for PopulatedHashSet<T, S> {
    type Error = HashSet<T, S>;

    fn try_from(hash_set: HashSet<T, S>) -> Result<PopulatedHashSet<T, S>, Self::Error> {
        if hash_set.is_empty() {
            Err(hash_set)
        } else {
            Ok(PopulatedHashSet(hash_set))
        }
    }
}

impl<T: Eq + Hash> PopulatedHashSet<T> {
    /// Creates a singleton populated hash set i.e. a hash set with a single value.
    ///
    /// # Example
    ///
    /// ```
    /// use populated::PopulatedHashSet;
    /// use std::num::NonZeroUsize;
    ///
    /// let hash_set = PopulatedHashSet::new(42);
    /// assert_eq!(hash_set.len(), NonZeroUsize::new(1).unwrap());
    /// ```
    pub fn new(value: T) -> PopulatedHashSet<T> {
        PopulatedHashSet(HashSet::from([value]))
    }

    /// Creates an empty `PopulatedHashSet` with the specified capacity
    /// and containing the given value.
    ///
    /// Note the capacity must be non-zero and a value must be
    /// supplied because this is a populated hash set i.e. non-empty
    /// hash set.
    ///
    /// # Example
    ///
    /// ```
    /// use populated::PopulatedHashSet;
    /// use std::num::NonZeroUsize;
    ///
    /// let hash_set = PopulatedHashSet::with_capacity(NonZeroUsize::new(1).unwrap(), 42);
    /// assert_eq!(hash_set.len(), NonZeroUsize::new(1).unwrap());
    /// ```
    pub fn with_capacity(capacity: NonZeroUsize, value: T) -> PopulatedHashSet<T> {
        let mut hash_set = HashSet::with_capacity(capacity.get());
        hash_set.insert(value);
        PopulatedHashSet(hash_set)
    }
}

impl<T, S> PopulatedHashSet<T, S> {
    /// Returns the number of elements the set can hold without reallocating.
    /// The capacity is returned as a `NonZeroUsize` because this is a populated hash set.
    ///
    /// # Example
    ///
    /// ```
    /// use populated::PopulatedHashSet;
    ///
    /// let mut hash_set = PopulatedHashSet::new(42);
    /// hash_set.insert(43);
    /// assert!(hash_set.capacity().get() >= 2);
    /// ```
    pub fn capacity(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.0.capacity()).unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedHashSet
    }

    /// Returns an iterator over the set.
    pub fn iter(&self) -> std::collections::hash_set::Iter<T> {
        self.0.iter()
    }

    /// Returns the number of elements in the set.
    ///
    /// The length is returned as a `NonZeroUsize` because this is a populated hash set.
    ///
    /// # Example
    ///
    /// ```
    /// use populated::PopulatedHashSet;
    ///
    /// let mut hash_set = PopulatedHashSet::new(42);
    /// hash_set.insert(43);
    /// assert_eq!(hash_set.len().get(), 2);
    /// ```
    pub fn len(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.0.len()).unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedHashSet
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements e such that `predicate(&e)`
    /// returns `false`.
    ///
    /// # Example
    ///
    /// ```
    /// use populated::PopulatedHashSet;
    ///
    /// let mut hash_set = PopulatedHashSet::new(42);
    /// hash_set.insert(43);
    /// let hash_set = hash_set.retain(|&e| e == 42);
    /// assert_eq!(hash_set.len(), 1);
    /// ```
    pub fn retain(self, predicate: impl FnMut(&T) -> bool) -> HashSet<T, S> {
        let mut hash_set = self.0;
        hash_set.retain(predicate);
        hash_set
    }

    /// Clears the set, removing all values.
    ///
    /// This method returns the set as a `HashSet` because after clearing
    /// the set is no longer guaranteed to be populated.
    ///
    /// # Example
    ///
    /// ```
    /// use populated::PopulatedHashSet;
    ///
    /// let mut hash_set = PopulatedHashSet::new(42);
    /// hash_set.insert(43);
    ///
    /// let hash_set = hash_set.clear();
    /// assert_eq!(hash_set.len(), 0);
    /// ```
    pub fn clear(self) -> HashSet<T, S> {
        let mut hash_set = self.0;
        hash_set.clear();
        hash_set
    }

    pub fn hasher(&self) -> &S {
        self.0.hasher()
    }
}

impl<T: Eq + Hash, S: BuildHasher> PopulatedHashSet<T, S> {
    /// Creates a new `PopulatedHashSet` with the given hasher and value.
    ///
    /// # Example
    ///
    /// ```
    /// use populated::PopulatedHashSet;
    /// use std::collections::hash_map::RandomState;
    ///
    /// let hash_set = PopulatedHashSet::with_hasher(RandomState::new(), 42);
    /// assert_eq!(hash_set.len().get(), 1);
    /// ```
    pub fn with_hasher(hash_builder: S, value: T) -> PopulatedHashSet<T, S> {
        let mut hash_set = HashSet::with_hasher(hash_builder);
        hash_set.insert(value);
        PopulatedHashSet(hash_set)
    }

    /// Creates an empty `PopulatedHashSet` with the specified capacity
    /// and containing the given value.
    ///
    /// Note the capacity must be non-zero and a value must be
    /// supplied because this is a populated hash set i.e. non-empty
    /// hash set.
    ///
    /// # Example
    ///
    /// ```
    /// use populated::PopulatedHashSet;
    /// use std::collections::hash_map::RandomState;
    /// use std::num::NonZeroUsize;
    ///
    /// let hash_set = PopulatedHashSet::with_capacity_and_hasher(NonZeroUsize::new(1).unwrap(), RandomState::new(), 42);
    /// assert_eq!(hash_set.len().get(), 1);
    /// ```
    pub fn with_capacity_and_hasher(
        capacity: NonZeroUsize,
        hasher: S,
        value: T,
    ) -> PopulatedHashSet<T, S> {
        let mut hash_set = HashSet::with_capacity_and_hasher(capacity.get(), hasher);
        hash_set.insert(value);
        PopulatedHashSet(hash_set)
    }

    /// Reserves capacity for at least additional more elements to be inserted in the set.
    pub fn reserve(&mut self, additional: usize) {
        self.0.reserve(additional);
    }

    /// Tries to reserve capacity for at least additional more elements to be inserted in the set.
    /// If the capacity is already sufficient, nothing happens. If allocation is needed and fails,
    /// an error is returned.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.0.try_reserve(additional)
    }

    /// Shrinks the capacity of the set as much as possible.
    pub fn shrink_to_fit(&mut self) {
        self.0.shrink_to_fit();
    }

    /// Shrinks the capacity of the set to the minimum needed to hold the elements.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.0.shrink_to(min_capacity);
    }

    /// Visits the values representing the difference, i.e., the values
    /// that are in self but not in other.
    pub fn difference<'a>(
        &'a self,
        other: &'a HashSet<T, S>,
    ) -> std::collections::hash_set::Difference<'a, T, S> {
        self.0.difference(other)
    }

    /// Visits the values representing the symmetric difference, i.e., the
    /// values that are in self or in other but not in both.
    pub fn symmetric_difference<'a>(
        &'a self,
        other: &'a HashSet<T, S>,
    ) -> std::collections::hash_set::SymmetricDifference<'a, T, S> {
        self.0.symmetric_difference(other)
    }

    /// Visits the values representing the intersection, i.e., the values
    /// that are both in `self` and `other`.
    ///
    /// When an equal element is present in `self` and `other` then the resulting
    /// `Intersection` may yield references to one or the other. This can be
    /// relevant if `T` contains fields which are not compared by its `Eq`
    /// implementation, and may hold different value between the two equal
    /// copies of `T` in the two sets.
    pub fn intersection<'a>(
        &'a self,
        other: &'a HashSet<T, S>,
    ) -> std::collections::hash_set::Intersection<'a, T, S> {
        self.0.intersection(other)
    }

    // TODO: union with populated iterator

    /// Returns true if the populated set contains a value.
    ///
    /// The value may be any borrowed form of the set’s value type, but
    /// `Hash` and `Eq` on the borrowed form must match those for the
    /// value type.
    pub fn contains<Q: Hash + Eq + ?Sized>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
    {
        self.0.contains(value)
    }

    /// Returns a reference to the value in the populated set, if any,
    /// that is equal to the given value.
    ///
    /// The value may be any borrowed form of the set’s value type, but
    /// `Hash` and `Eq` on the borrowed form must match those for the
    /// value type.
    pub fn get<Q: Hash + Eq + ?Sized>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
    {
        self.0.get(value)
    }

    /// Returns `true` if `self` has no elements in common with `other`. This is equivalent to checking for an empty intersection.
    pub fn is_disjoint(&self, other: &HashSet<T, S>) -> bool {
        self.0.is_disjoint(other)
    }

    /// Returns `true` if the set is a subset of another, i.e., `other` contains at least all the values in `self`.
    pub fn is_subset(&self, other: &HashSet<T, S>) -> bool {
        self.0.is_subset(other)
    }

    /// Returns `true` if the set is a superset of another, i.e., `self` contains at least all the values in `other`.
    pub fn is_superset(&self, other: &HashSet<T, S>) -> bool {
        self.0.is_superset(other)
    }

    /// Adds a value to the set.
    ///
    /// Returns whether the value was newly inserted. That is:
    ///
    /// - If the set did not previously contain this value, true is returned.
    /// - If the set already contained this value, false is returned, and the set is not modified: original value is not replaced, and the
    ///   value passed as argument is dropped.
    pub fn insert(&mut self, value: T) -> bool {
        self.0.insert(value)
    }

    /// Add a value to the hash set, returning the set as a `PopulatedHashSet` since inserting guarantees a `len()` > 0.
    ///
    /// Returns whether the value was newly inserted. That is:
    ///
    /// - If the set did not previously contain this value, true is returned.
    /// - If the set already contained this value, false is returned, and the set is not modified: original value is not replaced, and the
    ///  value passed as argument is dropped.
    ///
    /// This method is useful when you want to ensure that the set is populated after inserting a value.
    ///
    /// ```
    /// use std::collections::HashSet;
    /// use populated::PopulatedHashSet;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut hash_set = HashSet::from([42]);
    /// let (inserted, populated_hash_set) = PopulatedHashSet::inserted(hash_set, 43);
    /// assert_eq!(populated_hash_set.len(), NonZeroUsize::new(2).unwrap());
    /// assert!(inserted);
    /// ```
    pub fn inserted(mut hash_set: HashSet<T, S>, value: T) -> (bool, PopulatedHashSet<T, S>) {
        let inserted = hash_set.insert(value);
        (inserted, PopulatedHashSet(hash_set))
    }

    /// Adds a value to the set, replacing the existing value, if any, that is equal to the given one. Returns the replaced value.
    pub fn replace(&mut self, value: T) -> Option<T> {
        self.0.replace(value)
    }

    /// Removes a value from the set. Returns whether the value was present in the set.
    ///
    /// The value may be any borrowed form of the set’s value type, but `Hash` and `Eq` on the borrowed form *must* match those for the value
    /// type.
    pub fn remove<Q: Hash + Eq + ?Sized>(
        self,
        value: &Q,
    ) -> Result<HashSet<T, S>, PopulatedHashSet<T, S>>
    where
        T: Borrow<Q>,
    {
        let mut set = self.0;
        if set.remove(value) {
            Ok(set)
        } else {
            Err(PopulatedHashSet(set))
        }
    }

    /// Removes and returns the value in the set, if any, that is equal to the given one.
    ///
    /// The value may be any borrowed form of the set’s value type, but Hash and Eq on the borrowed form must match those for the value type.
    pub fn take<Q: Hash + Eq + ?Sized>(
        self,
        value: &Q,
    ) -> Result<(T, HashSet<T, S>), PopulatedHashSet<T, S>>
    where
        T: Borrow<Q>,
    {
        let mut set = self.0;
        if let Some(value) = set.take(value) {
            Ok((value, set))
        } else {
            Err(PopulatedHashSet(set))
        }
    }
}

impl<'a, T, S> IntoIterator for &'a PopulatedHashSet<T, S> {
    type Item = &'a T;
    type IntoIter = std::collections::hash_set::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<T, S> IntoIterator for PopulatedHashSet<T, S> {
    type Item = T;
    type IntoIter = std::collections::hash_set::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T: Eq + Hash + Clone, S: BuildHasher + Default> BitOr for &PopulatedHashSet<T, S> {
    type Output = PopulatedHashSet<T, S>;

    fn bitor(self, other: &PopulatedHashSet<T, S>) -> Self::Output {
        let hash_set = &self.0 | &other.0;
        PopulatedHashSet(hash_set)
    }
}

impl<T: Eq + Hash + Clone, S: BuildHasher + Default> BitOr<&HashSet<T, S>>
    for &PopulatedHashSet<T, S>
{
    type Output = PopulatedHashSet<T, S>;

    fn bitor(self, other: &HashSet<T, S>) -> Self::Output {
        let hash_set = &self.0 | other;
        PopulatedHashSet(hash_set)
    }
}

impl<T: Eq + Hash + Clone, S: BuildHasher + Default> BitOr<&PopulatedHashSet<T, S>>
    for &HashSet<T, S>
{
    type Output = PopulatedHashSet<T, S>;

    fn bitor(self, other: &PopulatedHashSet<T, S>) -> Self::Output {
        let hash_set = self | &other.0;
        PopulatedHashSet(hash_set)
    }
}
pub mod hash_set {

    use crate::{IntoPopulatedIterator, PopulatedHashSet, PopulatedIterator};

    pub struct IntoPopulatedIter<T> {
        iter: std::collections::hash_set::IntoIter<T>,
    }

    impl<T> IntoIterator for IntoPopulatedIter<T> {
        type Item = T;
        type IntoIter = std::collections::hash_set::IntoIter<T>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<T> PopulatedIterator for IntoPopulatedIter<T> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedHashSet
            (first, self.iter)
        }
    }

    impl<T, S> IntoPopulatedIterator for PopulatedHashSet<T, S> {
        type PopulatedIntoIter = IntoPopulatedIter<T>;

        fn into_populated_iter(self) -> IntoPopulatedIter<T> {
            IntoPopulatedIter {
                iter: self.into_iter(),
            }
        }
    }

    pub struct PopulatedIter<'a, T> {
        iter: std::collections::hash_set::Iter<'a, T>,
        first: &'a T,
    }

    impl<'a, T> IntoIterator for PopulatedIter<'a, T> {
        type Item = &'a T;
        type IntoIter = std::collections::hash_set::Iter<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<'a, T> PopulatedIterator for PopulatedIter<'a, T> {
        fn next(self) -> (Self::Item, Self::IntoIter) {
            (self.first, self.iter)
        }
    }

    impl<'a, T, S> IntoPopulatedIterator for &'a PopulatedHashSet<T, S> {
        type PopulatedIntoIter = PopulatedIter<'a, T>;

        fn into_populated_iter(self) -> PopulatedIter<'a, T> {
            let mut iter = self.iter();
            let first = iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedHashSet
            PopulatedIter { iter, first }
        }
    }
}

/// The result of removing an entry from a `PopulatedBTreeMap`.
pub type EntryRemovalResult<K, V> = Result<((K, V), BTreeMap<K, V>), PopulatedBTreeMap<K, V>>;

/// An ordered map based on a B-Tree that is guaranteed to have at least one key-value pair.
#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord)]
pub struct PopulatedBTreeMap<K, V>(BTreeMap<K, V>);

impl<K, V> From<PopulatedBTreeMap<K, V>> for BTreeMap<K, V> {
    fn from(populated_btree_map: PopulatedBTreeMap<K, V>) -> BTreeMap<K, V> {
        populated_btree_map.0
    }
}

impl<K, V> TryFrom<BTreeMap<K, V>> for PopulatedBTreeMap<K, V> {
    type Error = BTreeMap<K, V>;

    fn try_from(btree_map: BTreeMap<K, V>) -> Result<PopulatedBTreeMap<K, V>, Self::Error> {
        if btree_map.is_empty() {
            Err(btree_map)
        } else {
            Ok(PopulatedBTreeMap(btree_map))
        }
    }
}

impl<K: Ord, V> PopulatedBTreeMap<K, V> {
    /// Makes a new `PopulatedBTreeMap` with a single key-value pair.
    pub fn new(key: K, value: V) -> PopulatedBTreeMap<K, V> {
        PopulatedBTreeMap(BTreeMap::from([(key, value)]))
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map’s key type, but the ordering on the borrowed form must match the ordering on the key type.
    pub fn get<Q: Ord + ?Sized>(&self, k: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
    {
        self.0.get(k)
    }

    /// Returns the key-value pair corresponding to the supplied key.
    ///
    /// The supplied key may be any borrowed form of the map’s key type, but the ordering on the borrowed form must match the ordering on
    /// the key type.
    pub fn get_key_value<Q: Ord + ?Sized>(&self, k: &Q) -> Option<(&K, &V)>
    where
        K: Borrow<Q>,
    {
        self.0.get_key_value(k)
    }

    /// Returns the first key-value pair in the map. The key in this pair is the minimum key in the map.
    pub fn first_key_value(&self) -> (&K, &V) {
        self.0.first_key_value().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedBTreeMap
    }

    /// Removes and returns the first element in the map. The key of this element is the minimum key that was in the map.
    pub fn pop_first(self) -> ((K, V), BTreeMap<K, V>) {
        let mut btree_map = self.0;
        let first = btree_map.pop_first().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedBTreeMap
        (first, btree_map)
    }

    /// Returns the last key-value pair in the map. The key in this pair is the maximum key in the map.
    pub fn last_key_value(&self) -> (&K, &V) {
        self.0.last_key_value().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedBTreeMap
    }

    /// Removes and returns the last element in the map. The key of this element is the maximum key that was in the map.
    pub fn pop_last(self) -> (BTreeMap<K, V>, (K, V)) {
        let mut btree_map = self.0;
        let last = btree_map.pop_last().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedBTreeMap
        (btree_map, last)
    }

    /// Returns true if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map’s key type, but the
    /// ordering on the borrowed form must match the ordering on the key
    /// type.
    pub fn contains_key<Q: Ord + ?Sized>(&self, k: &Q) -> bool
    where
        K: Borrow<Q>,
    {
        self.0.contains_key(k)
    }

    /// Returns a mutable reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map’s key type, but the
    /// ordering on the borrowed form must match the ordering on the key
    /// type.
    pub fn get_mut<Q: Ord + ?Sized>(&mut self, k: &Q) -> Option<&mut V>
    where
        K: Borrow<Q>,
    {
        self.0.get_mut(k)
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, None is returned.
    ///
    /// If the map did have this key present, the value is updated, and
    /// the old value is returned. The key is not updated, though; this
    /// matters for types that can be == without being identical. See the
    /// module-level documentation for more.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.0.insert(key, value)
    }

    /// Removes a key from the map, returning the value at the key if the key was previously in the map.
    ///
    /// The key may be any borrowed form of the map’s key type, but the
    /// ordering on the borrowed form must match the ordering on the key
    /// type.
    pub fn remove<Q: Ord + ?Sized>(
        self,
        k: &Q,
    ) -> Result<(V, BTreeMap<K, V>), PopulatedBTreeMap<K, V>>
    where
        K: Borrow<Q>,
    {
        let mut btree_map = self.0;
        if let Some(value) = btree_map.remove(k) {
            Ok((value, btree_map))
        } else {
            Err(PopulatedBTreeMap(btree_map))
        }
    }

    /// Removes a key from the map, returning the stored key and value if the key was previously in the map.
    ///
    /// The key may be any borrowed form of the map’s key type, but the
    /// ordering on the borrowed form must match the ordering on the key
    /// type.
    pub fn remove_entry<Q: Ord + ?Sized>(self, k: &Q) -> EntryRemovalResult<K, V>
    where
        K: Borrow<Q>,
    {
        let mut btree_map = self.0;
        if let Some(tuple) = btree_map.remove_entry(k) {
            Ok((tuple, btree_map))
        } else {
            Err(PopulatedBTreeMap(btree_map))
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all pairs `(k, v)` for which `f(&k, &mut v)`
    /// returns `false`. The elements are visited in ascending key order.
    pub fn retain(self, predicate: impl FnMut(&K, &mut V) -> bool) -> BTreeMap<K, V> {
        let mut btree_map = self.0;
        btree_map.retain(predicate);
        btree_map
    }

    /// Moves all elements from other into self, leaving other empty.
    ///
    /// If a key from other is already present in self, the respective
    /// value from self will be overwritten with the respective value
    /// from other.
    pub fn append(&mut self, other: &mut BTreeMap<K, V>) {
        self.0.append(other);
    }

    /// Constructs a double-ended iterator over a sub-range of elements in
    /// the map. The simplest way is to use the range syntax `min..max`,
    /// thus `range(min..max)` will yield elements from min (inclusive) to
    /// max (exclusive). The range may also be entered as
    /// `(Bound<T>, Bound<T>)`, so for example
    /// `range((Excluded(4), Included(10)))` will yield a left-exclusive,
    /// right-inclusive range from 4 to 10.
    pub fn range<T: Ord + ?Sized>(
        &self,
        range: impl RangeBounds<T>,
    ) -> std::collections::btree_map::Range<'_, K, V>
    where
        K: Borrow<T>,
    {
        self.0.range(range)
    }

    pub fn split_at<Q: Ord + ?Sized>(self, key: &Q) -> (BTreeMap<K, V>, BTreeMap<K, V>)
    where
        K: Borrow<Q>,
    {
        let mut whole_btree_map = self.0;
        let other_half = whole_btree_map.split_off(key);
        (whole_btree_map, other_half)
    }
}

impl<K, V> PopulatedBTreeMap<K, V> {
    pub fn clear(self) -> BTreeMap<K, V> {
        let mut btree_map = self.0;
        btree_map.clear();
        btree_map
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.0.len()).unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedBTreeMap
    }
}

impl<K, V> IntoIterator for PopulatedBTreeMap<K, V> {
    type Item = (K, V);
    type IntoIter = std::collections::btree_map::IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, K, V> IntoIterator for &'a PopulatedBTreeMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = std::collections::btree_map::Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<'a, K, V> IntoIterator for &'a mut PopulatedBTreeMap<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = std::collections::btree_map::IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter_mut()
    }
}

pub mod btree_map {
    use crate::{IntoPopulatedIterator, PopulatedBTreeMap, PopulatedIterator};

    pub struct IntoPopulatedIter<K, V> {
        iter: std::collections::btree_map::IntoIter<K, V>,
    }

    impl<K, V> IntoIterator for IntoPopulatedIter<K, V> {
        type Item = (K, V);
        type IntoIter = std::collections::btree_map::IntoIter<K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<K, V> PopulatedIterator for IntoPopulatedIter<K, V> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedBTreeMap
            (first, self.iter)
        }
    }

    impl<K, V> IntoPopulatedIterator for PopulatedBTreeMap<K, V> {
        type PopulatedIntoIter = IntoPopulatedIter<K, V>;

        fn into_populated_iter(self) -> IntoPopulatedIter<K, V> {
            IntoPopulatedIter {
                iter: self.into_iter(),
            }
        }
    }

    pub struct PopulatedIter<'a, K, V> {
        iter: std::collections::btree_map::Iter<'a, K, V>,
    }

    impl<'a, K, V> IntoIterator for PopulatedIter<'a, K, V> {
        type Item = (&'a K, &'a V);
        type IntoIter = std::collections::btree_map::Iter<'a, K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<'a, K, V> PopulatedIterator for PopulatedIter<'a, K, V> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedBTreeMap
            (first, self.iter)
        }
    }

    impl<'a, K, V> IntoPopulatedIterator for &'a PopulatedBTreeMap<K, V> {
        type PopulatedIntoIter = PopulatedIter<'a, K, V>;

        fn into_populated_iter(self) -> PopulatedIter<'a, K, V> {
            PopulatedIter {
                iter: self.into_iter(),
            }
        }
    }

    pub struct PopulatedIterMut<'a, K, V> {
        iter: std::collections::btree_map::IterMut<'a, K, V>,
    }

    impl<'a, K, V> IntoIterator for PopulatedIterMut<'a, K, V> {
        type Item = (&'a K, &'a mut V);
        type IntoIter = std::collections::btree_map::IterMut<'a, K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<'a, K, V> PopulatedIterator for PopulatedIterMut<'a, K, V> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedBTreeMap
            (first, self.iter)
        }
    }

    impl<'a, K, V> IntoPopulatedIterator for &'a mut PopulatedBTreeMap<K, V> {
        type PopulatedIntoIter = PopulatedIterMut<'a, K, V>;

        fn into_populated_iter(self) -> PopulatedIterMut<'a, K, V> {
            PopulatedIterMut {
                iter: self.into_iter(),
            }
        }
    }

    pub struct PopulatedKeys<'a, K, V> {
        pub(crate) iter: std::collections::btree_map::Keys<'a, K, V>,
    }

    impl<'a, K, V> IntoIterator for PopulatedKeys<'a, K, V> {
        type Item = &'a K;
        type IntoIter = std::collections::btree_map::Keys<'a, K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<'a, K, V> PopulatedIterator for PopulatedKeys<'a, K, V> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedBTreeMap
            (first, self.iter)
        }
    }

    pub struct PopulatedValues<'a, K, V> {
        pub(crate) iter: std::collections::btree_map::Values<'a, K, V>,
    }

    impl<'a, K, V> IntoIterator for PopulatedValues<'a, K, V> {
        type Item = &'a V;
        type IntoIter = std::collections::btree_map::Values<'a, K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<'a, K, V> PopulatedIterator for PopulatedValues<'a, K, V> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedBTreeMap
            (first, self.iter)
        }
    }

    pub struct PopulatedValuesMut<'a, K, V> {
        pub(crate) iter: std::collections::btree_map::ValuesMut<'a, K, V>,
    }

    impl<'a, K, V> IntoIterator for PopulatedValuesMut<'a, K, V> {
        type Item = &'a mut V;
        type IntoIter = std::collections::btree_map::ValuesMut<'a, K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<'a, K, V> PopulatedIterator for PopulatedValuesMut<'a, K, V> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedBTreeMap
            (first, self.iter)
        }
    }

    pub struct PopulatedIntoKeys<K, V> {
        pub(crate) iter: std::collections::btree_map::IntoKeys<K, V>,
    }

    impl<K, V> IntoIterator for PopulatedIntoKeys<K, V> {
        type Item = K;
        type IntoIter = std::collections::btree_map::IntoKeys<K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<K, V> PopulatedIterator for PopulatedIntoKeys<K, V> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedBTreeMap
            (first, self.iter)
        }
    }

    pub struct PopulatedIntoValues<K, V> {
        pub(crate) iter: std::collections::btree_map::IntoValues<K, V>,
    }

    impl<K, V> IntoIterator for PopulatedIntoValues<K, V> {
        type Item = V;
        type IntoIter = std::collections::btree_map::IntoValues<K, V>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<K, V> PopulatedIterator for PopulatedIntoValues<K, V> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedBTreeMap
            (first, self.iter)
        }
    }
}

impl<K, V> PopulatedBTreeMap<K, V> {
    pub fn iter(&self) -> btree_map::PopulatedIter<K, V> {
        self.into_populated_iter()
    }

    pub fn iter_mut(&mut self) -> btree_map::PopulatedIterMut<K, V> {
        self.into_populated_iter()
    }

    pub fn keys(&self) -> btree_map::PopulatedKeys<K, V> {
        btree_map::PopulatedKeys {
            iter: self.0.keys(),
        }
    }

    pub fn values(&self) -> btree_map::PopulatedValues<K, V> {
        btree_map::PopulatedValues {
            iter: self.0.values(),
        }
    }

    pub fn values_mut(&mut self) -> btree_map::PopulatedValuesMut<K, V> {
        btree_map::PopulatedValuesMut {
            iter: self.0.values_mut(),
        }
    }

    pub fn into_keys(self) -> btree_map::PopulatedIntoKeys<K, V> {
        btree_map::PopulatedIntoKeys {
            iter: self.0.into_keys(),
        }
    }

    pub fn into_values(self) -> btree_map::PopulatedIntoValues<K, V> {
        btree_map::PopulatedIntoValues {
            iter: self.0.into_values(),
        }
    }
}

// Indexing BTreeMap
impl<K, V> std::ops::Index<&K> for PopulatedBTreeMap<K, V>
where
    K: Ord,
{
    type Output = V;

    fn index(&self, key: &K) -> &V {
        &self.0[key]
    }
}

/// A populated (i.e. guaranteed to be non-empty) ordered set based on a B-Tree.
#[derive(Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct PopulatedBTreeSet<T>(BTreeSet<T>);

impl<T> From<PopulatedBTreeSet<T>> for BTreeSet<T> {
    fn from(populated_btree_set: PopulatedBTreeSet<T>) -> BTreeSet<T> {
        populated_btree_set.0
    }
}

impl<T> TryFrom<BTreeSet<T>> for PopulatedBTreeSet<T> {
    type Error = BTreeSet<T>;

    fn try_from(btree_set: BTreeSet<T>) -> Result<PopulatedBTreeSet<T>, Self::Error> {
        if btree_set.is_empty() {
            Err(btree_set)
        } else {
            Ok(PopulatedBTreeSet(btree_set))
        }
    }
}

impl<T> PopulatedBTreeSet<T> {
    pub fn iter(&self) -> btree_set::PopulatedIter<T> {
        self.into_populated_iter()
    }
}

impl<T: Ord> PopulatedBTreeSet<T> {
    /// Creates a singleton set with the given value.
    pub fn new(value: T) -> PopulatedBTreeSet<T> {
        PopulatedBTreeSet(BTreeSet::from([value]))
    }

    /// Visits the elements representing the difference, i.e., the
    /// elements that are in self but not in other, in ascending order.
    pub fn difference<'a>(
        &'a self,
        other: &'a BTreeSet<T>,
    ) -> std::collections::btree_set::Difference<'a, T> {
        self.0.difference(other)
    }

    /// Visits the elements representing the symmetric difference, i.e.,
    /// the elements that are in self or in other but not in both, in
    /// ascending order.
    pub fn symmetric_difference<'a>(
        &'a self,
        other: &'a BTreeSet<T>,
    ) -> std::collections::btree_set::SymmetricDifference<'a, T> {
        self.0.symmetric_difference(other)
    }

    /// Visits the elements representing the intersection, i.e., the
    /// elements that are both in self and other, in ascending order.
    pub fn intersection<'a>(
        &'a self,
        other: &'a BTreeSet<T>,
    ) -> std::collections::btree_set::Intersection<'a, T> {
        self.0.intersection(other)
    }

    // TODO: union should return a populated iterator

    /// Returns true if the populated set contains an element equal to the
    /// value.
    ///
    /// The value may be any borrowed form of the set’s element type, but
    /// the ordering on the borrowed form must match the ordering on the
    /// element type.
    pub fn contains<Q: Ord + ?Sized>(&self, value: &Q) -> bool
    where
        T: Borrow<Q>,
    {
        self.0.contains(value)
    }

    /// Returns a reference to the element in the populated set, if any,
    /// that is equal to the value.
    ///
    /// The value may be any borrowed form of the set’s element type, but
    /// the ordering on the borrowed form must match the ordering on the
    /// element type.
    pub fn get<Q: Ord + ?Sized>(&self, value: &Q) -> Option<&T>
    where
        T: Borrow<Q>,
    {
        self.0.get(value)
    }

    /// Returns true if self has no elements in common with other. This
    /// is equivalent to checking for an empty intersection.
    pub fn is_disjoint(&self, other: &BTreeSet<T>) -> bool {
        self.0.is_disjoint(other)
    }

    /// Returns true if the set is a subset of another, i.e., other
    /// contains at least all the elements in self.
    pub fn is_subset(&self, other: &BTreeSet<T>) -> bool {
        self.0.is_subset(other)
    }

    /// Returns true if the set is a superset of another, i.e., self
    /// contains at least all the elements in other.
    pub fn is_superset(&self, other: &BTreeSet<T>) -> bool {
        self.0.is_superset(other)
    }

    /// Returns a reference to the first element in the populated set.
    /// This element is always the minimum of all elements in the set.
    pub fn first(&self) -> &T {
        self.0.first().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedBTreeSet
    }

    /// Returns a reference to the last element in the populated set. This
    /// element is always the maximum of all elements in the set.
    pub fn last(&self) -> &T {
        self.0.last().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedBTreeSet
    }

    /// Removes the first element from the populated set and returns it
    /// along with the remaining set. The first element is always the
    /// minimum element in the set.
    pub fn pop_first(self) -> (T, BTreeSet<T>) {
        let mut btree_set = self.0;
        let first = btree_set.pop_first().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedBTreeSet
        (first, btree_set)
    }

    /// Removes the last element from the set and returns it, if any. The
    /// last element is always the maximum element in the set.
    pub fn pop_last(self) -> (BTreeSet<T>, T) {
        let mut btree_set = self.0;
        let last = btree_set.pop_last().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedBTreeSet
        (btree_set, last)
    }

    /// Adds a value to the populated set.
    ///
    /// Returns whether the value was newly inserted. That is:
    ///
    /// - If the set did not previously contain an equal value, true is
    ///   returned.
    /// - If the set already contained an equal value, false is returned, and
    ///   the entry is not updated.
    pub fn insert(&mut self, value: T) -> bool {
        self.0.insert(value)
    }

    /// Adds a value to the set, replacing the existing element, if any,
    /// that is equal to the value. Returns the replaced element.
    pub fn replace(&mut self, value: T) -> Option<T> {
        self.0.replace(value)
    }

    /// Removes and returns the element in the set, if any, that is equal
    /// to the value.
    ///
    /// The value may be any borrowed form of the set’s element type, but
    /// the ordering on the borrowed form must match the ordering on the
    /// element type.
    pub fn take<Q: Ord + ?Sized>(self, value: &Q) -> Result<(T, BTreeSet<T>), PopulatedBTreeSet<T>>
    where
        T: Borrow<Q>,
    {
        let mut set = self.0;
        if let Some(value) = set.take(value) {
            Ok((value, set))
        } else {
            Err(PopulatedBTreeSet(set))
        }
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements `e` for which `f(&e)` returns
    /// `false`. The elements are visited in ascending order.
    pub fn retain(self, predicate: impl FnMut(&T) -> bool) -> BTreeSet<T> {
        let mut btree_set = self.0;
        btree_set.retain(predicate);
        btree_set
    }

    /// Moves all elements from other into self, leaving other empty.
    pub fn append(&mut self, other: &mut BTreeSet<T>) {
        self.0.append(other);
    }
}

impl<T> PopulatedBTreeSet<T> {
    /// Unpopulates the underlying set and returns it.
    pub fn clear(self) -> BTreeSet<T> {
        let mut btree_set = self.0;
        btree_set.clear();
        btree_set
    }

    /// Returns the number of elements in the set.
    pub fn len(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.0.len()).unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedBTreeSet
    }
}

impl<T> IntoIterator for PopulatedBTreeSet<T> {
    type Item = T;
    type IntoIter = std::collections::btree_set::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a PopulatedBTreeSet<T> {
    type Item = &'a T;
    type IntoIter = std::collections::btree_set::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

pub mod btree_set {

    use crate::{IntoPopulatedIterator, PopulatedBTreeSet, PopulatedIterator};

    pub struct IntoPopulatedIter<T> {
        iter: std::collections::btree_set::IntoIter<T>,
    }

    impl<T> IntoIterator for IntoPopulatedIter<T> {
        type Item = T;
        type IntoIter = std::collections::btree_set::IntoIter<T>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<T> PopulatedIterator for IntoPopulatedIter<T> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedBTreeSet
            (first, self.iter)
        }
    }

    impl<T> IntoPopulatedIterator for std::collections::BTreeSet<T> {
        type PopulatedIntoIter = IntoPopulatedIter<T>;

        fn into_populated_iter(self) -> IntoPopulatedIter<T> {
            IntoPopulatedIter {
                iter: self.into_iter(),
            }
        }
    }

    pub struct PopulatedIter<'a, T> {
        iter: std::collections::btree_set::Iter<'a, T>,
    }

    impl<'a, T> IntoIterator for PopulatedIter<'a, T> {
        type Item = &'a T;
        type IntoIter = std::collections::btree_set::Iter<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<'a, T> PopulatedIterator for PopulatedIter<'a, T> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedBTreeSet
            (first, self.iter)
        }
    }

    impl<'a, T> IntoPopulatedIterator for &'a PopulatedBTreeSet<T> {
        type PopulatedIntoIter = PopulatedIter<'a, T>;

        fn into_populated_iter(self) -> PopulatedIter<'a, T> {
            PopulatedIter {
                iter: self.into_iter(),
            }
        }
    }
}

impl<T: Ord + Clone> BitOr<&PopulatedBTreeSet<T>> for &PopulatedBTreeSet<T> {
    type Output = PopulatedBTreeSet<T>;

    fn bitor(self, other: &PopulatedBTreeSet<T>) -> Self::Output {
        let btree_set = &self.0 | &other.0;
        PopulatedBTreeSet(btree_set)
    }
}

impl<T: Ord + Clone> BitOr<&BTreeSet<T>> for &PopulatedBTreeSet<T> {
    type Output = PopulatedBTreeSet<T>;

    fn bitor(self, other: &BTreeSet<T>) -> Self::Output {
        let btree_set = &self.0 | other;
        PopulatedBTreeSet(btree_set)
    }
}

impl<T: Ord + Clone> BitOr<&PopulatedBTreeSet<T>> for &BTreeSet<T> {
    type Output = PopulatedBTreeSet<T>;

    fn bitor(self, other: &PopulatedBTreeSet<T>) -> Self::Output {
        let btree_set = self | &other.0;
        PopulatedBTreeSet(btree_set)
    }
}

#[derive(Clone, Debug)]
pub struct PopulatedBinaryHeap<T>(BinaryHeap<T>);

impl<T: Ord> PopulatedBinaryHeap<T> {
    /// Creates a binary heap populated with a single value.
    pub fn new(value: T) -> PopulatedBinaryHeap<T> {
        PopulatedBinaryHeap(BinaryHeap::from([value]))
    }

    /// Creates a binary populated with a single value and the given
    /// capacity.
    pub fn with_capacity(capacity: NonZeroUsize, value: T) -> PopulatedBinaryHeap<T> {
        let mut binary_heap = BinaryHeap::with_capacity(capacity.get());
        binary_heap.push(value);
        PopulatedBinaryHeap(binary_heap)
    }

    /// Pushes an item onto the binary heap.
    pub fn push(&mut self, item: T) {
        self.0.push(item);
    }

    /// Consumes the `PopulatedBinaryHeap` and returns a populated vector
    /// in sorted (ascending) order.
    pub fn into_sorted_vec(self) -> PopulatedVec<T> {
        let vec = self.0.into_sorted_vec();
        PopulatedVec(vec)
    }

    /// Moves all the elements of other into self, leaving other empty.
    pub fn append(&mut self, other: &mut BinaryHeap<T>) {
        self.0.append(other);
    }

    /// Retains only the elements specified by the predicate.
    ///
    /// In other words, remove all elements e for which f(&e) returns
    /// false. The elements are visited in unsorted (and unspecified) order.
    ///
    /// Note that the binary heap is not guaranteed to be populated after
    /// calling this method.
    ///
    /// ```
    /// use populated::PopulatedBinaryHeap;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut binary_heap = PopulatedBinaryHeap::with_capacity(NonZeroUsize::new(5).unwrap(), 1);
    /// binary_heap.push(2);
    /// binary_heap.push(3);
    /// let binary_heap = binary_heap.retain(|&x| x > 2);
    /// assert_eq!(binary_heap.len(), 1);
    /// ```
    pub fn retain(self, predicate: impl FnMut(&T) -> bool) -> BinaryHeap<T> {
        let mut binary_heap = self.0;
        binary_heap.retain(predicate);
        binary_heap
    }

    /// Removes the greatest item from the binary heap and returns it along
    /// with the remaining binary heap. Note that the returned binary heap
    /// is not guaranteed to be populated.
    ///
    /// ```
    /// use populated::PopulatedBinaryHeap;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut binary_heap = PopulatedBinaryHeap::with_capacity(NonZeroUsize::new(5).unwrap(), 1);
    /// binary_heap.push(2);
    /// binary_heap.push(3);
    /// let (item, binary_heap) = binary_heap.pop();
    /// assert_eq!(item, 3);
    /// assert_eq!(binary_heap.len(), 2);
    /// ```
    pub fn pop(self) -> (T, BinaryHeap<T>) {
        let mut binary_heap = self.0;
        let item = binary_heap.pop().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedBinaryHeap
        (item, binary_heap)
    }
}

impl<T> PopulatedBinaryHeap<T> {
    /// Returns the greatest item in the populated binary heap.
    ///
    /// ```
    /// use populated::PopulatedBinaryHeap;
    ///
    /// let mut binary_heap = PopulatedBinaryHeap::new(1);
    /// binary_heap.push(2);
    /// binary_heap.push(3);
    /// assert_eq!(binary_heap.peek(), &3);
    /// ```
    pub fn peek(&self) -> &T {
        self.0.peek().unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedBinaryHeap
    }

    /// Returns the number of elements the binary heap can hold without reallocating.
    ///
    /// ```
    /// use populated::PopulatedBinaryHeap;
    /// use std::num::NonZeroUsize;
    ///
    /// let binary_heap = PopulatedBinaryHeap::with_capacity(NonZeroUsize::new(10).unwrap(), 1);
    /// assert_eq!(binary_heap.capacity(), NonZeroUsize::new(10).unwrap());
    /// ```
    pub fn capacity(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.0.capacity()).unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedBinaryHeap
    }

    /// Discards as much additional capacity as possible.
    /// The capacity will never be less than the length.
    pub fn shrink_to_fit(&mut self) {
        self.0.shrink_to_fit();
    }

    /// Consumes the `PopulatedBinaryHeap` and returns a populated vector with elements in arbitrary order.
    ///
    /// ```
    /// use populated::PopulatedBinaryHeap;
    ///
    /// let mut binary_heap = PopulatedBinaryHeap::new(1);
    /// binary_heap.push(2);
    /// binary_heap.push(3);
    /// let vec = binary_heap.into_vec();
    /// assert_eq!(vec.len().get(), 3);
    /// ```
    pub fn into_vec(self) -> PopulatedVec<T> {
        let vec = self.0.into_vec();
        PopulatedVec(vec)
    }

    /// Returns the number of elements in the binary heap. Guaranteed to be non-zero.
    ///
    /// ```
    /// use populated::PopulatedBinaryHeap;
    ///
    /// let mut binary_heap = PopulatedBinaryHeap::new(1);
    /// binary_heap.push(2);
    /// binary_heap.push(3);
    /// assert_eq!(binary_heap.len().get(), 3);
    /// ```
    pub fn len(&self) -> NonZeroUsize {
        NonZeroUsize::new(self.0.len()).unwrap() // TODO: this can be done without unwrap safely because this is a PopulatedBinaryHeap
    }

    /// Drops all items from the binary heap. Since the underlying binary heap is now empty, the return value is the underlying binary heap that
    /// that is empty but still allocated.
    ///
    /// ```
    /// use populated::PopulatedBinaryHeap;
    /// use std::num::NonZeroUsize;
    ///
    /// let mut binary_heap = PopulatedBinaryHeap::with_capacity(NonZeroUsize::new(5).unwrap(), 1);
    /// binary_heap.push(2);
    /// binary_heap.push(3);
    /// let binary_heap = binary_heap.clear();
    /// assert_eq!(binary_heap.len(), 0);
    /// assert_eq!(binary_heap.capacity(), 5);
    /// ```
    pub fn clear(self) -> BinaryHeap<T> {
        let mut binary_heap = self.0;
        binary_heap.clear();
        binary_heap
    }
}

impl<T> IntoIterator for PopulatedBinaryHeap<T> {
    type Item = T;
    type IntoIter = std::collections::binary_heap::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a PopulatedBinaryHeap<T> {
    type Item = &'a T;
    type IntoIter = std::collections::binary_heap::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<T> From<PopulatedBinaryHeap<T>> for PopulatedVec<T> {
    fn from(populated_binary_heap: PopulatedBinaryHeap<T>) -> PopulatedVec<T> {
        PopulatedVec(Vec::from(populated_binary_heap.0))
    }
}

impl<T: Ord> From<PopulatedVec<T>> for PopulatedBinaryHeap<T> {
    fn from(populated_vec: PopulatedVec<T>) -> PopulatedBinaryHeap<T> {
        PopulatedBinaryHeap(BinaryHeap::from(populated_vec.0))
    }
}

pub mod binary_heap {
    use crate::{IntoPopulatedIterator, PopulatedBinaryHeap, PopulatedIterator};

    pub struct IntoPopulatedIter<T> {
        iter: std::collections::binary_heap::IntoIter<T>,
    }

    impl<T> IntoIterator for IntoPopulatedIter<T> {
        type Item = T;
        type IntoIter = std::collections::binary_heap::IntoIter<T>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<T> PopulatedIterator for IntoPopulatedIter<T> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedBinaryHeap
            (first, self.iter)
        }
    }

    impl<T> IntoPopulatedIterator for PopulatedBinaryHeap<T> {
        type PopulatedIntoIter = IntoPopulatedIter<T>;

        fn into_populated_iter(self) -> IntoPopulatedIter<T> {
            IntoPopulatedIter {
                iter: self.0.into_iter(),
            }
        }
    }

    pub struct PopulatedIter<'a, T> {
        iter: std::collections::binary_heap::Iter<'a, T>,
    }

    impl<'a, T> IntoIterator for PopulatedIter<'a, T> {
        type Item = &'a T;
        type IntoIter = std::collections::binary_heap::Iter<'a, T>;

        fn into_iter(self) -> Self::IntoIter {
            self.iter
        }
    }

    impl<'a, T> PopulatedIterator for PopulatedIter<'a, T> {
        fn next(mut self) -> (Self::Item, Self::IntoIter) {
            let first = self.iter.next().unwrap(); // TODO: this can be done without unwrap safely because this is a PopulatedBinaryHeap
            (first, self.iter)
        }
    }

    impl<'a, T> IntoPopulatedIterator for &'a PopulatedBinaryHeap<T> {
        type PopulatedIntoIter = PopulatedIter<'a, T>;

        fn into_populated_iter(self) -> PopulatedIter<'a, T> {
            PopulatedIter {
                iter: self.into_iter(),
            }
        }
    }
}

// Add tests for unsafe coercions involving PopulatedSlice

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;

    #[test]
    fn test_populated_vec() {
        let mut vec = PopulatedVec(vec![1, 2, 3]);
        assert_eq!(vec.len().get(), 3);
        assert_eq!(vec[0], 1);
        assert_eq!(vec[1], 2);
        assert_eq!(vec[2], 3);
        assert_eq!(vec.first(), &1);
        assert_eq!(vec.last(), &3);
        assert_eq!(vec.iter().collect::<Vec<_>>(), vec![&1, &2, &3]);
        assert_eq!(
            vec.iter_mut().collect::<Vec<_>>(),
            vec![&mut 1, &mut 2, &mut 3]
        );
        assert_eq!(vec.clone().into_iter().collect::<Vec<_>>(), vec![1, 2, 3]);
        assert_eq!(
            vec.clone().into_populated_iter().collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(vec.capacity(), NonZeroUsize::new(3).unwrap());
        assert_eq!(vec.clone().clear().len(), 0);
        assert_eq!(vec.into_inner(), vec![1, 2, 3]);

        let mut vec = PopulatedVec::with_capacity(NonZeroUsize::new(5).unwrap(), 1);
        vec.push(2);
        vec.push(3);
        assert_eq!(vec.len().get(), 3);
        assert_eq!(vec.capacity(), NonZeroUsize::new(5).unwrap());

        let mut other = vec![4, 5, 6];
        vec.append(&mut other);
        assert_eq!(vec.len().get(), 6);
    }

    #[test]
    fn test_populated_btree_map() {
        let mut btree_map = PopulatedBTreeMap::new(0, 0);
        btree_map.insert(1, 2);
        btree_map.insert(3, 4);
        assert_eq!(btree_map.len().get(), 3);
        assert_eq!(btree_map[&1], (2));
        assert_eq!(btree_map.get_mut(&1), Some(&mut 2));
        assert_eq!(btree_map.contains_key(&1), true);
        assert_eq!(btree_map.contains_key(&2), false);
        assert_eq!(
            btree_map.iter().collect::<Vec<_>>(),
            vec![(&0, &0), (&1, &2), (&3, &4)]
        );
        assert_eq!(
            btree_map.iter_mut().collect::<Vec<_>>(),
            vec![(&0, &mut 0), (&1, &mut 2), (&3, &mut 4)]
        );
        assert_eq!(
            btree_map.clone().into_iter().collect::<Vec<_>>(),
            vec![(0, 0), (1, 2), (3, 4)]
        );
        assert_eq!(
            btree_map.clone().into_populated_iter().collect::<Vec<_>>(),
            vec![(0, 0), (1, 2), (3, 4)]
        );

        let (value, regular_btree_map) = btree_map.clone().remove(&0).unwrap();
        assert_eq!(value, 0);
        assert_eq!(regular_btree_map.len(), 2);

        let mut other = BTreeMap::from([(4, 5), (6, 7)]);
        btree_map.append(&mut other);
        assert_eq!(btree_map.len().get(), 5);
    }

    #[test]
    fn test_populated_btree_set() {
        let mut btree_set = PopulatedBTreeSet::new(1);
        btree_set.insert(2);
        btree_set.insert(3);
        assert_eq!(btree_set.len().get(), 3);
        assert_eq!(btree_set.contains(&1), true);
        assert_eq!(btree_set.contains(&4), false);
        assert_eq!(btree_set.is_disjoint(&BTreeSet::from([4, 5, 6])), true);
        assert_eq!(btree_set.is_subset(&BTreeSet::from([1, 2, 3, 4])), true);
        assert_eq!(btree_set.is_superset(&BTreeSet::from([1, 2, 3])), true);
        assert_eq!(btree_set.first(), &1);
        assert_eq!(btree_set.last(), &3);
        assert_eq!(btree_set.iter().collect::<Vec<_>>(), vec![&1, &2, &3]);

        assert_eq!(
            btree_set.clone().into_iter().collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            btree_set.clone().into_populated_iter().collect::<Vec<_>>(),
            vec![&1, &2, &3]
        );

        assert_eq!(
            btree_set
                .clone()
                .difference(&BTreeSet::from([1, 2, 3]))
                .count(),
            0
        );
        assert_eq!(
            btree_set
                .clone()
                .symmetric_difference(&BTreeSet::from([1, 2, 3]))
                .count(),
            0
        );
        assert_eq!(
            btree_set
                .clone()
                .intersection(&BTreeSet::from([1, 2, 3]))
                .count(),
            3
        );

        assert_eq!(btree_set.clone().retain(|&x| x > 2).len(), 1);
        assert_eq!(btree_set.clone().pop_first().0, 1);
        assert_eq!(btree_set.clone().pop_last().1, 3);

        let mut other = BTreeSet::from([4, 5, 6]);
        btree_set.append(&mut other);
        assert_eq!(btree_set.len().get(), 6);
        assert_eq!(other.len(), 0);

        let btree_set = &btree_set | &BTreeSet::from([7, 8, 9]);
        assert_eq!(btree_set.len().get(), 9);

        // Check for equality
        let mut btree_set = PopulatedBTreeSet::new(1);
        btree_set.insert(2);
        btree_set.insert(3);
        let mut other = PopulatedBTreeSet::new(3);
        other.insert(2);
        other.insert(1);
        assert_eq!(btree_set, other);
    }

    #[test]
    fn test_populated_binary_heap() {
        let mut binary_heap = PopulatedBinaryHeap::new(1);
        binary_heap.push(2);
        binary_heap.push(3);
        assert_eq!(binary_heap.len().get(), 3);
        assert_eq!(binary_heap.peek(), &3);
        assert_eq!(binary_heap.clone().into_vec().len().get(), 3);
        assert_eq!(binary_heap.clone().into_sorted_vec().len().get(), 3);
        assert_eq!(binary_heap.clone().clear().len(), 0);
        assert_eq!(binary_heap.clone().pop().0, 3);
        assert_eq!(binary_heap.clone().pop().1.len(), 2);
        assert_eq!(binary_heap.clone().retain(|&x| x > 2).len(), 1);

        let mut other = BinaryHeap::from([4, 5, 6]);
        binary_heap.append(&mut other);
        assert_eq!(binary_heap.len().get(), 6);
    }

    #[test]
    fn test_populated_slice() {
        let test = [1, 2, 3];
        let slice: &PopulatedSlice<_> = TryFrom::try_from(&test[..]).unwrap();
        assert_eq!(slice.len().get(), 3);
        assert_eq!(slice[0], 1);
        assert_eq!(slice[1], 2);
        assert_eq!(slice[2], 3);
        assert_eq!(slice.first(), &1);
        assert_eq!(slice.last(), &3);
        assert_eq!(slice.iter().collect::<Vec<_>>(), vec![&1, &2, &3]);
    }

    #[test]
    fn test_populated_slice_mut() {
        let mut test = [1, 2, 3];
        let slice: &mut PopulatedSlice<_> = TryFrom::try_from(&mut test[..]).unwrap();
        assert_eq!(slice.len().get(), 3);
        assert_eq!(slice[0], 1);
        assert_eq!(slice[1], 2);
        assert_eq!(slice[2], 3);
        assert_eq!(slice.first(), &1);
        assert_eq!(slice.last(), &3);
        assert_eq!(slice.iter().collect::<Vec<_>>(), vec![&1, &2, &3]);
    }

    #[test]
    fn test_populated_slice_from_vec() {
        let vec = vec![1, 2, 3];
        let slice: &PopulatedSlice<_> = TryFrom::try_from(&vec[..]).unwrap();
        assert_eq!(slice.len().get(), 3);
        assert_eq!(slice[0], 1);
        assert_eq!(slice[1], 2);
        assert_eq!(slice[2], 3);
        assert_eq!(slice.first(), &1);
        assert_eq!(slice.last(), &3);
        assert_eq!(slice.iter().collect::<Vec<_>>(), vec![&1, &2, &3]);
    }

    #[test]
    fn test_populated_slice_mut_from_vec() {
        let mut vec = vec![1, 2, 3];
        let slice: &mut PopulatedSlice<_> = TryFrom::try_from(&mut vec[..]).unwrap();
        assert_eq!(slice.len().get(), 3);
        assert_eq!(slice[0], 1);
        assert_eq!(slice[1], 2);
        assert_eq!(slice[2], 3);
        assert_eq!(slice.first(), &1);
        assert_eq!(slice.last(), &3);
        assert_eq!(slice.iter().collect::<Vec<_>>(), vec![&1, &2, &3]);
    }

    #[test]
    fn test_populated_slice_from_slice() {
        let slice = &[1, 2, 3][..];
        let slice: &PopulatedSlice<_> = TryFrom::try_from(slice).unwrap();
        assert_eq!(slice.len().get(), 3);
        assert_eq!(slice[0], 1);
        assert_eq!(slice[1], 2);
        assert_eq!(slice[2], 3);
        assert_eq!(slice.first(), &1);
        assert_eq!(slice.last(), &3);
        assert_eq!(slice.iter().collect::<Vec<_>>(), vec![&1, &2, &3]);
    }

    #[test]
    fn test_populated_slice_mut_from_slice() {
        let slice = &mut [1, 2, 3][..];
        let slice: &mut PopulatedSlice<_> = TryFrom::try_from(slice).unwrap();
        assert_eq!(slice.len().get(), 3);
        assert_eq!(slice[0], 1);
        assert_eq!(slice[1], 2);
        assert_eq!(slice[2], 3);
        assert_eq!(slice.first(), &1);
        assert_eq!(slice.last(), &3);
        assert_eq!(slice.iter().collect::<Vec<_>>(), vec![&1, &2, &3]);
    }

    #[test]
    fn test_populated_slice_from_populated_vec() {
        let vec = PopulatedVec(vec![1, 2, 3]);
        let slice: &PopulatedSlice<_> = TryFrom::try_from(&vec[..]).unwrap();
        assert_eq!(slice.len().get(), 3);
        assert_eq!(slice[0], 1);
        assert_eq!(slice[1], 2);
        assert_eq!(slice[2], 3);
        assert_eq!(slice.first(), &1);
        assert_eq!(slice.last(), &3);
        assert_eq!(slice.iter().collect::<Vec<_>>(), vec![&1, &2, &3]);
    }

    #[test]
    fn test_populated_slice_mut_from_populated_vec() {
        let mut vec = PopulatedVec(vec![1, 2, 3]);
        let slice: &mut PopulatedSlice<_> = TryFrom::try_from(&mut vec[..]).unwrap();
        assert_eq!(slice.len().get(), 3);
        assert_eq!(slice[0], 1);
        assert_eq!(slice[1], 2);
        assert_eq!(slice[2], 3);
        assert_eq!(slice.first(), &1);
        assert_eq!(slice.last(), &3);
        assert_eq!(slice.iter().collect::<Vec<_>>(), vec![&1, &2, &3]);
    }

    #[test]
    fn test_populated_slice_from_populated_slice() {
        let slice: &PopulatedSlice<_> = TryFrom::try_from(&[1, 2, 3][..]).unwrap();
        assert_eq!(slice.len().get(), 3);
        assert_eq!(slice[0], 1);
        assert_eq!(slice[1], 2);
        assert_eq!(slice[2], 3);
        assert_eq!(slice.first(), &1);
        assert_eq!(slice.last(), &3);
        assert_eq!(slice.iter().collect::<Vec<_>>(), vec![&1, &2, &3]);
    }

    #[test]
    fn test_populated_slice_mut_from_populated_slice() {
        let mut array = [1, 2, 3];
        let slice: &mut PopulatedSlice<_> = TryFrom::try_from(&mut array[..]).unwrap();
        assert_eq!(slice.len().get(), 3);
        assert_eq!(slice[0], 1);
        assert_eq!(slice[1], 2);
        assert_eq!(slice[2], 3);
        assert_eq!(slice.first(), &1);
        assert_eq!(slice.last(), &3);
        assert_eq!(slice.iter().collect::<Vec<_>>(), vec![&1, &2, &3]);
    }

    #[test]
    fn test_populated_vec_deque() {
        let mut vec_deque = PopulatedVecDeque::new(1);
        vec_deque.push_back(2);
        vec_deque.push_back(3);
        assert_eq!(vec_deque.len().get(), 3);
        assert_eq!(vec_deque[0], 1);
        assert_eq!(vec_deque[1], 2);
        assert_eq!(vec_deque[2], 3);
        assert_eq!(vec_deque.front(), &1);
        assert_eq!(vec_deque.back(), &3);
        assert_eq!(vec_deque.iter().collect::<Vec<_>>(), vec![&1, &2, &3]);
        assert_eq!(
            vec_deque.iter_mut().collect::<Vec<_>>(),
            vec![&mut 1, &mut 2, &mut 3]
        );
        assert_eq!(
            vec_deque.clone().into_iter().collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(
            vec_deque.clone().into_populated_iter().collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
        assert_eq!(vec_deque.clone().clear().len(), 0);
        assert_eq!(vec_deque.clone().pop_front(), (1, VecDeque::from([2, 3])));
        assert_eq!(vec_deque.clone().pop_back(), (VecDeque::from([1, 2]), 3));
    }

    #[test]
    fn test_populated_hash_map() {
        use std::collections::HashSet;

        let mut hash_map = PopulatedHashMap::new(1, 2);
        hash_map.insert(3, 4);

        assert_eq!(hash_map.len().get(), 2);
        assert_eq!(hash_map[&1], 2);
        assert_eq!(hash_map.get_mut(&1), Some(&mut 2));
        assert_eq!(hash_map.contains_key(&1), true);
        assert_eq!(hash_map.contains_key(&2), false);

        let clone_iter_set: HashSet<_> = hash_map.clone().into_iter().collect();
        let expected_clone_set: HashSet<_> = vec![(1, 2), (3, 4)].into_iter().collect();
        assert_eq!(clone_iter_set, expected_clone_set);

        let clone_populated_iter_set: HashSet<_> = hash_map.clone().into_populated_iter().collect();
        let expected_populated_set: HashSet<_> = vec![(1, 2), (3, 4)].into_iter().collect();
        assert_eq!(clone_populated_iter_set, expected_populated_set);
    }

    #[test]
    fn test_populated_hash_set() {
        let mut hash_set = PopulatedHashSet::new(1);
        hash_set.insert(2);
        hash_set.insert(3);
        assert_eq!(hash_set.len().get(), 3);
        assert_eq!(hash_set.contains(&1), true);
        assert_eq!(hash_set.contains(&4), false);
        assert_eq!(hash_set.is_disjoint(&HashSet::from([4, 5, 6])), true);
        assert_eq!(hash_set.is_subset(&HashSet::from([1, 2, 3, 4])), true);
        assert_eq!(hash_set.is_superset(&HashSet::from([1, 2, 3])), true);

        assert_eq!(hash_set, HashSet::from([1, 2, 3]));
    }
}
