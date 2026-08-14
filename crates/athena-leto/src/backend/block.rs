use leto::{Array1, ArrayView1, ArrayViewMut1, Layout};

use crate::LetoBackendError;

/// Contiguous host storage for a fixed count of equal-length vectors.
///
/// The whole set is one Leto array. Vector `index` occupies
/// `[index * len, (index + 1) * len)`, and borrowing it yields a view over
/// exactly that range, so a recurrence reads and writes the same elements it
/// would with one allocation per vector. What changes is placement: an Arnoldi
/// basis becomes one extent the prefetcher can walk, instead of `count`
/// independently placed ones whose addresses the allocator chose.
///
/// The stride between vectors is `len`, with no padding. Individual vectors
/// are therefore aligned only to `T` rather than to a vector-unit boundary.
/// That is the deliberate trade: the basis is traversed one vector at a time
/// by kernels that tolerate an unaligned prologue, and padding to restore
/// per-vector alignment would reintroduce the dead space between vectors that
/// flattening exists to remove.
pub struct LetoVectorBlock<T> {
    storage: Array1<T>,
    len: usize,
    count: usize,
}

impl<T: Default + Clone> LetoVectorBlock<T> {
    /// Allocate `count` zero-initialized vectors of `len` elements.
    pub(super) fn new(count: usize, len: usize) -> Result<Self, LetoBackendError> {
        let extent = count
            .checked_mul(len)
            .ok_or(LetoBackendError::BlockExtentOverflow { count, len })?;
        Ok(Self {
            storage: Array1::zeros([extent]),
            len,
            count,
        })
    }
}

impl<T> LetoVectorBlock<T> {
    /// Borrow vector `index` immutably.
    pub(super) fn view(&self, index: usize) -> ArrayView1<'_, T> {
        let data = self
            .span(index)
            .and_then(|span| self.storage.as_slice()?.get(span))
            .expect("invariant: block storage is contiguous and index is below count");
        ArrayView1::new(self.vector_layout(), data)
    }

    /// Borrow vector `index` for writing.
    pub(super) fn view_mut(&mut self, index: usize) -> ArrayViewMut1<'_, T> {
        let layout = self.vector_layout();
        let span = self.span(index);
        let data = span
            .and_then(|span| self.storage.as_slice_mut()?.get_mut(span))
            .expect("invariant: block storage is contiguous and index is below count");
        ArrayViewMut1::new(layout, data)
    }

    /// Element range backing vector `index`, or `None` past the block.
    ///
    /// A zero-length block admits no index at all: without the `count` test,
    /// every index there would produce the same empty range and an
    /// out-of-range borrow would silently succeed.
    fn span(&self, index: usize) -> Option<core::ops::Range<usize>> {
        if index >= self.count {
            return None;
        }
        let start = index * self.len;
        Some(start..start + self.len)
    }

    /// Dense rank-one layout of one vector within the block.
    ///
    /// The offset is zero because [`Self::span`] has already positioned the
    /// backing slice. Carrying the offset in the layout instead would leave
    /// every downstream `as_slice` resolving against the block's base pointer.
    const fn vector_layout(&self) -> Layout<1> {
        Layout::new([self.len], [1], 0)
    }
}
