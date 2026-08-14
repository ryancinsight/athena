//! Value-semantic conformance for the Leto `KrylovBackend` seam.
//!
//! The vector-block cases pin the contiguity claim itself. Every other suite
//! would still pass if `allocate_block` returned independently allocated
//! vectors, because the recurrence only ever asks for indexed views — so
//! without these, the layout the block exists to provide is unverified.

use athena_core::KrylovBackend;
use athena_leto::LetoBackend;

const COUNT: usize = 4;
const LEN: usize = 8;

fn block(backend: LetoBackend<f64>) -> <LetoBackend<f64> as KrylovBackend>::VectorBlock {
    backend
        .allocate_block(COUNT, LEN)
        .expect("invariant: host allocation succeeds")
}

/// The `index`th vector's contents, counting from `index * LEN`.
///
/// Values are distinguishable per index so an off-by-one in the span
/// arithmetic surfaces as a neighbour's contents rather than passing.
#[expect(
    clippy::cast_precision_loss,
    reason = "indices below COUNT * LEN are exactly representable in f64"
)]
fn expected_contents(index: usize) -> [f64; LEN] {
    core::array::from_fn(|offset| (index * LEN + offset) as f64)
}

fn vector_slice<'a>(
    backend: &'a LetoBackend<f64>,
    block: &'a <LetoBackend<f64> as KrylovBackend>::VectorBlock,
    index: usize,
) -> &'a [f64] {
    backend
        .block_view(block, index)
        .as_slice()
        .expect("invariant: block views are contiguous")
}

#[test]
fn a_vector_block_occupies_one_contiguous_extent() {
    let backend = LetoBackend::<f64>::default();
    let block = block(backend);

    let base = vector_slice(&backend, &block, 0).as_ptr();
    for index in 0..COUNT {
        let vector = vector_slice(&backend, &block, index);
        assert_eq!(
            vector.len(),
            LEN,
            "vector {index} must carry the full length"
        );
        assert_eq!(
            vector.as_ptr(),
            base.wrapping_add(index * LEN),
            "vector {index} must begin at its flat offset from the block base"
        );
    }
}

#[test]
fn a_vector_block_is_zero_initialized() {
    let backend = LetoBackend::<f64>::default();
    let block = block(backend);

    for index in 0..COUNT {
        assert_eq!(
            vector_slice(&backend, &block, index),
            [0.0_f64; LEN],
            "vector {index} must be zeroed"
        );
    }
}

#[test]
fn writing_one_block_vector_leaves_its_neighbours_untouched() {
    let backend = LetoBackend::<f64>::default();
    let mut block = block(backend);

    for index in 0..COUNT {
        let mut view = backend.block_view_mut(&mut block, index);
        let target = view
            .as_mut_slice()
            .expect("invariant: block views are contiguous");
        target.copy_from_slice(&expected_contents(index));
    }

    for index in 0..COUNT {
        assert_eq!(
            vector_slice(&backend, &block, index),
            expected_contents(index),
            "vector {index} does not hold its own contents"
        );
    }
}

#[test]
#[should_panic(expected = "index is below count")]
fn borrowing_past_the_block_panics_rather_than_aliasing() {
    let backend = LetoBackend::<f64>::default();
    let block = block(backend);

    let _ = backend.block_view(&block, COUNT);
}

#[test]
#[should_panic(expected = "index is below count")]
fn a_zero_length_block_admits_no_index() {
    let backend = LetoBackend::<f64>::default();
    let block = backend
        .allocate_block(0, 0)
        .expect("invariant: an empty block allocates");

    let _ = backend.block_view(&block, 0);
}

#[test]
fn a_block_of_zero_length_vectors_still_lends_every_index() {
    let backend = LetoBackend::<f64>::default();
    let block = backend
        .allocate_block(COUNT, 0)
        .expect("invariant: host allocation succeeds");

    for index in 0..COUNT {
        assert!(
            vector_slice(&backend, &block, index).is_empty(),
            "vector {index} of a zero-length system must be empty, not absent"
        );
    }
}
