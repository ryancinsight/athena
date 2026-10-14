//! The deflation basis: an orthonormal subspace with its factorised coarse
//! matrix, and the two-level projections built from it.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;
use eunomia::NumericElement;

use crate::{KrylovBackend, LinearOperator};

use super::lu::{CoarseFailure, factorise, solve_in_place};

/// An orthonormal deflation subspace `U` with the factorised coarse matrix
/// `Uᵀ A U`.
///
/// Construction orthonormalises the input vectors (modified Gram–Schmidt) —
/// the Euclidean complement projection needs an orthonormal `U`, so the
/// invariant is earned here, once. Construction caches `A U` and factorises
/// `E = Uᵀ A U` with partial pivoting. The Schur complement
/// `A − A U E⁻¹ Uᵀ A` includes coarse/complement coupling for every supported
/// basis; invariance is not required for correctness. The subspace must be
/// used with the same operator it was built from.
///
/// With an exact invariant subspace of `k` eigenvectors of an SPD operator,
/// the deflated solve terminates in at most `n − k` iterations: the
/// complement spectrum has `n − k` values, and the two-level correction
/// resolves the deflated directions exactly. That finite termination is the
/// oracle the deflation contract pins.
pub struct DeflationSubspace<B: KrylovBackend> {
    basis: B::VectorBlock,
    /// Cached operator images `A U`, shared by Schur applications and RHS
    /// construction. Each deflated application still uses one base apply.
    images: B::VectorBlock,
    /// The factorised coarse matrix `Uᵀ A U` (row-major, LU in place).
    coarse: Vec<B::Scalar>,
    /// Row-interchange record of the coarse factorisation.
    pivots: Vec<usize>,
    dimension: usize,
    width: usize,
}

#[expect(
    clippy::missing_fields_in_debug,
    reason = "the basis contents are the caller's data; diagnostics carry the shape"
)]
impl<B: KrylovBackend> fmt::Debug for DeflationSubspace<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The basis block and coarse factorisation are shape-only in
        // diagnostics: the vectors' contents are the caller's data.
        formatter
            .debug_struct("DeflationSubspace")
            .field("dimension", &self.dimension)
            .field("width", &self.width)
            .finish()
    }
}

/// Why a deflation subspace was rejected, or the backend failure that
/// stopped its construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeflationSubspaceError<E> {
    /// A deflation vector's normalisation vanished: the input is zero or
    /// numerically dependent on the vectors before it.
    Degenerate {
        /// The index of the vector that could not be orthonormalised.
        index: usize,
    },
    /// The coarse matrix `Uᵀ A U` is singular: a basis direction is
    /// `A`-orthogonal to the whole space.
    SingularCoarse {
        /// The elimination step that lost rank.
        step: usize,
    },
    /// A backend operation failed during assembly.
    Backend {
        /// The backend's own failure.
        source: E,
    },
    /// The candidate count is outside `1..=dimension`: an empty deflation
    /// or more vectors than the space can hold.
    Invalid {
        /// The candidate vector count.
        width: usize,
        /// The operator dimension.
        dimension: usize,
    },
}

impl<B: KrylovBackend> DeflationSubspace<B> {
    /// Build the subspace from candidate deflation vectors.
    ///
    /// The input vectors are orthonormalised (modified Gram–Schmidt), so any
    /// spanning set of the intended subspace is accepted; the operator is
    /// applied `width` times to assemble the coarse matrix.
    ///
    /// # Errors
    ///
    /// Returns a typed construction failure for a degenerate input vector
    /// or a singular coarse matrix; backend failures carry their source.
    pub fn from_vectors<O>(
        backend: &B,
        operator: &O,
        vectors: &[B::Vector],
    ) -> Result<Self, DeflationSubspaceError<B::Error>>
    where
        O: LinearOperator<B>,
    {
        let dimension = operator.dimension();
        let width = vectors.len();
        let zero = <B::Scalar as NumericElement>::ZERO;
        let one = <B::Scalar as NumericElement>::ONE;
        if width == 0 || width > dimension {
            return Err(DeflationSubspaceError::Invalid { width, dimension });
        }
        let mut basis = backend
            .allocate_block(width, dimension)
            .map_err(|source| DeflationSubspaceError::Backend { source })?;
        let mut work = backend
            .allocate(dimension)
            .map_err(|source| DeflationSubspaceError::Backend { source })?;

        // Modified Gram–Schmidt: each vector is normalised against every
        // vector already accepted, so a repeated or numerically dependent
        // input collapses to zero norm and is rejected.
        for (index, vector) in vectors.iter().enumerate() {
            backend
                .copy(backend.view(vector), backend.view_mut(&mut work))
                .map_err(|source| DeflationSubspaceError::Backend { source })?;
            for prior in 0..index {
                let coefficient = backend
                    .dot(backend.block_view(&basis, prior), backend.view(&work))
                    .map_err(|source| DeflationSubspaceError::Backend { source })?;
                backend
                    .axpy(
                        backend.view_mut(&mut work),
                        backend.block_view(&basis, prior),
                        -coefficient,
                    )
                    .map_err(|source| DeflationSubspaceError::Backend { source })?;
            }
            let norm = backend
                .norm_l2(backend.view(&work))
                .map_err(|source| DeflationSubspaceError::Backend { source })?;
            if norm == zero || !norm.is_finite() {
                return Err(DeflationSubspaceError::Degenerate { index });
            }
            backend
                .copy(
                    backend.view(&work),
                    backend.block_view_mut(&mut basis, index),
                )
                .map_err(|source| DeflationSubspaceError::Backend { source })?;
            backend
                .scale(backend.block_view_mut(&mut basis, index), one / norm)
                .map_err(|source| DeflationSubspaceError::Backend { source })?;
        }

        // Coarse matrix `Uᵀ A U`, assembled column by column through the
        // operator image of each basis vector.
        let mut coarse = Vec::with_capacity(width * width);
        let mut image = backend
            .allocate(dimension)
            .map_err(|source| DeflationSubspaceError::Backend { source })?;
        let mut images = backend
            .allocate_block(width, dimension)
            .map_err(|source| DeflationSubspaceError::Backend { source })?;
        for column in 0..width {
            operator
                .apply(
                    backend,
                    backend.block_view(&basis, column),
                    backend.view_mut(&mut image),
                )
                .map_err(|source| DeflationSubspaceError::Backend { source })?;
            backend
                .copy(
                    backend.view(&image),
                    backend.block_view_mut(&mut images, column),
                )
                .map_err(|source| DeflationSubspaceError::Backend { source })?;
            for row in 0..width {
                let value = backend
                    .dot(backend.block_view(&basis, row), backend.view(&image))
                    .map_err(|source| DeflationSubspaceError::Backend { source })?;
                coarse.push(value);
            }
        }
        let mut pivots = vec![0; width];
        if let Err(CoarseFailure::ZeroPivot { step, .. }) =
            factorise(&mut coarse, width, &mut pivots)
        {
            return Err(DeflationSubspaceError::SingularCoarse { step });
        }

        Ok(Self {
            basis,
            images,
            coarse,
            pivots,
            dimension,
            width,
        })
    }

    /// The deflation-subspace dimension `k`.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// The operator dimension `n`.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// `target ← (I − UUᵀ)·v`, the complement projection of `v`.
    ///
    /// # Errors
    ///
    /// Returns the backend's copy, dot, and axpy failures.
    pub fn project(
        &self,
        backend: &B,
        vector: &B::Vector,
        target: &mut B::Vector,
    ) -> Result<(), B::Error> {
        backend.copy(backend.view(vector), backend.view_mut(target))?;
        for column in 0..self.width {
            let coefficient = backend.dot(
                backend.block_view(&self.basis, column),
                backend.view(vector),
            )?;
            backend.axpy(
                backend.view_mut(target),
                backend.block_view(&self.basis, column),
                -coefficient,
            )?;
        }
        Ok(())
    }

    /// `target ← U·(Uᵀ A U)⁻¹·Uᵀ·v`, the coarse-space correction of `v`.
    ///
    /// # Errors
    ///
    /// Returns the backend's dot, copy, scale, and axpy failures.
    pub fn coarse_correction(
        &self,
        backend: &B,
        vector: &B::Vector,
        target: &mut B::Vector,
    ) -> Result<(), B::Error> {
        let mut coordinates = vec![B::Scalar::ZERO; self.width];
        self.combine_coarse(backend, vector, target, &self.basis, &mut coordinates)
    }

    /// `target ← A U E⁻¹ Uᵀ vector`, using the cached images A U.
    pub(super) fn coarse_image(
        &self,
        backend: &B,
        vector: &B::Vector,
        target: &mut B::Vector,
        coordinates: &mut [B::Scalar],
    ) -> Result<(), B::Error> {
        self.combine_coarse(backend, vector, target, &self.images, coordinates)
    }

    fn combine_coarse(
        &self,
        backend: &B,
        vector: &B::Vector,
        target: &mut B::Vector,
        block: &B::VectorBlock,
        coordinates: &mut [B::Scalar],
    ) -> Result<(), B::Error> {
        for (column, coordinate) in coordinates.iter_mut().enumerate() {
            *coordinate = backend.dot(
                backend.block_view(&self.basis, column),
                backend.view(vector),
            )?;
        }
        solve_in_place(&self.coarse, self.width, &self.pivots, coordinates);
        backend.copy(backend.block_view(block, 0), backend.view_mut(target))?;
        backend.scale(backend.view_mut(target), coordinates[0])?;
        for (index, coordinate) in coordinates.iter().enumerate().skip(1) {
            backend.axpy(
                backend.view_mut(target),
                backend.block_view(block, index),
                *coordinate,
            )?;
        }
        Ok(())
    }

    /// Borrow the basis block (for the deflated operator's projections).
    pub(super) const fn basis(&self) -> &B::VectorBlock {
        &self.basis
    }
}
