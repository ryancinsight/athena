use core::cell::RefCell;

use crate::{
    Cg, CgWorkspace, ConvergencePolicy, DeflationSubspace, Identity, KrylovBackend, LinearOperator,
    SolveError, SolveReport,
};

use super::super::dimension::validate_dimensions;

/// The two-sided projected operator `A_def = (I − UUᵀ)·A·(I − UUᵀ)`.
///
/// Symmetric for any orthonormal `U` — both projections are Euclidean
/// orthogonal — and positive definite on the complement of `U`, where it
/// acts as the restriction of the SPD base operator. That is what lets the
/// deflated solve run through the ordinary CG recurrence.
///
/// `apply` needs one scratch vector for the intermediate
/// `(I − UUᵀ)·v`; the scratch is allocated at construction and borrowed
/// per call, so the operator is single-threaded per solve — `!Sync` by
/// construction, matching the seam it serves.
pub struct DeflatedOperator<'a, B, O>
where
    B: KrylovBackend,
    O: LinearOperator<B>,
{
    operator: &'a O,
    subspace: &'a DeflationSubspace<B>,
    /// `(projected input, operator image)` — the two intermediates `apply`
    /// composes through. They are allocated at construction and borrowed per
    /// call, so the operator is single-threaded per solve — `!Sync` by
    /// construction, matching the seam it serves.
    scratch: RefCell<(B::Vector, B::Vector)>,
    width: usize,
}

impl<'a, B, O> DeflatedOperator<'a, B, O>
where
    B: KrylovBackend,
    O: LinearOperator<B>,
{
    /// Build the deflated view of `operator` over `subspace`.
    ///
    /// # Errors
    ///
    /// Returns the backend's allocation failure for the internal scratch
    /// pair.
    pub fn new(
        backend: &B,
        operator: &'a O,
        subspace: &'a DeflationSubspace<B>,
    ) -> Result<Self, B::Error> {
        let dimension = subspace.dimension();
        Ok(Self {
            operator,
            subspace,
            scratch: RefCell::new((backend.allocate(dimension)?, backend.allocate(dimension)?)),
            width: subspace.width(),
        })
    }
}

impl<B, O> LinearOperator<B> for DeflatedOperator<'_, B, O>
where
    B: KrylovBackend,
    O: LinearOperator<B>,
{
    fn dimension(&self) -> usize {
        self.operator.dimension()
    }

    fn apply(
        &self,
        backend: &B,
        input: B::View<'_>,
        output: B::ViewMut<'_>,
    ) -> Result<(), B::Error> {
        let mut scratch = self.scratch.borrow_mut();
        let (projected, image) = &mut *scratch;
        // projected <- (I - U U^T) v
        backend.copy(input, backend.view_mut(projected))?;
        for column in 0..self.width {
            let coefficient = backend.dot(
                backend.block_view(self.subspace.basis(), column),
                backend.view(projected),
            )?;
            backend.axpy(
                backend.view_mut(projected),
                backend.block_view(self.subspace.basis(), column),
                -coefficient,
            )?;
        }
        // image <- A · projected, then re-project onto the complement: the
        // second projection is what makes the operator symmetric for any
        // orthonormal basis.
        self.operator
            .apply(backend, backend.view(projected), backend.view_mut(image))?;
        for column in 0..self.width {
            let coefficient = backend.dot(
                backend.block_view(self.subspace.basis(), column),
                backend.view(image),
            )?;
            backend.axpy(
                backend.view_mut(image),
                backend.block_view(self.subspace.basis(), column),
                -coefficient,
            )?;
        }
        backend.copy(backend.view(image), output)
    }
}

/// Zero-sized deflated-CG marker: two-level CG over a deflation subspace.
///
/// The solve is two phases on one driver: the right-hand side's complement
/// projection `(I − UUᵀ)·b` drives CG through the deflated operator
/// `(I − UUᵀ)·A·(I − UUᵀ)` — symmetric for any orthonormal basis, positive
/// definite on the complement — and the coarse correction
/// `U·(Uᵀ A U)⁻¹·Uᵀ·b` resolves the deflated directions exactly, so the
/// combined iterate is the solution of `A·x = b`.
///
/// With `U` an exact invariant subspace of `k` eigenvectors of an SPD
/// operator, the CG phase terminates in at most `n − k` iterations: the
/// iterated space never contains the deflated directions, and the complement
/// spectrum has `n − k` values. The small-eigenvalue directions that make
/// plain CG stall are exactly the ones the coarse phase carries — the
/// property the deflation contract pins as its finite-termination oracle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeflatedCg<B>(core::marker::PhantomData<fn() -> B>);

impl<B: KrylovBackend> DeflatedCg<B> {
    /// Solve `A x = b` in two levels over `subspace`, into caller-owned
    /// `solution`.
    ///
    /// # Errors
    ///
    /// Returns the dimension validation, the CG phase, or the coarse
    /// correction's backend failures. Numerical termination of the CG phase
    /// is returned value-semantically in [`SolveReport`].
    pub fn solve_into<O>(
        backend: &B,
        operator: &O,
        subspace: &DeflationSubspace<B>,
        right_hand_side: &B::Vector,
        solution: &mut B::Vector,
        workspace: &mut CgWorkspace<B>,
        policy: ConvergencePolicy<B::Scalar>,
    ) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>>
    where
        O: LinearOperator<B>,
    {
        let dimension = operator.dimension();
        validate_dimensions(&[
            (
                "right-hand side",
                dimension,
                backend.vector_len(right_hand_side),
            ),
            ("solution", dimension, backend.vector_len(solution)),
            ("CG workspace", dimension, workspace.len()),
        ])?;

        // Phase 1: the complement solve. The right-hand side's deflated
        // directions are removed before the iteration starts, so the CG
        // iterates stay in the complement where the deflated operator is
        // positive definite.
        let deflated_operator =
            DeflatedOperator::new(backend, operator, subspace).map_err(SolveError::Backend)?;
        let mut complement = backend.allocate(dimension).map_err(SolveError::Backend)?;
        subspace
            .project(backend, right_hand_side, &mut complement)
            .map_err(SolveError::Backend)?;
        let mut phase = backend.allocate(dimension).map_err(SolveError::Backend)?;
        let report = Cg::<B>::solve_into(
            backend,
            &deflated_operator,
            &Identity,
            &complement,
            &mut phase,
            workspace,
            policy,
        )?;

        // Phase 2: the coarse correction carries the deflated directions
        // exactly, and the solution assembles the two levels.
        subspace
            .coarse_correction(backend, right_hand_side, solution)
            .map_err(SolveError::Backend)?;
        backend
            .axpy(
                backend.view_mut(solution),
                backend.view(&phase),
                <B::Scalar as eunomia::NumericElement>::ONE,
            )
            .map_err(SolveError::Backend)?;
        Ok(report)
    }
}
