use core::cell::RefCell;
use eunomia::NumericElement;

use crate::{
    CgWorkspace, ConvergencePolicy, DeflationSubspace, KrylovBackend, LinearOperator, SolveError,
    SolveReport,
};

use super::super::dimension::validate_dimensions;

/// The deflated Schur operator `Q·(A − A C A)·Q`, where
/// `Q = I − UUᵀ` and `C = U·(Uᵀ A U)⁻¹·Uᵀ`.
///
/// For SPD A this is symmetric and positive definite on U's complement:
/// eliminating the coarse coordinates gives exactly this Schur complement.
/// For nonzero `y` with `Uᵀ y = 0`, put `z = y − C A y`. Then
/// `yᵀ(A − A C A)y = zᵀ A z > 0`: `Q z = y` ensures `z ≠ 0`.
/// It annihilates U, including when U is not invariant under A. Invariance
/// reduces it to Q A Q but is not required for correctness.
///
/// Cached A U images keep each apply to one base operator application.
/// Internal scratch is borrowed per call, making this view `!Sync`.
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
        // image <- (A - A C A) Q v. The Schur correction removes
        // coupling to the coarse coordinates before the complement solve.
        self.operator
            .apply(backend, backend.view(projected), backend.view_mut(image))?;
        self.subspace.coarse_image(backend, image, projected)?;
        backend.axpy(
            backend.view_mut(image),
            backend.view(projected),
            -B::Scalar::ONE,
        )?;
        // Reproject roundoff out of U's directions.
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
/// With `C = U·(Uᵀ A U)⁻¹·Uᵀ` and `r₀ = b − A x₀`, CG solves
/// `(A − A C A) y = (I − A C) r₀` in U's Euclidean complement.
/// The recovered iterate is `x = x₀ + y + C(r₀ − A y)`. Substitution gives
/// `b − A x = (I − A C) r₀ − (A − A C A) y`, the Schur residual.
/// Thus the combined solve is valid for any supported basis, including
/// noninvariant ones. The final report checks `b − A x` explicitly against
/// the original RHS policy; the incoming solution is the initial guess.
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
            ("deflation subspace", dimension, subspace.dimension()),
        ])?;

        super::execution::Execution {
            backend,
            operator,
            subspace,
            right_hand_side,
            solution,
            workspace,
            policy,
        }
        .solve()
    }
}
