//! Schur RHS construction, coarse recovery, and true-residual reporting.

use super::algorithm::DeflatedOperator;
use crate::{
    Cg, CgWorkspace, ConvergencePolicy, DeflationSubspace, Identity, KrylovBackend, LinearOperator,
    SolveError, SolveReport, Termination,
};
use eunomia::NumericElement;

pub(super) struct Execution<'a, B: KrylovBackend, O> {
    pub(super) backend: &'a B,
    pub(super) operator: &'a O,
    pub(super) subspace: &'a DeflationSubspace<B>,
    pub(super) right_hand_side: &'a B::Vector,
    pub(super) solution: &'a mut B::Vector,
    pub(super) workspace: &'a mut CgWorkspace<B>,
    pub(super) policy: ConvergencePolicy<B::Scalar>,
}

impl<B: KrylovBackend, O: LinearOperator<B>> Execution<'_, B, O> {
    pub(super) fn solve(mut self) -> Result<SolveReport<B::Scalar>, SolveError<B::Error>> {
        let dimension = self.operator.dimension();
        let mut image = self
            .backend
            .allocate(dimension)
            .map_err(SolveError::Backend)?;
        let mut residual = self
            .backend
            .allocate(dimension)
            .map_err(SolveError::Backend)?;
        let initial = self.true_residual(&mut image, &mut residual)?;
        let rhs_norm = self
            .backend
            .norm_l2(self.backend.view(self.right_hand_side))
            .map_err(SolveError::Backend)?;
        let threshold = self.policy.threshold(rhs_norm);
        if !initial.is_finite() || !threshold.is_finite() || initial <= threshold {
            let termination = if initial.is_finite() && threshold.is_finite() {
                Termination::InitialResidual
            } else {
                Termination::NonFinite
            };
            return Ok(SolveReport::new(
                termination,
                0,
                1,
                0,
                initial,
                initial,
                threshold,
            ));
        }

        let deflated = DeflatedOperator::new(self.backend, self.operator, self.subspace)
            .map_err(SolveError::Backend)?;
        let mut complement = self
            .backend
            .allocate(dimension)
            .map_err(SolveError::Backend)?;
        self.project_rhs(&residual, &mut image, &mut complement)?;
        let mut phase = self
            .backend
            .allocate(dimension)
            .map_err(SolveError::Backend)?;
        // The transformed RHS can have a very different norm. Preserve the
        // ORIGINAL RHS threshold instead of redefining relative tolerance.
        let report = Cg::<B>::solve_into(
            self.backend,
            &deflated,
            &Identity,
            &complement,
            &mut phase,
            self.workspace,
            self.policy.with_absolute_threshold(threshold),
        )?;
        self.recover(&phase, &residual, &mut image, &mut complement)?;
        let final_residual = self.true_residual(&mut image, &mut residual)?;
        let termination = if !final_residual.is_finite() {
            Termination::NonFinite
        } else if final_residual <= threshold {
            Termination::Converged
        } else if report.converged() {
            // Recovery missed the true-residual policy despite a converged
            // projected recurrence. Never report false convergence.
            if report.iterations == self.policy.max_iterations() {
                Termination::MaxIterations
            } else {
                Termination::Stagnated
            }
        } else {
            report.termination
        };
        Ok(SolveReport::new(
            termination,
            report.iterations,
            report.operator_applications + 3,
            report.preconditioner_applications,
            initial,
            final_residual,
            threshold,
        ))
    }

    fn true_residual(
        &self,
        image: &mut B::Vector,
        residual: &mut B::Vector,
    ) -> Result<B::Scalar, SolveError<B::Error>> {
        self.operator
            .apply(
                self.backend,
                self.backend.view(self.solution),
                self.backend.view_mut(image),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .residual(
                self.backend.view(self.right_hand_side),
                self.backend.view(image),
                self.backend.view_mut(residual),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .norm_l2(self.backend.view(residual))
            .map_err(SolveError::Backend)
    }

    fn project_rhs(
        &self,
        residual: &B::Vector,
        image: &mut B::Vector,
        complement: &mut B::Vector,
    ) -> Result<(), SolveError<B::Error>> {
        // g = (I - A C) r0. Q b alone discards coarse/complement coupling.
        self.subspace
            .coarse_image(self.backend, residual, image)
            .map_err(SolveError::Backend)?;
        self.backend
            .residual(
                self.backend.view(residual),
                self.backend.view(image),
                self.backend.view_mut(complement),
            )
            .map_err(SolveError::Backend)?;
        // Q removes roundoff components outside the complement, where the
        // Schur operator is positive definite.
        self.subspace
            .project(self.backend, complement, image)
            .map_err(SolveError::Backend)?;
        core::mem::swap(complement, image);
        Ok(())
    }

    fn recover(
        &mut self,
        phase: &B::Vector,
        residual: &B::Vector,
        image: &mut B::Vector,
        complement: &mut B::Vector,
    ) -> Result<(), SolveError<B::Error>> {
        // x = x0 + y + C(r0 - A y): recover eliminated coarse coordinates.
        self.operator
            .apply(
                self.backend,
                self.backend.view(phase),
                self.backend.view_mut(image),
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .residual(
                self.backend.view(residual),
                self.backend.view(image),
                self.backend.view_mut(complement),
            )
            .map_err(SolveError::Backend)?;
        self.subspace
            .coarse_correction(self.backend, complement, image)
            .map_err(SolveError::Backend)?;
        self.backend
            .axpy(
                self.backend.view_mut(self.solution),
                self.backend.view(phase),
                B::Scalar::ONE,
            )
            .map_err(SolveError::Backend)?;
        self.backend
            .axpy(
                self.backend.view_mut(self.solution),
                self.backend.view(image),
                B::Scalar::ONE,
            )
            .map_err(SolveError::Backend)
    }
}
