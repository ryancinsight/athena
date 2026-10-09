//! Hierarchy assembly: coarsening, transfer construction, and smoothing setup.

use super::super::scalars::{count_to_scalar, diagonal_epsilon};
use crate::LetoBackendError;
use eunomia::{FloatElement, NumericElement, RealField};
use leto_ops::{CsrMatrix, RealScalar};

use super::super::{
    ChebyshevSmoother, CoarseningStrategy, GaussSeidelSmoother, InterpolationStrategy,
    JacobiSmoother, MultigridLevel, SORSmoother, Smoother, SmootherType,
    SymmetricGaussSeidelSmoother,
};
use super::state::AlgebraicMultigrid;
use super::state::sparse_product;

impl<T: RealField + Copy + FloatElement + RealScalar> AlgebraicMultigrid<T> {
    pub(super) fn setup(&mut self, fine_matrix: &CsrMatrix<T>) -> Result<(), LetoBackendError> {
        let setup_start = std::time::Instant::now();

        // Create finest level
        let finest_level = MultigridLevel {
            matrix: fine_matrix.clone(),
            restriction: None,
            interpolation: None,
            smoother: self.create_smoother(fine_matrix),
        };

        self.levels.push(finest_level);
        self.statistics.level_sizes.push(fine_matrix.nrows());

        // Build hierarchy
        if let Some(ref hierarchy) = self.hierarchy {
            // Use cached operators to build coarse matrices
            for (restriction, interpolation) in &hierarchy.operators {
                let coarse_matrix = {
                    let current_level = self
                        .levels
                        .last_mut()
                        .expect("invariant: cached hierarchy starts from the finest level");
                    let temp_matrix = sparse_product(restriction, &current_level.matrix)?;
                    let coarse_matrix = sparse_product(&temp_matrix, interpolation)?;

                    // Store operators in the finer level
                    current_level.restriction = Some(restriction.clone());
                    current_level.interpolation = Some(interpolation.clone());

                    coarse_matrix
                };

                let smoother = self.create_smoother(&coarse_matrix);
                self.levels.push(MultigridLevel {
                    matrix: coarse_matrix,
                    restriction: None,
                    interpolation: None,
                    smoother,
                });
                self.statistics.level_sizes.push(
                    self.levels
                        .last()
                        .expect("invariant: cached hierarchy level was just inserted")
                        .matrix
                        .nrows(),
                );
            }
        } else {
            // Standard build
            while self.levels.len() < self.config.max_levels
                && self
                    .levels
                    .last()
                    .expect("invariant: AMG setup created the finest level")
                    .matrix
                    .nrows()
                    > self.config.min_coarse_size
            {
                self.build_next_level()?;
            }
        }

        // Record statistics
        self.statistics.num_levels = self.levels.len();
        self.statistics.setup_time = setup_start.elapsed().as_secs_f64();
        self.compute_complexities();

        self.is_setup = true;
        Ok(())
    }

    /// Build the next coarser level in the hierarchy.
    fn build_next_level(&mut self) -> Result<(), LetoBackendError> {
        use super::super::coarsening::{
            aggregation_coarsening, falgout_coarsening, hmis_coarsening, hybrid_coarsening,
            pmis_coarsening, ruge_stueben_coarsening,
        };
        use super::super::interpolation::{
            create_classical_interpolation, create_direct_interpolation,
            create_standard_interpolation,
        };

        let (restriction, interpolation, coarse_matrix) = {
            let current_level = self
                .levels
                .last()
                .expect("invariant: AMG hierarchy has a current level");

            let coarsening_result = match self.config.coarsening_strategy {
                CoarseningStrategy::RugeStueben => ruge_stueben_coarsening(
                    &current_level.matrix,
                    <T as FloatElement>::from_f64(self.config.strength_threshold),
                ),
                CoarseningStrategy::Aggregation => aggregation_coarsening(&current_level.matrix, 8),
                CoarseningStrategy::Hybrid => hybrid_coarsening(
                    &current_level.matrix,
                    <T as FloatElement>::from_f64(self.config.strength_threshold),
                    4,
                ),
                CoarseningStrategy::Falgout => falgout_coarsening(
                    &current_level.matrix,
                    <T as FloatElement>::from_f64(self.config.strength_threshold),
                ),
                CoarseningStrategy::PMIS => pmis_coarsening(
                    &current_level.matrix,
                    <T as FloatElement>::from_f64(self.config.strength_threshold),
                ),
                CoarseningStrategy::HMIS => hmis_coarsening(
                    &current_level.matrix,
                    <T as FloatElement>::from_f64(self.config.strength_threshold),
                    <T as FloatElement>::from_f64(0.5),
                ),
            }
            .map_err(|e| {
                LetoBackendError::Leto(leto::LetoError::InvalidInput(format!(
                    "AMG coarsening failed: {e}"
                )))
            })?;

            let interpolation = match self.config.interpolation_strategy {
                InterpolationStrategy::Classical => create_classical_interpolation(
                    &current_level.matrix,
                    &coarsening_result.coarse_points,
                    &coarsening_result.strength_matrix,
                    self.config.max_interpolation_points,
                ),
                InterpolationStrategy::Direct => Ok(create_direct_interpolation(
                    &coarsening_result.fine_to_coarse_map,
                    current_level.matrix.nrows(),
                    coarsening_result.coarse_points.len(),
                )),
                InterpolationStrategy::Standard => create_standard_interpolation(
                    &current_level.matrix,
                    &coarsening_result.coarse_points,
                    &coarsening_result.strength_matrix,
                ),
            }
            .map_err(|e| {
                LetoBackendError::Leto(leto::LetoError::InvalidInput(format!(
                    "AMG interpolation failed: {e}"
                )))
            })?;

            // Create restriction operator (transpose of interpolation)
            let restriction = interpolation.transpose();

            // Create coarse matrix: R * A_fine * P
            let temp_matrix = sparse_product(&restriction, &current_level.matrix)?;
            let coarse_matrix = sparse_product(&temp_matrix, &interpolation)?;

            (restriction, interpolation, coarse_matrix)
        };

        // Update the current level with its operators to the next level
        let last_idx = self.levels.len() - 1;
        self.levels[last_idx].restriction = Some(restriction);
        self.levels[last_idx].interpolation = Some(interpolation);

        // Create smoother for this level
        let smoother = self.create_smoother(&coarse_matrix);

        // Add new level (coarsest level starts with no restriction/interpolation)
        let level_size = coarse_matrix.nrows();
        let new_level = MultigridLevel {
            matrix: coarse_matrix,
            restriction: None,
            interpolation: None,
            smoother,
        };

        self.levels.push(new_level);
        self.statistics.level_sizes.push(level_size);

        Ok(())
    }

    /// Create the level smoother for a given matrix.
    ///
    /// The match is exhaustive over [`SmootherType`]: every configured
    /// smoother builds its own concrete instance, so no configuration can
    /// silently degrade to a different smoother. Chebyshev bounds come from
    /// [`ChebyshevSmoother::estimate_eigenvalues`] on the level matrix.
    pub(super) fn create_smoother(&self, matrix: &CsrMatrix<T>) -> Smoother<T> {
        let relaxation = <T as FloatElement>::from_f64(self.config.relaxation_factor);
        match self.config.smoother_type {
            SmootherType::GaussSeidel => {
                Smoother::GaussSeidel(GaussSeidelSmoother::new(relaxation))
            }
            SmootherType::SymmetricGaussSeidel => {
                Smoother::SymmetricGaussSeidel(SymmetricGaussSeidelSmoother::new(relaxation))
            }
            SmootherType::Jacobi => Smoother::Jacobi(JacobiSmoother::new(relaxation)),
            SmootherType::SOR => Smoother::Sor(SORSmoother::new(relaxation)),
            SmootherType::Chebyshev => {
                let (eigenvalues_min, eigenvalues_max) =
                    ChebyshevSmoother::estimate_eigenvalues(matrix);
                Smoother::Chebyshev(ChebyshevSmoother::new(
                    eigenvalues_min,
                    eigenvalues_max,
                    self.config.chebyshev_degree,
                ))
            }
        }
    }

    /// Compute operator and grid complexities.
    fn compute_complexities(&mut self) {
        if self.levels.is_empty() {
            return;
        }

        let fine_nnz = self.levels[0]
            .matrix
            .values()
            .iter()
            .filter(|&&x| NumericElement::abs(x) > diagonal_epsilon())
            .count();
        let mut total_nnz = 0;
        let mut total_vars = 0;

        for level in &self.levels {
            total_nnz += level
                .matrix
                .values()
                .iter()
                .filter(|&&x| NumericElement::abs(x) > diagonal_epsilon())
                .count();
            total_vars += level.matrix.nrows();
        }

        self.statistics.operator_complexity =
            count_to_scalar::<f64>(total_nnz) / count_to_scalar::<f64>(fine_nnz);
        self.statistics.grid_complexity = count_to_scalar::<f64>(total_vars)
            / count_to_scalar::<f64>(self.levels[0].matrix.nrows());
    }
}
