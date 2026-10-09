//! The AMG preconditioner state: sparse helpers, the hierarchy owner, and
//! its constructors.

use super::super::{AMGConfig, AMGHierarchy, AMGStatistics, MultigridLevel};
use crate::LetoBackendError;
use eunomia::RealField;
use leto::Array1;
use leto_ops::{CsrMatrix, RealScalar, spgemm, spmv_into as leto_spmv_into};
use std::sync::{Arc, Mutex};

use super::cycle::AMGLevelWorkspace;

pub(super) fn sparse_product<T: RealField + Copy + RealScalar>(
    lhs: &CsrMatrix<T>,
    rhs: &CsrMatrix<T>,
) -> Result<CsrMatrix<T>, LetoBackendError> {
    spgemm(lhs, rhs).map_err(|error| {
        LetoBackendError::Leto(leto::LetoError::InvalidInput(format!(
            "AMG sparse product failed: {error}"
        )))
    })
}

pub(super) fn sparse_apply_into<T: RealField + Copy + RealScalar>(
    matrix: &CsrMatrix<T>,
    vector: &Array1<T>,
    output: &mut Array1<T>,
) -> Result<(), LetoBackendError> {
    let output_slice = output
        .as_slice_mut()
        .ok_or(LetoBackendError::NonContiguousVector)?;
    leto_spmv_into(matrix, &vector.view(), output_slice).map_err(LetoBackendError::from)
}

type AthenaBoundaryBuffers<T> = Arc<Mutex<Option<(Array1<T>, Array1<T>)>>>;

/// Algebraic Multigrid preconditioner.
///
/// Builds a hierarchy of coarsened systems with transfer operators and
/// smoothers, then applies V-cycles as the preconditioning action. The
/// constructor assembles the hierarchy once; [`Self::apply_to`] and the
/// Athena seam below share one cycle body.
#[derive(Clone)]
pub struct AlgebraicMultigrid<T: RealField + Copy + RealScalar> {
    /// Multigrid hierarchy levels
    pub(super) levels: Vec<MultigridLevel<T>>,
    /// AMG configuration
    pub(super) config: AMGConfig,
    /// Performance statistics
    pub(super) statistics: AMGStatistics,
    /// Setup completion flag
    pub(super) is_setup: bool,
    /// Optional cached hierarchy operators
    pub(super) hierarchy: Option<AMGHierarchy<T>>,
    /// Reusable V-cycle vectors, serialized per preconditioner application.
    pub(super) workspace: Arc<Mutex<Option<Vec<AMGLevelWorkspace<T>>>>>,
    /// Owned buffers backing the Athena preconditioner boundary.
    ///
    /// The V-cycle recurses over owned `Array1` vectors while Athena hands
    /// the preconditioner borrowed views, so the boundary copies in and out.
    /// The two `O(n)` passes are small beside the cycle's `O(nnz)` sweeps,
    /// and caching the buffers keeps a preconditioner application
    /// allocation-free.
    pub(super) athena_boundary: AthenaBoundaryBuffers<T>,
}

impl<T: RealField + Copy + eunomia::FloatElement + RealScalar> AlgebraicMultigrid<T> {
    /// Create a new AMG preconditioner.
    ///
    /// # Errors
    ///
    /// Returns the coarsening, interpolation, or sparse-product failure of
    /// the hierarchy assembly.
    pub fn new(matrix: &CsrMatrix<T>, config: AMGConfig) -> Result<Self, LetoBackendError> {
        let mut amg = Self {
            levels: Vec::new(),
            config,
            statistics: AMGStatistics::default(),
            is_setup: false,
            hierarchy: None,
            workspace: Arc::new(Mutex::new(None)),
            athena_boundary: Arc::new(Mutex::new(None)),
        };

        amg.setup(matrix)?;
        Ok(amg)
    }

    /// Create a new AMG preconditioner with a specified configuration.
    ///
    /// # Errors
    ///
    /// See [`Self::new`].
    pub fn with_config(matrix: &CsrMatrix<T>, config: AMGConfig) -> Result<Self, LetoBackendError> {
        Self::new(matrix, config)
    }

    /// Create a new AMG preconditioner with a pre-existing hierarchy.
    ///
    /// # Errors
    ///
    /// See [`Self::new`].
    pub fn with_hierarchy(
        matrix: &CsrMatrix<T>,
        config: AMGConfig,
        hierarchy: AMGHierarchy<T>,
    ) -> Result<Self, LetoBackendError> {
        let mut amg = Self {
            levels: Vec::new(),
            config,
            statistics: AMGStatistics::default(),
            is_setup: false,
            hierarchy: Some(hierarchy),
            workspace: Arc::new(Mutex::new(None)),
            athena_boundary: Arc::new(Mutex::new(None)),
        };

        amg.setup(matrix)?;
        Ok(amg)
    }

    /// Get the current hierarchy for caching.
    #[must_use]
    pub fn get_hierarchy(&self) -> AMGHierarchy<T> {
        AMGHierarchy::from_levels(&self.levels)
    }

    /// The configured strategy vocabulary.
    #[must_use]
    pub const fn config(&self) -> &AMGConfig {
        &self.config
    }

    /// The recorded hierarchy statistics.
    #[must_use]
    pub const fn statistics(&self) -> &AMGStatistics {
        &self.statistics
    }

    /// Recompute the hierarchy operators for a new matrix with the same
    /// sparsity pattern.
    ///
    /// # Errors
    ///
    /// Returns [`leto::LetoError::InvalidInput`] when the hierarchy was never
    /// assembled or a level lacks its transfer operators, and the
    /// sparse-product failure otherwise.
    pub fn recompute(&mut self, fine_matrix: &CsrMatrix<T>) -> Result<(), LetoBackendError> {
        if self.levels.is_empty() {
            return Err(LetoBackendError::Leto(leto::LetoError::InvalidInput(
                "AMG hierarchy not initialized".to_string(),
            )));
        }

        // Update the finest level matrix
        self.levels[0].matrix = fine_matrix.clone();
        self.levels[0].smoother = self.create_smoother(fine_matrix);

        // Recompute coarse matrices using existing transfer operators
        for i in 0..self.levels.len() - 1 {
            let (restriction, interpolation) = {
                let current = &self.levels[i];
                match (&current.restriction, &current.interpolation) {
                    (Some(r), Some(p)) => (r, p),
                    _ => {
                        return Err(LetoBackendError::Leto(leto::LetoError::InvalidInput(
                            "Missing transfer operators".to_string(),
                        )));
                    }
                }
            };

            // A_coarse = R * A_fine * P
            let temp = sparse_product(restriction, &self.levels[i].matrix)?;
            let coarse_matrix = sparse_product(&temp, interpolation)?;

            // Update next level
            self.levels[i + 1].matrix = coarse_matrix.clone();
            self.levels[i + 1].smoother = self.create_smoother(&coarse_matrix);
        }

        Ok(())
    }
}
