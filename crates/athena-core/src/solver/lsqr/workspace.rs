use crate::KrylovBackend;

/// Reusable LSQR vector workspace.
///
/// LSQR bidiagonalises a rectangular operator, so its vectors come in two
/// lengths: `rows` for the range of `A` and `columns` for its domain. The
/// workspace carries both, and construction performs every allocation and
/// prepares the reductions bound to those vectors, keeping the recurrence
/// allocation-free across solves.
pub struct LsqrWorkspace<B: KrylovBackend> {
    /// Left bidiagonalisation vector, length `rows`.
    pub(super) left: B::Vector,
    /// Right bidiagonalisation vector, length `columns`.
    pub(super) right: B::Vector,
    /// Search direction, length `columns`.
    pub(super) direction: B::Vector,
    /// Operator image `A·v`, length `rows`.
    pub(super) image: B::Vector,
    /// Adjoint image `Aᵀ·u`, length `columns`.
    pub(super) adjoint_image: B::Vector,
    pub(super) left_norm: B::PreparedNorm,
    pub(super) right_norm: B::PreparedNorm,
    rows: usize,
    columns: usize,
}

impl<B: KrylovBackend> LsqrWorkspace<B> {
    /// Allocate a workspace for a `rows × columns` operator.
    ///
    /// # Errors
    ///
    /// Returns the first backend allocation or reduction-preparation failure.
    pub fn new(backend: &B, rows: usize, columns: usize) -> Result<Self, B::Error> {
        let left = backend.allocate(rows)?;
        let right = backend.allocate(columns)?;
        let direction = backend.allocate(columns)?;
        let image = backend.allocate(rows)?;
        let adjoint_image = backend.allocate(columns)?;
        let left_norm = backend.prepare_norm_l2(backend.view(&left))?;
        let right_norm = backend.prepare_norm_l2(backend.view(&right))?;
        Ok(Self {
            left,
            right,
            direction,
            image,
            adjoint_image,
            left_norm,
            right_norm,
            rows,
            columns,
        })
    }

    /// Row count this workspace was allocated for.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Column count this workspace was allocated for.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }
}
