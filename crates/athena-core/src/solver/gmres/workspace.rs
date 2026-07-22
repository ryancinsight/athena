use alloc::vec::Vec;
use eunomia::NumericElement;

use crate::KrylovBackend;

/// Reusable restarted-GMRES vector and scalar workspace.
///
/// Construction performs every host and backend allocation and prepares the
/// reductions bound to the workspace vectors. A workspace can serve repeated
/// solves with the same vector length and restart width without allocating or
/// rebuilding accelerator dispatch resources.
pub struct GmresWorkspace<B: KrylovBackend, const RESTART: usize> {
    pub(super) residual: B::Vector,
    pub(super) work: B::Vector,
    pub(super) basis: Vec<B::Vector>,
    pub(super) preconditioned_basis: Vec<B::Vector>,
    pub(super) residual_norm: B::PreparedNorm,
    pub(super) work_norm: B::PreparedNorm,
    pub(super) work_basis_dot: Vec<B::PreparedDot>,
    pub(super) hessenberg: Vec<B::Scalar>,
    pub(super) cosine: Vec<B::Scalar>,
    pub(super) sine: Vec<B::Scalar>,
    pub(super) transformed_residual: Vec<B::Scalar>,
    pub(super) coefficients: Vec<B::Scalar>,
    len: usize,
}

impl<B: KrylovBackend, const RESTART: usize> GmresWorkspace<B, RESTART> {
    const VALID_RESTART: () = assert!(
        RESTART > 0 && RESTART < usize::MAX,
        "GMRES restart width must be in 1..usize::MAX"
    );

    /// Allocate a workspace for `len` unknowns.
    ///
    /// # Errors
    ///
    /// Returns the first backend allocation or reduction-preparation failure.
    ///
    /// # Panics
    ///
    /// Panics during monomorphization when `RESTART` is zero, or when its
    /// scalar workspace size cannot fit in `usize`.
    pub fn new(backend: &B, len: usize) -> Result<Self, B::Error> {
        let () = Self::VALID_RESTART;
        let basis_len = RESTART
            .checked_add(1)
            .expect("invariant: restart width excludes usize::MAX");
        let hessenberg_len = basis_len
            .checked_mul(RESTART)
            .expect("invariant: GMRES scalar workspace size fits usize");

        let mut basis = Vec::with_capacity(basis_len);
        for _ in 0..basis_len {
            basis.push(backend.allocate(len)?);
        }
        let mut preconditioned_basis = Vec::with_capacity(RESTART);
        for _ in 0..RESTART {
            preconditioned_basis.push(backend.allocate(len)?);
        }
        let residual = backend.allocate(len)?;
        let work = backend.allocate(len)?;
        let residual_norm = backend.prepare_norm_l2(backend.view(&residual))?;
        let work_norm = backend.prepare_norm_l2(backend.view(&work))?;
        let mut work_basis_dot = Vec::with_capacity(basis_len);
        for basis_vector in &basis {
            work_basis_dot
                .push(backend.prepare_dot(backend.view(&work), backend.view(basis_vector))?);
        }

        Ok(Self {
            residual,
            work,
            basis,
            preconditioned_basis,
            residual_norm,
            work_norm,
            work_basis_dot,
            hessenberg: alloc::vec![B::Scalar::ZERO; hessenberg_len],
            cosine: alloc::vec![B::Scalar::ZERO; RESTART],
            sine: alloc::vec![B::Scalar::ZERO; RESTART],
            transformed_residual: alloc::vec![B::Scalar::ZERO; basis_len],
            coefficients: alloc::vec![B::Scalar::ZERO; RESTART],
            len,
        })
    }

    /// Workspace vector length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Compile-time restart width.
    #[must_use]
    pub const fn restart_width(&self) -> usize {
        RESTART
    }

    /// Return whether this is a zero-length vector workspace.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(super) fn reset_cycle(&mut self) {
        self.hessenberg.fill(B::Scalar::ZERO);
        self.cosine.fill(B::Scalar::ZERO);
        self.sine.fill(B::Scalar::ZERO);
        self.transformed_residual.fill(B::Scalar::ZERO);
        self.coefficients.fill(B::Scalar::ZERO);
    }

    pub(super) const fn hessenberg_index(row: usize, column: usize) -> usize {
        column * (RESTART + 1) + row
    }
}
