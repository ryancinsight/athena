//! Preconditioners for the accelerator backend.
//!
//! # What this backend provides
//!
//! [`Jacobi`] — the diagonal preconditioner — and nothing else. Its application
//! is one elementwise product over device-resident vectors, which is the shape
//! an accelerator executes at full bandwidth, so the whole preconditioned
//! recurrence stays on the device.
//!
//! # Why the triangular family is absent
//!
//! The CPU backend additionally ships incomplete LU, a triangular solve, and
//! successive over-relaxation. All three apply by forward and backward
//! substitution: element `i` of the result depends on elements `0..i`, so the
//! sweep is inherently sequential in the matrix dimension. Dispatching that
//! recurrence per element would run one device thread against a launch per row,
//! which is slower than the unpreconditioned iteration it is meant to
//! accelerate. Shipping the naive port would therefore hand callers a
//! preconditioner that costs more than it saves. Admitting the family here
//! requires a parallel formulation — level-scheduled substitution or an
//! iterative approximate triangular solve — which is a distinct design with its
//! own convergence contract, not a port of the CPU code.

mod jacobi;

pub use jacobi::{Jacobi, inverse_diagonal_from_csr};
