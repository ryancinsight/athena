//! Element access for dense and strided multigrid operands.

use std::ops::Index;

use leto::{Array1, ArrayView1, ArrayViewMut1, Layout};
use leto_ops::{CsrMatrix, RealScalar, spmv_into};

pub(super) trait Vector<T>: Index<usize, Output = T> {
    fn len(&self) -> usize;
    fn dense(&self) -> Option<&[T]>;

    fn copy_into(&self, target: &mut [T])
    where
        T: Copy,
    {
        if let Some(slice) = self.dense() {
            target.copy_from_slice(slice);
        } else {
            for (index, entry) in target.iter_mut().enumerate() {
                *entry = self[index];
            }
        }
    }
}

impl<T> Vector<T> for [T] {
    fn len(&self) -> usize {
        self.len()
    }

    fn dense(&self) -> Option<&[T]> {
        Some(self)
    }
}

impl<T> Vector<T> for Array1<T> {
    fn len(&self) -> usize {
        self.shape()[0]
    }

    fn dense(&self) -> Option<&[T]> {
        self.as_slice()
    }
}

impl<T> Vector<T> for ArrayViewMut1<'_, T> {
    fn len(&self) -> usize {
        self.shape()[0]
    }

    fn dense(&self) -> Option<&[T]> {
        self.as_slice()
    }
}

pub(super) fn sparse_apply_into<T: RealScalar + Copy>(
    matrix: &CsrMatrix<T>,
    input: &(impl Vector<T> + ?Sized),
    output: &mut [T],
) -> leto::Result<()> {
    // Leto's CSR gather consumes dense storage. Pack only the strided
    // SpMV operand, preserving Leto's row reduction and interleaved views.
    let packed;
    let values = if let Some(values) = input.dense() {
        values
    } else {
        packed = (0..input.len())
            .map(|index| input[index])
            .collect::<Vec<_>>();
        &packed
    };
    let layout = Layout::c_contiguous([values.len()])?;
    spmv_into(matrix, &ArrayView1::new(layout, values), output)
}
