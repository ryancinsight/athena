use core::marker::PhantomData;

use athena_core::KrylovBackend;
use eunomia::RealField;
use leto::{Array1, ArrayView1, ArrayViewMut1};
use leto_ops::{RealScalar, dot};

use crate::LetoBackendError;

/// Zero-sized Leto CPU backend marker.
///
/// Every operation monomorphizes over `T`; the marker carries no runtime state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LetoBackend<T>(PhantomData<fn() -> T>);

impl<T: RealScalar + RealField> KrylovBackend for LetoBackend<T> {
    type Scalar = T;
    type Error = LetoBackendError;
    type Vector = Array1<T>;
    type View<'a>
        = ArrayView1<'a, T>
    where
        Self: 'a;
    type ViewMut<'a>
        = ArrayViewMut1<'a, T>
    where
        Self: 'a;

    #[inline]
    fn allocate(&self, len: usize) -> Result<Self::Vector, Self::Error> {
        Ok(Array1::zeros([len]))
    }

    #[inline]
    fn view<'a>(&'a self, vector: &'a Self::Vector) -> Self::View<'a> {
        vector.view()
    }

    #[inline]
    fn view_mut<'a>(&'a self, vector: &'a mut Self::Vector) -> Self::ViewMut<'a> {
        vector.view_mut()
    }

    #[inline]
    fn vector_len(&self, vector: &Self::Vector) -> usize {
        vector.len()
    }

    fn copy(
        &self,
        source: Self::View<'_>,
        mut target: Self::ViewMut<'_>,
    ) -> Result<(), Self::Error> {
        let source = contiguous(&source)?;
        let target = contiguous_mut(&mut target)?;
        validate_lengths(source.len(), target.len())?;
        target.copy_from_slice(source);
        Ok(())
    }

    fn scale(
        &self,
        mut target: Self::ViewMut<'_>,
        factor: Self::Scalar,
    ) -> Result<(), Self::Error> {
        for value in contiguous_mut(&mut target)? {
            *value *= factor;
        }
        Ok(())
    }

    fn axpy(
        &self,
        mut target: Self::ViewMut<'_>,
        source: Self::View<'_>,
        factor: Self::Scalar,
    ) -> Result<(), Self::Error> {
        let target = contiguous_mut(&mut target)?;
        let source = contiguous(&source)?;
        validate_lengths(target.len(), source.len())?;
        for (target_value, &source_value) in target.iter_mut().zip(source.iter()) {
            *target_value += factor * source_value;
        }
        Ok(())
    }

    #[inline]
    fn dot(
        &self,
        left: Self::View<'_>,
        right: Self::View<'_>,
    ) -> Result<Self::Scalar, Self::Error> {
        dot(&left, &right).map_err(Into::into)
    }

    #[inline]
    fn norm_l2(&self, vector: Self::View<'_>) -> Result<Self::Scalar, Self::Error> {
        Ok(dot(&vector, &vector)?.sqrt())
    }

    fn residual(
        &self,
        right_hand_side: Self::View<'_>,
        image: Self::View<'_>,
        mut residual: Self::ViewMut<'_>,
    ) -> Result<(), Self::Error> {
        let right_hand_side = contiguous(&right_hand_side)?;
        let image = contiguous(&image)?;
        let residual = contiguous_mut(&mut residual)?;
        validate_three_lengths(right_hand_side.len(), image.len(), residual.len())?;
        for ((out, &right), &mapped) in residual
            .iter_mut()
            .zip(right_hand_side.iter())
            .zip(image.iter())
        {
            *out = right - mapped;
        }
        Ok(())
    }

    fn fused_cg_update(
        &self,
        mut solution: Self::ViewMut<'_>,
        direction: Self::View<'_>,
        mut residual: Self::ViewMut<'_>,
        image: Self::View<'_>,
        alpha: Self::Scalar,
    ) -> Result<(), Self::Error> {
        let solution = contiguous_mut(&mut solution)?;
        let direction = contiguous(&direction)?;
        let residual = contiguous_mut(&mut residual)?;
        let image = contiguous(&image)?;
        validate_four_lengths(solution.len(), direction.len(), residual.len(), image.len())?;
        for (((solution_value, &direction_value), residual_value), &image_value) in solution
            .iter_mut()
            .zip(direction.iter())
            .zip(residual.iter_mut())
            .zip(image.iter())
        {
            *solution_value += alpha * direction_value;
            *residual_value -= alpha * image_value;
        }
        Ok(())
    }

    fn combine_direction(
        &self,
        mut direction: Self::ViewMut<'_>,
        preconditioned_residual: Self::View<'_>,
        beta: Self::Scalar,
    ) -> Result<(), Self::Error> {
        let direction = contiguous_mut(&mut direction)?;
        let preconditioned_residual = contiguous(&preconditioned_residual)?;
        validate_lengths(direction.len(), preconditioned_residual.len())?;
        for (direction_value, &residual_value) in
            direction.iter_mut().zip(preconditioned_residual.iter())
        {
            *direction_value = residual_value + beta * *direction_value;
        }
        Ok(())
    }
}

fn contiguous<'a, T>(view: &'a ArrayView1<'_, T>) -> Result<&'a [T], LetoBackendError> {
    view.as_slice().ok_or(LetoBackendError::NonContiguousVector)
}

fn contiguous_mut<'a, T>(
    view: &'a mut ArrayViewMut1<'_, T>,
) -> Result<&'a mut [T], LetoBackendError> {
    view.as_mut_slice()
        .ok_or(LetoBackendError::NonContiguousVector)
}

fn validate_lengths(left: usize, right: usize) -> Result<(), LetoBackendError> {
    if left == right {
        Ok(())
    } else {
        Err(LetoBackendError::LengthMismatch { left, right })
    }
}

fn validate_three_lengths(
    first: usize,
    second: usize,
    third: usize,
) -> Result<(), LetoBackendError> {
    validate_lengths(first, second)?;
    validate_lengths(first, third)
}

fn validate_four_lengths(
    first: usize,
    second: usize,
    third: usize,
    fourth: usize,
) -> Result<(), LetoBackendError> {
    validate_three_lengths(first, second, third)?;
    validate_lengths(first, fourth)
}
