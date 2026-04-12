//! Leaf numerical kernels used by the hot covariance/extraction paths.
//!
//! The only `unsafe` code in the crate lives here, isolated to optional AArch64
//! NEON implementations. All other code paths remain safe Rust.

#[inline]
pub(crate) fn axpy_in_place(dst: &mut [f64], src: &[f64], scale: f64) {
    debug_assert_eq!(dst.len(), src.len());

    #[cfg(target_arch = "aarch64")]
    unsafe {
        axpy_in_place_neon(dst, src, scale);
    }

    #[cfg(not(target_arch = "aarch64"))]
    scalar_axpy_in_place(dst, src, scale);
}

#[inline]
pub(crate) fn scale_into(dst: &mut [f64], src: &[f64], scale: f64) {
    debug_assert_eq!(dst.len(), src.len());

    #[cfg(target_arch = "aarch64")]
    unsafe {
        scale_into_neon(dst, src, scale);
    }

    #[cfg(not(target_arch = "aarch64"))]
    scalar_scale_into(dst, src, scale);
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn scalar_axpy_in_place(dst: &mut [f64], src: &[f64], scale: f64) {
    for (dst_value, src_value) in dst.iter_mut().zip(src.iter().copied()) {
        *dst_value += scale * src_value;
    }
}

#[cfg(not(target_arch = "aarch64"))]
#[inline]
fn scalar_scale_into(dst: &mut [f64], src: &[f64], scale: f64) {
    for (dst_value, src_value) in dst.iter_mut().zip(src.iter().copied()) {
        *dst_value = scale * src_value;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn axpy_in_place_neon(dst: &mut [f64], src: &[f64], scale: f64) {
    use std::arch::aarch64::{vaddq_f64, vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64};

    let scale_vec = vdupq_n_f64(scale);
    let mut i = 0;

    while i + 2 <= dst.len() {
        // SAFETY:
        // - `i + 2 <= len`, so reading and writing two f64 lanes at offset `i`
        //   stays within both slices.
        // - `dst` and `src` lengths are asserted equal by the caller wrapper.
        // - AArch64 guarantees NEON support for this target configuration.
        let dst_ptr = unsafe { dst.as_mut_ptr().add(i) };
        let src_ptr = unsafe { src.as_ptr().add(i) };
        let dst_vec = unsafe { vld1q_f64(dst_ptr) };
        let src_vec = unsafe { vld1q_f64(src_ptr) };
        unsafe { vst1q_f64(dst_ptr, vaddq_f64(dst_vec, vmulq_f64(src_vec, scale_vec))) };
        i += 2;
    }

    while i < dst.len() {
        // SAFETY: loop guard ensures `i < len` for both equally-sized slices.
        unsafe {
            *dst.get_unchecked_mut(i) += scale * *src.get_unchecked(i);
        }
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_into_neon(dst: &mut [f64], src: &[f64], scale: f64) {
    use std::arch::aarch64::{vdupq_n_f64, vld1q_f64, vmulq_f64, vst1q_f64};

    let scale_vec = vdupq_n_f64(scale);
    let mut i = 0;

    while i + 2 <= dst.len() {
        // SAFETY:
        // - `i + 2 <= len`, so both lane loads/stores are in-bounds.
        // - `dst` and `src` lengths are asserted equal by the caller wrapper.
        // - AArch64 guarantees NEON support for this target configuration.
        let dst_ptr = unsafe { dst.as_mut_ptr().add(i) };
        let src_ptr = unsafe { src.as_ptr().add(i) };
        let src_vec = unsafe { vld1q_f64(src_ptr) };
        unsafe { vst1q_f64(dst_ptr, vmulq_f64(src_vec, scale_vec)) };
        i += 2;
    }

    while i < dst.len() {
        // SAFETY: loop guard ensures `i < len` for both equally-sized slices.
        unsafe {
            *dst.get_unchecked_mut(i) = scale * *src.get_unchecked(i);
        }
        i += 1;
    }
}
