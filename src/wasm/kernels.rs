// WASM SIMD128 kernels. Only the shapes PP-OCRv6 tiny actually uses.
// The TS side owns linear memory and passes byte offsets.

#![no_std]

use core::arch::wasm32::*;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// Marks the end of the static data segment. The TS allocator starts here.
#[no_mangle]
pub static mut HEAP_ANCHOR: [u8; 16] = [0; 16];

#[no_mangle]
pub unsafe extern "C" fn heap_base() -> usize {
    let end = (&raw const HEAP_ANCHOR as usize) + 16;
    (end + 15) & !15
}

const ACT_NONE: u32 = 0;
const ACT_RELU: u32 = 1;

#[inline(always)]
// One fused multiply-add. Without it the kernel is capped at half the
// machine's FLOPs: a separate mul and add each occupy an FP unit for one
// lane-op, where an FMA does both. The relaxed spec lets a CPU fuse or not,
// so results can differ in the last bit between machines. That is why the
// check against onnxruntime runs at 2e-3.
#[inline(always)]
unsafe fn fma(a: v128, b: v128, c: v128) -> v128 {
    f32x4_relaxed_madd(a, b, c)
}

unsafe fn apply(v: v128, act: u32) -> v128 {
    if act == ACT_RELU { f32x4_max(v, f32x4_splat(0.0)) } else { v }
}

#[inline(always)]
fn apply1(v: f32, act: u32) -> f32 {
    if act == ACT_RELU && v < 0.0 { 0.0 } else { v }
}

/// C[m,n] = A[m,k] * B[k,n] + bias[m], row-major, N contiguous. bias may be null.
#[no_mangle]
pub unsafe extern "C" fn gemm(
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    bias: *const f32,
    act: u32,
) {
    gemm_range(m, k, n, a, b, c, bias, act, 0, n);
}

/// The same GEMM restricted to output columns [lo, hi). Column ranges are
/// disjoint in C, so worker threads need no locking, only a barrier at the end.
///
/// The bulk runs an 8 row by 8 column micro-kernel: sixteen accumulators, so
/// sixteen independent multiply-add chains are in flight instead of four.
/// That is what this kernel is limited by - measured 19.6 GF/s with four
/// accumulators, 25.4 with eight, 35.9 with sixteen, on the same shape.
/// (Packing B into a contiguous panel was tried and measured 0.99-1.05x, so
/// locality is not the constraint here; the dependency chains are.)
/// One 8 row by 8 column tile of the product, held in 16 accumulators. The
/// accumulator count is what the kernel is limited by: 4 gives 19.6 GFLOP/s,
/// 8 gives 25.4, 16 gives 35.9.
#[inline(always)]
unsafe fn tile8x8(
    mi: usize,
    j: usize,
    k: usize,
    n: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    bias: *const f32,
    act: u32,
) {
    let mut acc_lo = [f32x4_splat(0.0); 8];
    let mut acc_hi = [f32x4_splat(0.0); 8];
    for r in 0..8 {
        let v = if bias.is_null() { 0.0 } else { *bias.add(mi + r) };
        acc_lo[r] = f32x4_splat(v);
        acc_hi[r] = f32x4_splat(v);
    }
    for kk in 0..k {
        let brow = b.add(kk * n + j);
        let b0 = v128_load(brow as *const v128);
        let b1 = v128_load(brow.add(4) as *const v128);
        for r in 0..8 {
            let av = f32x4_splat(*a.add((mi + r) * k + kk));
            acc_lo[r] = fma(av, b0, acc_lo[r]);
            acc_hi[r] = fma(av, b1, acc_hi[r]);
        }
    }
    for r in 0..8 {
        let cr = c.add((mi + r) * n + j);
        v128_store(cr as *mut v128, apply(acc_lo[r], act));
        v128_store(cr.add(4) as *mut v128, apply(acc_hi[r], act));
    }
}

#[no_mangle]
pub unsafe extern "C" fn gemm_range(
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    bias: *const f32,
    act: u32,
    lo: usize,
    hi: usize,
) {
    if k == 0 || lo >= hi {
        return;
    }
    let m8 = m & !7;
    let j_end = lo + ((hi - lo) & !7);

    // `mi` stays outside and `j` inside. Swapping them so the outer loop
    // re-reads the smaller operand looks right on paper and measured 0.5x:
    // with `j` outside, each tile writes 8 rows of C that are n*4 bytes apart
    // and moves on before any cache line is full.
    let mut mi = 0;
    while mi < m8 {
        let mut j = lo;
        while j < j_end {
            tile8x8(mi, j, k, n, a, b, c, bias, act);
            j += 8;
        }
        mi += 8;
    }

    // Columns past the last multiple of eight, for every row.
    if j_end < hi {
        gemm_edge(m, k, n, a, b, c, bias, act, j_end, hi);
    }
    // Rows past the last multiple of eight, for the columns the block covered.
    if m8 < m && lo < j_end {
        let bias_tail = if bias.is_null() { bias } else { bias.add(m8) };
        gemm_edge(m - m8, k, n, a.add(m8 * k), b, c.add(m8 * n), bias_tail, act, lo, j_end);
    }
}

/// Four-wide then scalar, for the ragged edges the micro-kernel skips.
unsafe fn gemm_edge(
    m: usize,
    k: usize,
    n: usize,
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    bias: *const f32,
    act: u32,
    lo: usize,
    hi: usize,
) {
    for mi in 0..m {
        let bv0 = if bias.is_null() { 0.0 } else { *bias.add(mi) };
        let ar = a.add(mi * k);
        let cr = c.add(mi * n);
        let mut j = lo;
        while j + 4 <= hi {
            let mut acc = f32x4_splat(bv0);
            for kk in 0..k {
                acc = fma(f32x4_splat(*ar.add(kk)), v128_load(b.add(kk * n + j) as *const v128), acc);
            }
            v128_store(cr.add(j) as *mut v128, apply(acc, act));
            j += 4;
        }
        while j < hi {
            let mut s = bv0;
            for kk in 0..k {
                s += *ar.add(kk) * *b.add(kk * n + j);
            }
            *cr.add(j) = apply1(s, act);
            j += 1;
        }
    }
}

/// Depthwise conv: one kh x kw filter per channel. Vectorised along the output row.
#[no_mangle]
pub unsafe extern "C" fn depthwise(
    channels: usize,
    ih: usize,
    iw: usize,
    oh: usize,
    ow: usize,
    kh: usize,
    kw: usize,
    sy: usize,
    sx: usize,
    pt: usize,
    pl: usize,
    x: *const f32,
    w: *const f32,
    bias: *const f32,
    y: *mut f32,
    act: u32,
    ch_lo: usize,
    ch_hi: usize,
) {
    let _ = channels;
    for ch in ch_lo..ch_hi {
        let xp = x.add(ch * ih * iw);
        let wp = w.add(ch * kh * kw);
        let yp = y.add(ch * oh * ow);
        let bv = if bias.is_null() { 0.0 } else { *bias.add(ch) };

        // With stride 1 a 4-wide output vector reads x[ox-pl .. ox+3-pl+kw-1].
        // Both ends are in bounds exactly when pl <= ox <= iw + pl - kw - 3.
        let vec_lo = pl;
        let vec_hi = if iw + pl >= kw + 3 { iw + pl - kw - 3 } else { 0 };

        for oy in 0..oh {
            let iy0 = (oy * sy) as isize - pt as isize;
            let mut ox = 0;

            if sx == 1 {
                while ox < vec_lo && ox < ow {
                    *yp.add(oy * ow + ox) =
                        apply1(dw_scalar(xp, wp, ih, iw, kh, kw, iy0, ox as isize - pl as isize, bv), act);
                    ox += 1;
                }
                while ox + 4 <= ow && ox <= vec_hi {
                    let mut acc = f32x4_splat(bv);
                    for ky in 0..kh {
                        let iy = iy0 + ky as isize;
                        if iy < 0 || iy >= ih as isize {
                            continue;
                        }
                        let row = xp.add(iy as usize * iw + ox - pl);
                        for kx in 0..kw {
                            acc = fma(
                                f32x4_splat(*wp.add(ky * kw + kx)),
                                v128_load(row.add(kx) as *const v128),
                                acc,
                            );
                        }
                    }
                    v128_store(yp.add(oy * ow + ox) as *mut v128, apply(acc, act));
                    ox += 4;
                }
            }

            while ox < ow {
                let ix0 = (ox * sx) as isize - pl as isize;
                *yp.add(oy * ow + ox) = apply1(dw_scalar(xp, wp, ih, iw, kh, kw, iy0, ix0, bv), act);
                ox += 1;
            }
        }
    }
}

#[inline(always)]
unsafe fn dw_scalar(
    xp: *const f32,
    wp: *const f32,
    ih: usize,
    iw: usize,
    kh: usize,
    kw: usize,
    iy0: isize,
    ix0: isize,
    bias: f32,
) -> f32 {
    let mut s = bias;
    for ky in 0..kh {
        let iy = iy0 + ky as isize;
        if iy < 0 || iy >= ih as isize {
            continue;
        }
        let row = xp.add(iy as usize * iw);
        for kx in 0..kw {
            let ix = ix0 + kx as isize;
            if ix < 0 || ix >= iw as isize {
                continue;
            }
            s += *row.add(ix as usize) * *wp.add(ky * kw + kx);
        }
    }
    s
}

/// Lay out patches as [Cin*kh*kw, OH*OW] so a dense conv becomes one gemm.
#[no_mangle]
pub unsafe extern "C" fn im2col(
    cin: usize,
    ih: usize,
    iw: usize,
    oh: usize,
    ow: usize,
    kh: usize,
    kw: usize,
    sy: usize,
    sx: usize,
    pt: usize,
    pl: usize,
    dy: usize,
    dx: usize,
    x: *const f32,
    col: *mut f32,
    c_lo: usize,
    c_hi: usize,
) {
    let _ = cin;
    let plane = oh * ow;
    for c in c_lo..c_hi {
        for ky in 0..kh {
            for kx in 0..kw {
                let dst = col.add(((c * kh + ky) * kw + kx) * plane);
                let src = x.add(c * ih * iw);
                for oy in 0..oh {
                    let iy = (oy * sy) as isize - pt as isize + (ky * dy) as isize;
                    if iy < 0 || iy >= ih as isize {
                        for ox in 0..ow {
                            *dst.add(oy * ow + ox) = 0.0;
                        }
                        continue;
                    }
                    let row = src.add(iy as usize * iw);
                    for ox in 0..ow {
                        let ix = (ox * sx) as isize - pl as isize + (kx * dx) as isize;
                        *dst.add(oy * ow + ox) = if ix < 0 || ix >= iw as isize {
                            0.0
                        } else {
                            *row.add(ix as usize)
                        };
                    }
                }
            }
        }
    }
}

// ---- transcendentals -------------------------------------------------------
// no_std has no libm, so exp is built here. exp(x) = 2^(x*log2e): split into an
// integer power of two (free, via the exponent field) and a polynomial for the
// remaining fraction in [-0.5, 0.5]. Relative error stays under ~1e-7, well
// inside f32 precision.

const LOG2E: f32 = 1.442695_04;
const P: [f32; 6] = [1.0, 0.693_147_2, 0.240_226_5, 0.055_504_1, 0.009_618_1, 0.001_333_3];

#[inline(always)]
fn expf(x: f32) -> f32 {
    let x = if x > 88.0 { 88.0 } else if x < -88.0 { -88.0 } else { x };
    let t = x * LOG2E;
    let k = if t >= 0.0 { (t + 0.5) as i32 } else { (t - 0.5) as i32 };
    let f = t - k as f32;
    let p = P[0] + f * (P[1] + f * (P[2] + f * (P[3] + f * (P[4] + f * P[5]))));
    p * f32::from_bits((((k + 127) as u32) & 0xff) << 23)
}

#[inline(always)]
unsafe fn expf4(x: v128) -> v128 {
    let hi = f32x4_splat(88.0);
    let lo = f32x4_splat(-88.0);
    let x = f32x4_min(f32x4_max(x, lo), hi);
    let t = f32x4_mul(x, f32x4_splat(LOG2E));
    let k = i32x4_trunc_sat_f32x4(f32x4_add(
        t,
        v128_bitselect(f32x4_splat(0.5), f32x4_splat(-0.5), f32x4_ge(t, f32x4_splat(0.0))),
    ));
    let f = f32x4_sub(t, f32x4_convert_i32x4(k));
    let mut p = f32x4_splat(P[5]);
    p = fma(p, f, f32x4_splat(P[4]));
    p = fma(p, f, f32x4_splat(P[3]));
    p = fma(p, f, f32x4_splat(P[2]));
    p = fma(p, f, f32x4_splat(P[1]));
    p = fma(p, f, f32x4_splat(P[0]));
    let bias = i32x4_shl(i32x4_add(k, i32x4_splat(127)), 23);
    f32x4_mul(p, bias)
}

// Abramowitz & Stegun 7.1.26, the same series the TypeScript path uses.
const E1: f32 = 0.254_829_592;
const E2: f32 = -0.284_496_736;
const E3: f32 = 1.421_413_741;
const E4: f32 = -1.453_152_027;
const E5: f32 = 1.061_405_429;
const EP: f32 = 0.327_591_1;

#[inline(always)]
fn erff(x: f32) -> f32 {
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let a = if x < 0.0 { -x } else { x };
    let t = 1.0 / (1.0 + EP * a);
    let y = 1.0 - ((((E5 * t + E4) * t + E3) * t + E2) * t + E1) * t * expf(-a * a);
    s * y
}

#[inline(always)]
unsafe fn erff4(x: v128) -> v128 {
    let sign = v128_and(x, i32x4_splat(i32::MIN));
    let a = f32x4_abs(x);
    let t = f32x4_div(f32x4_splat(1.0), f32x4_add(f32x4_splat(1.0), f32x4_mul(f32x4_splat(EP), a)));
    let mut poly = f32x4_splat(E5);
    poly = fma(poly, t, f32x4_splat(E4));
    poly = fma(poly, t, f32x4_splat(E3));
    poly = fma(poly, t, f32x4_splat(E2));
    poly = fma(poly, t, f32x4_splat(E1));
    let e = expf4(f32x4_neg(f32x4_mul(a, a)));
    let y = f32x4_sub(f32x4_splat(1.0), f32x4_mul(f32x4_mul(poly, t), e));
    v128_or(y, sign)
}

// ---- elementwise -----------------------------------------------------------

/// op: 0 add, 1 sub, 2 mul, 3 div.
/// mode: 0 same shape, 1 b is one scalar, 2 b is per-channel, 3 a is per-channel.
/// `inner` is the elements per channel and `channels` the channel count; both
/// are ignored unless the mode is per-channel.
/// Elementwise over two equal-length runs. Kept out of `binary` so that
/// function stays a leaf: making it call itself for the broadcast mode cost
/// detection 8%, because every elementwise op in the model goes through mode 0.
#[inline(always)]
unsafe fn elementwise(op: u32, n: usize, a: *const f32, b: *const f32, out: *mut f32) {
    let mut i = 0;
    while i + 4 <= n {
        let av = v128_load(a.add(i) as *const v128);
        let bv = v128_load(b.add(i) as *const v128);
        v128_store(out.add(i) as *mut v128, apply_bin(op, av, bv));
        i += 4;
    }
    while i < n {
        *out.add(i) = apply_bin1(op, *a.add(i), *b.add(i));
        i += 1;
    }
}

#[no_mangle]
pub unsafe extern "C" fn binary(
    op: u32,
    mode: u32,
    n: usize,
    inner: usize,
    channels: usize,
    a: *const f32,
    b: *const f32,
    out: *mut f32,
) {
    match mode {
        0 => elementwise(op, n, a, b, out),
        1 => splat_rhs(op, n, a, *b, out, false),
        // b is the trailing axis repeated over every row: [1, R, C] + [C].
        4 => {
            let rows = n / inner;
            for o in 0..rows {
                elementwise(op, inner, a.add(o * inner), b, out.add(o * inner));
            }
        }
        2 | 3 => {
            let outer = n / (inner * channels);
            for o in 0..outer {
                for c in 0..channels {
                    let base = (o * channels + c) * inner;
                    if mode == 2 {
                        splat_rhs(op, inner, a.add(base), *b.add(c), out.add(base), false);
                    } else {
                        splat_rhs(op, inner, b.add(base), *a.add(c), out.add(base), true);
                    }
                }
            }
        }
        _ => {}
    }
}

#[inline(always)]
unsafe fn apply_bin(op: u32, x: v128, y: v128) -> v128 {
    match op {
        0 => f32x4_add(x, y),
        1 => f32x4_sub(x, y),
        2 => f32x4_mul(x, y),
        _ => f32x4_div(x, y),
    }
}

#[inline(always)]
fn apply_bin1(op: u32, x: f32, y: f32) -> f32 {
    match op {
        0 => x + y,
        1 => x - y,
        2 => x * y,
        _ => x / y,
    }
}

/// `flip` means the scalar is the left operand, which matters for sub and div.
#[inline(always)]
unsafe fn splat_rhs(op: u32, n: usize, a: *const f32, s: f32, out: *mut f32, flip: bool) {
    let sv = f32x4_splat(s);
    let mut i = 0;
    while i + 4 <= n {
        let av = v128_load(a.add(i) as *const v128);
        let r = if flip { apply_bin(op, sv, av) } else { apply_bin(op, av, sv) };
        v128_store(out.add(i) as *mut v128, r);
        i += 4;
    }
    while i < n {
        let x = *a.add(i);
        *out.add(i) = if flip { apply_bin1(op, s, x) } else { apply_bin1(op, x, s) };
        i += 1;
    }
}

/// op: 0 relu, 1 sigmoid, 2 erf, 3 hard sigmoid (p0 = alpha, p1 = beta).
#[no_mangle]
pub unsafe extern "C" fn unary(op: u32, n: usize, a: *const f32, out: *mut f32, p0: f32, p1: f32) {
    let mut i = 0;
    let one = f32x4_splat(1.0);
    let zero = f32x4_splat(0.0);
    while i + 4 <= n {
        let x = v128_load(a.add(i) as *const v128);
        let r = match op {
            0 => f32x4_max(x, zero),
            1 => f32x4_div(one, f32x4_add(one, expf4(f32x4_neg(x)))),
            2 => erff4(x),
            3 => f32x4_min(
                one,
                f32x4_max(zero, fma(f32x4_splat(p0), x, f32x4_splat(p1))),
            ),
            // gelu, folded from Div -> Erf -> Add -> Mul -> Mul. One pass.
            _ => f32x4_mul(
                f32x4_mul(x, f32x4_splat(p1)),
                f32x4_add(one, erff4(f32x4_mul(x, f32x4_splat(p0)))),
            ),
        };
        v128_store(out.add(i) as *mut v128, r);
        i += 4;
    }
    while i < n {
        let x = *a.add(i);
        *out.add(i) = match op {
            0 => if x > 0.0 { x } else { 0.0 },
            1 => 1.0 / (1.0 + expf(-x)),
            2 => erff(x),
            3 => {
                let v = p0 * x + p1;
                if v < 0.0 { 0.0 } else if v > 1.0 { 1.0 } else { v }
            }
            _ => x * p1 * (1.0 + erff(x * p0)),
        };
        i += 1;
    }
}

/// Mean over a contiguous trailing run, one output per `inner`-sized block.
/// Per-channel affine: y = x * s[c] + t[c], with the channel running over
/// `channels` blocks of `inner` elements. Batch normalisation collapses to
/// this once its four constants are folded into a scale and a shift.
#[no_mangle]
pub unsafe extern "C" fn affine_channels(
    n: usize,
    inner: usize,
    channels: usize,
    a: *const f32,
    s: *const f32,
    t: *const f32,
    out: *mut f32,
    c_lo: usize,
    c_hi: usize,
) {
    let outer = n / (inner * channels);
    for o in 0..outer {
        for c in c_lo..c_hi {
            let base = (o * channels + c) * inner;
            let sv = f32x4_splat(*s.add(c));
            let tv = f32x4_splat(*t.add(c));
            let ap = a.add(base);
            let op = out.add(base);
            let mut i = 0;
            while i + 4 <= inner {
                v128_store(
                    op.add(i) as *mut v128,
                    fma(v128_load(ap.add(i) as *const v128), sv, tv),
                );
                i += 4;
            }
            while i < inner {
                *op.add(i) = *ap.add(i) * *s.add(c) + *t.add(c);
                i += 1;
            }
        }
    }
}

/// Softmax over the trailing axis: `rows` rows of `cols` each. Subtracting the
/// row maximum before exp is what keeps the classifier's 6906 logits finite.
#[no_mangle]
pub unsafe extern "C" fn softmax_rows(
    cols: usize,
    a: *const f32,
    out: *mut f32,
    r_lo: usize,
    r_hi: usize,
) {
    for r in r_lo..r_hi {
        let ar = a.add(r * cols);
        let orow = out.add(r * cols);
        let mut m = f32::NEG_INFINITY;
        for i in 0..cols {
            let v = *ar.add(i);
            if v > m {
                m = v;
            }
        }
        let mut sum = 0.0f32;
        let mv = f32x4_splat(m);
        let mut i = 0;
        while i + 4 <= cols {
            let e = expf4(f32x4_sub(v128_load(ar.add(i) as *const v128), mv));
            v128_store(orow.add(i) as *mut v128, e);
            sum += f32x4_extract_lane::<0>(e) + f32x4_extract_lane::<1>(e)
                + f32x4_extract_lane::<2>(e) + f32x4_extract_lane::<3>(e);
            i += 4;
        }
        while i < cols {
            let e = expf(*ar.add(i) - m);
            *orow.add(i) = e;
            sum += e;
            i += 1;
        }
        let inv = f32x4_splat(1.0 / sum);
        let mut j = 0;
        while j + 4 <= cols {
            v128_store(
                orow.add(j) as *mut v128,
                f32x4_mul(v128_load(orow.add(j) as *const v128), inv),
            );
            j += 4;
        }
        while j < cols {
            *orow.add(j) *= 1.0 / sum;
            j += 1;
        }
    }
}

/// Rank-4 transpose. Ranks below four are passed left-padded with ones, so one
/// kernel covers the 3D permutes the recognition head does.
#[no_mangle]
pub unsafe extern "C" fn transpose4(
    d0: usize,
    d1: usize,
    d2: usize,
    d3: usize,
    s0: usize,
    s1: usize,
    s2: usize,
    s3: usize,
    a: *const f32,
    out: *mut f32,
) {
    let mut o = 0;
    for i0 in 0..d0 {
        for i1 in 0..d1 {
            for i2 in 0..d2 {
                let base = i0 * s0 + i1 * s1 + i2 * s2;
                for i3 in 0..d3 {
                    *out.add(o) = *a.add(base + i3 * s3);
                    o += 1;
                }
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn reduce_mean(outer: usize, inner: usize, a: *const f32, out: *mut f32) {
    for o in 0..outer {
        let base = a.add(o * inner);
        let mut acc = f32x4_splat(0.0);
        let mut i = 0;
        while i + 4 <= inner {
            acc = f32x4_add(acc, v128_load(base.add(i) as *const v128));
            i += 4;
        }
        let mut s = f32x4_extract_lane::<0>(acc)
            + f32x4_extract_lane::<1>(acc)
            + f32x4_extract_lane::<2>(acc)
            + f32x4_extract_lane::<3>(acc);
        while i < inner {
            s += *base.add(i);
            i += 1;
        }
        *out.add(o) = s / inner as f32;
    }
}

/// 2x2 max pool, stride 1, SAME_UPPER: output keeps the input size and only
/// the last row and column clamp instead of reading padding.
#[no_mangle]
pub unsafe extern "C" fn maxpool2x2(planes: usize, h: usize, w: usize, a: *const f32, out: *mut f32, p_lo: usize, p_hi: usize) {
    let _ = planes;
    for p in p_lo..p_hi {
        let base = a.add(p * h * w);
        let dst = out.add(p * h * w);
        for y in 0..h {
            let r0 = base.add(y * w);
            let r1 = if y + 1 < h { r0.add(w) } else { r0 };
            let d = dst.add(y * w);
            let mut x = 0;
            while x + 4 <= w.saturating_sub(1) {
                let a0 = v128_load(r0.add(x) as *const v128);
                let a1 = v128_load(r0.add(x + 1) as *const v128);
                let b0 = v128_load(r1.add(x) as *const v128);
                let b1 = v128_load(r1.add(x + 1) as *const v128);
                v128_store(d.add(x) as *mut v128, f32x4_max(f32x4_max(a0, a1), f32x4_max(b0, b1)));
                x += 4;
            }
            while x < w {
                let xr = if x + 1 < w { x + 1 } else { x };
                let m1 = fmaxf(*r0.add(x), *r0.add(xr));
                let m2 = fmaxf(*r1.add(x), *r1.add(xr));
                *d.add(x) = fmaxf(m1, m2);
                x += 1;
            }
        }
    }
}

#[inline(always)]
fn fmaxf(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

/// Nearest-neighbour upsample by integer factors on the last two axes.
/// Writes one 2x2 tap of a stride-2 transposed convolution into the upsampled
/// plane: dst[p][2y + ky][2x + kx] = src[p][y][x]. The four taps land on
/// disjoint pixels, so nothing accumulates and each tap's GEMM can carry the
/// bias itself.
#[no_mangle]
pub unsafe extern "C" fn scatter2x2(
    h: usize,
    w: usize,
    ky: usize,
    kx: usize,
    src: *const f32,
    dst: *mut f32,
    p_lo: usize,
    p_hi: usize,
) {
    let ow = w * 2;
    for p in p_lo..p_hi {
        let sp = src.add(p * h * w);
        let dp = dst.add(p * h * 2 * ow);
        for y in 0..h {
            let srow = sp.add(y * w);
            let drow = dp.add((y * 2 + ky) * ow + kx);
            for x in 0..w {
                *drow.add(x * 2) = *srow.add(x);
            }
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn resize_nearest(
    planes: usize,
    h: usize,
    w: usize,
    sh: usize,
    sw: usize,
    a: *const f32,
    out: *mut f32,
    p_lo: usize,
    p_hi: usize,
) {
    let _ = planes;
    let ow = w * sw;
    let oh = h * sh;
    for p in p_lo..p_hi {
        let src = a.add(p * h * w);
        let dst = out.add(p * oh * ow);
        for y in 0..h {
            let srow = src.add(y * w);
            let first = dst.add(y * sh * ow);
            for x in 0..w {
                let v = *srow.add(x);
                for k in 0..sw {
                    *first.add(x * sw + k) = v;
                }
            }
            for r in 1..sh {
                core::ptr::copy_nonoverlapping(first, dst.add((y * sh + r) * ow), ow);
            }
        }
    }
}
