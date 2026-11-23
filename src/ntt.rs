// This file implements functionality from FIPS 204 section 7.5 `NTT` and `invNTT`

use crate::helpers::{full_reduce32, mont_reduce, partial_reduce32, ZETA_TABLE_MONT};
use crate::types::{R, T};
use crate::Q;


/// # Algorithm 41 NTT(w) on page 43.
/// Computes the Number-Theoretic Transform. An inner loop over `w/w_hat` has
/// been refactored into this function, so it processes an array of elements.
///
/// **Input**: polynomial `w(X) = ∑_{j=0}^{255} w_j X^j ∈ R_q` <br>
/// **Output**: `w_hat = (w_hat[0], ... , w_hat[255]) ∈ T_q`
///
/// Range: each of the 8 layers adds or subtracts one Montgomery result below `Q`, so an input
/// `|w_j| <= 2^19` (the largest any caller passes: `z`) leaves every output below `2^19 + 8 * Q`,
/// under `2^27`. With `|zeta| < Q`, each product stays well inside `mont_reduce`'s input range.
pub(crate) fn ntt<const KL: usize>(w: &[R; KL]) -> [T; KL] {
    // 1: for j from 0 to 255 do
    // 2: w_hat[j] ← w_j
    // 3: end for
    let mut w_hat: [T; KL] = core::array::from_fn(|x| T(core::array::from_fn(|n| w[x].0[n])));

    // for each element of w_hat
    for w_poly in &mut w_hat {
        //
        // 4: m ← 0
        let mut m = 0;

        // 5: len ← 128
        let mut len = 128;

        // 6: while len ≥ 1 do
        while len >= 1 {
            //
            // 7: start ← 0
            let mut start = 0;

            // 8: while start < 256 do
            while start < 256 {
                //
                // 9: m ← m + 1
                m += 1;

                // 10: zeta ← ζ^{brv(k)} mod q
                let zeta = i64::from(ZETA_TABLE_MONT[m]);

                // 11: for j from start to start + len − 1 do
                for j in start..(start + len) {
                    //
                    // 12: t ← zeta · w_hat[j + len]
                    let t = mont_reduce(zeta * i64::from(w_poly.0[j + len]));

                    // 13: w_hat[j + len] ← w_hat[j] − t
                    w_poly.0[j + len] = w_poly.0[j] - t;

                    // 14: w_hat[j] ← w_hat[j] + t
                    w_poly.0[j] += t;

                    // 15: end for
                }

                // 16: start ← start + 2 · len
                start += 2 * len;

                // 17: end while
            }

            // 18: len ← ⌊len/2⌋
            len >>= 1;

            // 19: end while
        }

        // end for each element of w_hat
    }

    // 20: return ŵ
    w_hat
}


/// # Algorithm 42 NTT−1 (`w_hat`) on page 44.
/// Computes the inverse of the Number-Theoretic Transform. An inner loop over `w/w_hat` has
/// been refactored into this function, so it processes an array of elements.
///
/// **Input**: `w_hat` = `(w_hat[0], . . . , w_hat[255]) ∈ T_q` <br>
/// **Output**: polynomial `w(X) = ∑_{j=0}^{255} w_j X^j ∈ R_q`
///
/// The input may be a sum of Montgomery products: up to `(L + 1) * Q` in magnitude, `L <= 7`. It
/// is first reduced below `Q`, as the reference implementation does before its inverse NTT,
/// because the butterfly's sum half is never reduced and doubles at each of the 8 layers.
pub(crate) fn inv_ntt<const KL: usize>(w_hat: &[T; KL]) -> [R; KL] {
    // |w_hat| <= 8Q < 2^26, inside `partial_reduce32`'s input range; the result is below Q.
    let reduced: [T; KL] =
        core::array::from_fn(|x| T(core::array::from_fn(|n| partial_reduce32(w_hat[x].0[n]))));
    inv_ntt_reduced(&reduced)
}


/// The layers of Algorithm 42, for an input already below `Q` in magnitude.
///
/// Range, for every such input. After layer `k` (1 to 8), every `|w_j| < 2^k * Q`:
/// - the sum and the difference of two values below `2^(k-1) * Q` are below `2^k * Q`;
/// - the difference is then multiplied by `zeta` and Montgomery-reduced, which leaves it below `Q`.
///
/// At `k = 8` that is `256 * Q = 2_145_386_752 < 2^31`, so no `i32` addition or subtraction
/// overflows. With `|zeta| < Q` and `|f| < Q`, each product is below `256 * Q^2 < 2^31 * Q`, the
/// input range of `mont_reduce`, whose result is below `Q`.
fn inv_ntt_reduced<const KL: usize>(w_hat: &[T; KL]) -> [R; KL] {
    #[allow(clippy::cast_possible_truncation)]
    const F_MONT: i64 = 8_347_681_i128.wrapping_mul(1 << 32).rem_euclid(Q as i128) as i64;
    debug_assert!(
        w_hat.iter().flat_map(|row| row.0.iter()).all(|x| x.abs() < Q),
        "inv_ntt_reduced input"
    );
    //
    // 1: for j from 0 to 255 do
    // 2: w_j ← w_hat[j]
    // 3: end for
    let mut w_out: [R; KL] = core::array::from_fn(|x| R(core::array::from_fn(|n| w_hat[x].0[n])));

    // for each element of w_hat
    for w_poly in &mut w_out {
        //
        // 4: m ← 256
        let mut m = 256;

        // 5: len ← 1
        let mut len = 1;

        // 6: while len < 256 do
        while len < 256 {
            //
            // 7: start ← 0
            let mut start = 0;

            // 8: while start < 256 do
            while start < 256 {
                //
                // 9: m ← m − 1
                m -= 1;

                // 10: zeta ← −ζ^{brv(k)} mod q    ▷ 𝑧 ← −𝜁 BitRev8 (𝑚) mod 𝑞
                let zeta = -ZETA_TABLE_MONT[m];

                // 11: for j from start to start + len − 1 do
                for j in start..(start + len) {
                    //
                    // 12: t ← w_j
                    let t = w_poly.0[j];

                    // 13: w_j ← t + w_{j+len}
                    w_poly.0[j] = t + w_poly.0[j + len];

                    // 14: w_{j+len} ← t − w_{j+len}
                    w_poly.0[j + len] = t - w_poly.0[j + len];

                    // 15: w_{j+len} ← zeta · w_{j+len}
                    w_poly.0[j + len] = mont_reduce(i64::from(zeta) * i64::from(w_poly.0[j + len]));

                    // 16: end for
                }

                // 17: start ← start + 2 · len
                start += 2 * len;

                // 18: end while
            }

            // 19: len ← 2 · len
            len <<= 1;

            // 20: end while
        }

        // 21: f ← 8347681          ▷ f = 256^{−1} mod q
        // 22: for j from 0 to 255 do
        // 23: wj ← f · wj
        for i in &mut w_poly.0 {
            *i = full_reduce32(mont_reduce(F_MONT * i64::from(*i)));
        }

        // 24: end for
    }

    // 25: return w
    w_out
}


#[cfg(test)]
mod tests {
    use super::*;

    const TOP: i32 = Q - 1;

    /// Algorithm 42 with no reduction first and every `i32` step checked: the arithmetic before
    /// the range hardening. `None` where an overflow check would have panicked.
    #[allow(clippy::cast_possible_truncation)]
    fn checked_inv_ntt(w_hat: &[i32; 256]) -> Option<[i32; 256]> {
        const F_MONT: i64 = 8_347_681_i128.wrapping_mul(1 << 32).rem_euclid(Q as i128) as i64;
        let mut w = *w_hat;
        let (mut m, mut len) = (256, 1);
        while len < 256 {
            let mut start = 0;
            while start < 256 {
                m -= 1;
                let zeta = -ZETA_TABLE_MONT[m];
                for j in start..(start + len) {
                    let t = w[j];
                    w[j] = t.checked_add(w[j + len])?;
                    let difference = t.checked_sub(w[j + len])?;
                    w[j + len] = mont_reduce(i64::from(zeta) * i64::from(difference));
                }
                start += 2 * len;
            }
            len <<= 1;
        }
        Some(core::array::from_fn(|i| full_reduce32(mont_reduce(F_MONT * i64::from(w[i])))))
    }

    /// Every sign pattern that keeps the sum half at its largest: all one sign, and alternating
    /// in blocks of each butterfly length.
    fn worst_patterns(magnitude: i32) -> [[i32; 256]; 18] {
        core::array::from_fn(|p| {
            let sign = if p % 2 == 0 { 1 } else { -1 };
            let block = 1_usize << (p / 2);
            core::array::from_fn(|i| {
                if p < 2 || (i / block) % 2 == 0 {
                    sign * magnitude
                } else {
                    -sign * magnitude
                }
            })
        })
    }

    // The bound on the layers, at its limit, with overflow checks on in this test profile.
    #[test]
    fn inv_ntt_layers_hold_every_input_at_the_largest_magnitude() {
        for pattern in worst_patterns(TOP) {
            let out = inv_ntt_reduced(&[T(pattern)]);
            assert!(out[0].0.iter().all(|x| (0..Q).contains(x)));
        }
    }

    // The largest input any caller passes: L + 1 = 8 Montgomery products.
    #[test]
    fn inv_ntt_takes_the_largest_sum_a_caller_passes() {
        for magnitude in [TOP, 2 * TOP, 6 * TOP, 8 * TOP] {
            for pattern in worst_patterns(magnitude) {
                let out = inv_ntt(&[T(pattern)]);
                assert!(out[0].0.iter().all(|x| (0..Q).contains(x)));
            }
        }
    }

    // The premises of the two range proofs above.
    #[test]
    fn ntt_constants_are_below_q() {
        assert!(ZETA_TABLE_MONT.iter().all(|zeta| zeta.abs() < Q));
        const { assert!(8_347_681 < Q) };
    }

    #[test]
    fn ntt_holds_its_bound_at_the_largest_input() {
        for pattern in worst_patterns(1 << 19) {
            let out = ntt(&[R(pattern)]);
            assert!(out[0].0.iter().all(|x| x.abs() < (1 << 19) + 8 * Q));
        }
    }

    // Wherever the old arithmetic returned a value, the hardened path returns the same one.
    #[test]
    fn inv_ntt_matches_the_old_arithmetic_wherever_it_returned() {
        let mut state = 0x243F_6A88_85A3_08D3_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut compared = 0;
        for round in 0..4_000_i64 {
            // Magnitudes from one to eight products, as the callers produce them.
            let bound = i64::from(Q) * (round % 8 + 1);
            let input: [i32; 256] = core::array::from_fn(|_| {
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                let value = (next() >> 1) as i64 % (2 * bound + 1) - bound;
                #[allow(clippy::cast_possible_truncation)]
                let value = value as i32;
                value
            });
            if let Some(old) = checked_inv_ntt(&input) {
                assert_eq!(inv_ntt(&[T(input)])[0].0, old);
                compared += 1;
            }
        }
        assert!(compared > 3_900, "the comparison must have run: {compared}");
    }
}
