use crate::types::{R, T, T0};
use crate::{Q, ZETA};

// Some arith routines leverage dilithium https://github.com/PQClean/PQClean/tree/master/crypto_sign


// # Algorithm 43 `BitRev8()` is not implemented; zetas are pulled from pre-computed table
// `ZETA_TABLE_MONT`; see below (near end)

/// # Macro ensure!()
/// If the condition is not met, return an error Result. Borrowed from the `anyhow` crate.
macro_rules! ensure {
    ($cond:expr, $msg:literal $(,)?) => {
        if !$cond {
            return Err($msg);
        }
    };
}

pub(crate) use ensure; // make available throughout crate


/// Ensure all coefficients of polynomial `w` are within -lo to +hi (inclusive)
/// Note, while both range parameters are i32, they should be both non-negative
pub(crate) fn is_in_range(w: &R, lo: i32, hi: i32) -> bool {
    w.0.iter().all(|&e| (e >= -lo) && (e <= hi)) // success is CT, failure vartime
}


/// Partial Barrett-style reduction
// Arguably very slightly faster than single-step i128 below; worth more experimentation
#[allow(clippy::cast_possible_truncation)]
pub(crate) const fn partial_reduce64(a: i64) -> i32 {
    const M: i64 = (1 << 48) / (Q as i64);
    debug_assert!(a.abs() < (67_058_539 << 32), "partial_reduce64 input");
    let x = a >> 23;
    let a = a - x * (Q as i64);
    let x = a >> 23;
    let a = a - x * (Q as i64);
    let q = (a * M) >> 48;
    let res = a - q * (Q as i64);
    debug_assert!(res.abs() < 2 * Q as i64, "partial_reduce64 output");
    res as i32
}

#[allow(dead_code, clippy::cast_possible_truncation)] // I may come back to this and experiment more
pub(crate) const fn partial_reduce64b(a: i64) -> i32 {
    const MM: i64 = ((1 << 64) / (Q as i128)) as i64;
    let q = (a as i128 * MM as i128) >> 64; // only top half is relevant
    let res = a - (q as i64 * Q as i64);
    debug_assert!(res.abs() < 2 * Q as i64, "partial_reduce64b output");
    res as i32
}


/// Partially reduce a signed 32-bit value mod Q ---> `-Q <~ result <~ Q`
// Considering the positive case for `a`, bits 23 and above can be loosely
// viewed as the 'number of Q' contained within `a` (with some rounding-down
// error). So, increment these bits and then subtract off the corresponding
// number of Q. The result is within (better than) -Q < res < Q.
//
// The reductions below take secret coefficients. They use wrapping arithmetic so that a build with
// `overflow-checks = true` has no conditional branch on the value. On the input range the
// `debug_assert!`s state nothing wraps, so the results are those of plain `+`, `-` and `*`.
pub(crate) const fn partial_reduce32(a: i32) -> i32 {
    debug_assert!(a.abs() < 2_143_289_344, "partial_reduce32 input");
    let x = a.wrapping_add(1 << 22) >> 23;
    let res = a.wrapping_sub(x.wrapping_mul(Q));
    debug_assert!(res.abs() < Q, "partial_reduce32 output");
    res
}


pub(crate) const fn full_reduce32(a: i32) -> i32 {
    debug_assert!(a.abs() < 2_143_289_344, "full_reduce32 input");
    let x = partial_reduce32(a); // puts us within better than -Q to +Q
    let res = x.wrapping_add((x >> 31) & Q); // add Q if negative
    debug_assert!(res < Q, "full_reduce32 output");
    res
}


// Note: this is only used on 'fixed' security parameters (not secret values), so as not to impact CT
/// Bit length required to express `a` in bits
pub(crate) const fn bit_length(x: i32) -> usize { x.ilog2() as usize + 1 }


/// Mod +/- see definition on page 6.
/// If `α` is a positive integer and `m ∈ Z` or `m ∈ Z_α` , then m mod± α denotes the unique
/// element `m′ ∈ Z` in the range `−α/2 < m′ ≤ α/2` such that `m` and `m′` are congruent
/// modulo `α`.  'ready to optimize'
pub(crate) fn center_mod(m: i32) -> i32 {
    debug_assert!(m.abs() < 2_143_289_344, "center_mod input"); // for clarity; caught in full_reduce32
    let t = full_reduce32(m);
    let over2 = (Q / 2).wrapping_sub(t); // check if t is larger than Q/2
    let res = t.wrapping_sub((over2 >> 31) & Q); // sub Q if over2 is negative
    debug_assert_eq!(m.rem_euclid(Q), res.rem_euclid(Q), "center_mod output");
    res
}


/// # Algorithm 48: `MatrixVectorNTT(M_hat, v_hat)` on page 46.
/// Matrix by vector multiplication: `w_hat` = `A_hat` mul `u_hat`.
#[must_use]
pub(crate) fn mat_vec_mul<const K: usize, const L: usize>(
    a_hat: &[[T; L]; K], u_hat: &[T; L],
) -> [T; K] {
    let mut w_hat = [T0; K];
    let u_hat_mont = to_mont(u_hat);
    for i in 0..K {
        #[allow(clippy::needless_range_loop)] // clarity
        for j in 0..L {
            for (n, e) in w_hat[i].0.iter_mut().enumerate() {
                *e += mont_reduce(i64::from(a_hat[i][j].0[n]) * i64::from(u_hat_mont[j].0[n]));
            }
        }
    }
    w_hat
}


// Note Algorithm 44 has been dissolved into its place of use(s)

/// # Algorithm 46: `AddVectorNTT(v_hat, w_hat)` on page 45.
/// Computes the sum `v_hat + w_hat` of two vectors `v_hat`, `w_hat` over `𝑇_𝑞`.
///
/// **Input**:  `ℓ ∈ ℕ, v_hat ∈ 𝑇_𝑞^ℓ , w_hat ∈ 𝑇_𝑞^ℓ`. <br>
/// **Output**: `u_hat ∈ 𝑇_𝑞^ℓ`.
#[must_use]
pub(crate) fn add_vector_ntt<const K: usize>(v_hat: &[R; K], w_hat: &[R; K]) -> [R; K] {
    core::array::from_fn(|k| R(core::array::from_fn(|n| v_hat[k].0[n] + w_hat[k].0[n])))
}


#[allow(clippy::cast_possible_truncation)] // as i32
pub(crate) fn to_mont<const L: usize>(vec_a: &[T; L]) -> [T; L] {
    core::array::from_fn(|l| T(core::array::from_fn(|n| to_mont_coefficient(vec_a[l].0[n]))))
}


/// One coefficient into Montgomery form.
///
/// An NTT output is up to `|w_0| + 8 * Q` in magnitude, and with `|w_0| <= 2^19` that exceeds the
/// input range of `partial_reduce64`. So it is first reduced below `Q`. The result is congruent
/// to the same value and below `2 * Q`.
fn to_mont_coefficient(a: i32) -> i32 { partial_reduce64(i64::from(partial_reduce32(a)) << 32) }


/// The largest absolute value of the centered coefficients of `w`.
///
/// Signing passes `z`, `r0` and `c*t0`. Of these, only an accepted `z` appears in the signature:
/// `r0`, `c*t0` and a rejected `z` are secret. So the absolute value and the maximum are masks,
/// with no conditional branch on a coefficient.
pub(crate) fn infinity_norm<const ROW: usize>(w: &[R; ROW]) -> i32 {
    let mut max = 0i32;
    for row in w {
        for &element in &row.0 {
            // `center_mod` returns a value in `-(Q / 2)..=Q / 2`, so neither step below can wrap.
            let centered = center_mod(element);
            let sign = centered >> 31;
            let abs = (centered ^ sign).wrapping_sub(sign);
            let max_is_less = max.wrapping_sub(abs) >> 31;
            max ^= (max ^ abs) & max_is_less;
        }
    }
    max
}


/// # Algorithm 49: MontgomeryReduce(𝑎) on page 50.
/// Computes 𝑎 ⋅ 2−32 mod 𝑞.
///
/// **Input**:  Integer 𝑎 with −2^{31}*𝑞 ≤ 𝑎 ≤ 2^{31}*𝑞.
/// **Output**: 𝑟 ≡ 𝑎 ⋅ 2^{−32} mod 𝑞.
#[allow(clippy::cast_possible_truncation)] // a as i32, res as i32
pub(crate) const fn mont_reduce(a: i64) -> i32 {
    const QINV: i32 = 58_728_449; // (Q * QINV) % 2**32 = 1
    debug_assert!(a >= -17_996_808_479_301_632, "mont_reduce input (a)");
    debug_assert!(a <= 17_996_808_470_921_215, "mont_reduce input (b)");
    let t = (a as i32).wrapping_mul(QINV);
    let res = (a - (t as i64).wrapping_mul(Q as i64)) >> 32;
    debug_assert!(res < (Q as i64), "mont_reduce output 1");
    debug_assert!(-(Q as i64) < res, "mont_reduce output 2");
    res as i32
}


// ----- The following function only runs at compile time (thus, not CT etc) -----

#[allow(clippy::cast_possible_truncation)]
const fn gen_zeta_table_mont() -> [i32; 256] {
    let mut result = [0i32; 256];
    let mut x = 1i64;
    let mut i = 0u32;
    while i < 256 {
        result[(i as u8).reverse_bits() as usize] = ((x << 32) % (Q as i64)) as i32;
        x = (x * ZETA as i64) % (Q as i64);
        i += 1;
    }
    result
}


pub(crate) static ZETA_TABLE_MONT: [i32; 256] = gen_zeta_table_mont();


#[cfg(test)]
mod tests {
    use super::*;

    /// The norm as an iterator maximum, the form the masked loop replaces.
    fn reference_norm<const ROW: usize>(w: &[R; ROW]) -> i32 {
        w.iter()
            .flat_map(|row| row.0)
            .map(|element| center_mod(element).abs())
            .max()
            .expect("non-empty")
    }

    #[test]
    fn infinity_norm_matches_the_iterator_maximum() {
        const HALF: i32 = Q / 2;
        const EDGES: [i32; 12] = [
            0,
            1,
            -1,
            HALF,
            -HALF,
            HALF + 1,
            -HALF - 1,
            Q - 1,
            Q,
            Q + 1,
            2_143_289_343,
            -2_143_289_343,
        ];
        let filled = |value: i32| [R([value; 256]), R([value; 256])];
        let check = |w: &[R; 2]| assert_eq!(infinity_norm(w), reference_norm(w));

        // All zeros, and every edge value everywhere (ties across all 512 positions).
        check(&filled(0));
        for edge in EDGES {
            check(&filled(edge));
        }
        // A single nonzero coefficient, first, last and at the row boundary.
        for edge in EDGES {
            for (row, n) in [(0, 0), (1, 255), (0, 255), (1, 0), (0, 128)] {
                let mut w = filled(0);
                w[row].0[n] = edge;
                check(&w);
            }
        }
        // The largest coefficient first and last, over smaller values, and tied at both ends.
        for (a, b) in [(HALF, 5), (-HALF, 5), (HALF, -HALF), (7, 7)] {
            let mut w = filled(b);
            w[0].0[0] = a;
            check(&w);
            let mut w = filled(b);
            w[1].0[255] = a;
            check(&w);
            w[0].0[0] = a;
            check(&w);
        }
        // Generated arrays over the whole input range (xorshift64, fixed start).
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        for _ in 0..2_000 {
            let mut w = filled(0);
            for coefficient in w.iter_mut().flat_map(|row| row.0.iter_mut()) {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                #[allow(clippy::cast_possible_truncation)]
                let value = (state >> 32) as i32;
                *coefficient = value / 2;
            }
            check(&w);
        }
    }

    // The secret-key encoders centred with a branch; `center_mod` does it with a mask.
    #[test]
    fn center_mod_centres_every_canonical_coefficient() {
        for x in 0..Q {
            assert_eq!(center_mod(x), if x > Q / 2 { x - Q } else { x }, "center_mod({x})");
        }
    }

    // Every value an NTT output can hold: |w_0| <= 2^19, plus Q per layer, and more.
    #[cfg(not(debug_assertions))]
    #[test]
    #[ignore = "walks every NTT output; run with --release -- --ignored"]
    fn to_mont_coefficient_is_in_range_on_every_ntt_output() {
        let limit = (1 << 19) + 9 * Q;
        for a in -limit..=limit {
            let result = to_mont_coefficient(a);
            assert!(result.abs() < 2 * Q, "to_mont({a}) = {result}");
            let expected = (i64::from(a) << 32).rem_euclid(i64::from(Q));
            assert_eq!(i64::from(result).rem_euclid(i64::from(Q)), expected, "to_mont({a})");
        }
    }

    /// `center_mod` as written with plain arithmetic, `None` where an overflow check would panic.
    #[cfg(not(debug_assertions))]
    fn checked_center_mod(m: i32) -> Option<i32> {
        let x = m.checked_add(1 << 22)? >> 23;
        let res = m.checked_sub(x.checked_mul(Q)?)?;
        let t = res.checked_add((res >> 31) & Q)?;
        let over2 = (Q / 2).checked_sub(t)?;
        t.checked_sub((over2 >> 31) & Q)
    }

    // Release only: the reductions' `debug_assert!`s reject part of the range this walks.
    #[cfg(not(debug_assertions))]
    #[test]
    #[ignore = "exhaustive over i32; run with --release -- --ignored"]
    fn center_mod_matches_checked_arithmetic_on_every_i32() {
        for m in i32::MIN..=i32::MAX {
            if let Some(expected) = checked_center_mod(m) {
                assert_eq!(center_mod(m), expected, "center_mod({m})");
            }
        }
    }

    // Every value `center_mod` can return, through the norm's own absolute value.
    #[cfg(not(debug_assertions))]
    #[test]
    #[ignore = "walks every centered value; run with --release -- --ignored"]
    fn infinity_norm_is_the_absolute_value_of_every_centered_value() {
        for value in -(Q / 2)..=(Q / 2) {
            let mut w = [R([0; 256])];
            #[allow(clippy::cast_sign_loss)]
            let at = (value & 0xFF) as usize;
            w[0].0[at] = value;
            assert_eq!(infinity_norm(&w), checked_center_mod(value).expect("in range").abs());
        }
    }

    #[test]
    fn check_zeta() {
        let val = gen_zeta_table_mont();
        assert_eq!(val[0], 4_193_792);
        assert_eq!(val[1], 25_847);
        assert_eq!(val[2], 5_771_523);
    }

    #[test]
    fn test_partial_reduce64b() {
        // Test with various input values
        assert_eq!(partial_reduce64b(0), 0);
        assert_eq!(partial_reduce64b(i64::from(Q)), partial_reduce64(i64::from(Q)));
        assert_eq!(partial_reduce64b(i64::from(-Q)), partial_reduce64b(i64::from(-Q)));

        // Test with large positive and negative values
        let large_pos = i64::MAX / 64;
        let large_neg = -i64::MAX / 64;
        assert!(partial_reduce64b(large_pos).abs() < 2 * Q);
        assert!(partial_reduce64b(large_neg).abs() < 2 * Q);

        // Test with some specific values
        assert_eq!(partial_reduce64b(12_345_678), partial_reduce64(12_345_678));
        assert_eq!(partial_reduce64b(-12_345_678), partial_reduce64(-12_345_678));
    }
}
