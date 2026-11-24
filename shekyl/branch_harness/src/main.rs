// Copyright (c) 2026, The Shekyl Foundation
//
// All rights reserved.
// BSD-3-Clause

//! Key generation, signing and verification for one ML-DSA parameter set.
//!
//! These paths reach the eight coefficient helpers `shekyl/check_branch_counts.py` names. The
//! harness goes through the public API because whether a helper compiles to a branch depends on
//! where it is inlined: calling them all from one function produced a shape no library caller has.

use core::hint::black_box;
use fips204::traits::{KeyGen, Signer, Verifier};

#[cfg(not(any(
    all(feature = "p44", not(feature = "p65"), not(feature = "p87")),
    all(feature = "p65", not(feature = "p44"), not(feature = "p87")),
    all(feature = "p87", not(feature = "p44"), not(feature = "p65")),
)))]
compile_error!("enable exactly one of the features p44, p65 and p87");

#[cfg(feature = "p44")]
use fips204::ml_dsa_44 as ml_dsa;
#[cfg(feature = "p65")]
use fips204::ml_dsa_65 as ml_dsa;
#[cfg(feature = "p87")]
use fips204::ml_dsa_87 as ml_dsa;

fn main() {
    let (public, private) = ml_dsa::KG::keygen_from_seed(black_box(&[7; 32]));
    let message = black_box(b"fips204 branch harness");
    let signature = private.try_sign_with_seed(black_box(&[9; 32]), message, b"").expect("signs");
    assert!(public.verify(message, &signature, b""), "the signature verifies");
}
