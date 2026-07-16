use crate::needle::{build_case, draw_identities, SplitMix64};

#[test]
fn case_generation_is_seed_deterministic() {
    let mut rng_a = SplitMix64(42);
    let mut rng_b = SplitMix64(42);
    let case_a = build_case(&mut rng_a, 200, 50, false);
    let case_b = build_case(&mut rng_b, 200, 50, false);
    assert_eq!(case_a.text, case_b.text);
    assert_eq!(case_a.expect, case_b.expect);
}

#[test]
fn case_contains_its_needle_and_question() {
    let mut rng = SplitMix64(7);
    let case = build_case(&mut rng, 100, 10, false);
    assert!(case.text.contains(&format!("is {}. ", case.expect)));
    assert!(case.text.contains("Reply with just the number."));
}

#[test]
fn multi_case_plants_three_distinct_needles() {
    let mut rng = SplitMix64(7);
    let case = build_case(&mut rng, 150, 25, true);
    assert_eq!(case.text.matches("The secret passkey for").count(), 3);
}

#[test]
fn identities_are_distinct() {
    let mut rng = SplitMix64(3);
    let identities = draw_identities(&mut rng, 3);
    assert_eq!(identities.len(), 3);
    for first in 0..3 {
        for second in (first + 1)..3 {
            assert!(identities[first].0 != identities[second].0);
            assert!(identities[first].1 != identities[second].1);
        }
    }
}

#[test]
fn depth_orders_needle_position() {
    // Same seed, different depth: the deep (10%) needle lands earlier in
    // the text than the shallow (90%) one.
    let mut rng_shallow = SplitMix64(9);
    let shallow = build_case(&mut rng_shallow, 300, 90, false);
    let mut rng_deep = SplitMix64(9);
    let deep = build_case(&mut rng_deep, 300, 10, false);
    let shallow_pos = shallow.text.find("secret passkey").unwrap();
    let deep_pos = deep.text.find("secret passkey").unwrap();
    assert!(deep_pos < shallow_pos);
}
