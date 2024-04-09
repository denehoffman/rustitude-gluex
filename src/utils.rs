pub fn breakup_momentum(m0: f64, m1: f64, m2: f64) -> f64 {
    f64::sqrt(f64::abs(
        m0.powi(4) + m1.powi(4) + m2.powi(4)
            - 2.0 * (m0.powi(2) * m1.powi(2) + m0.powi(2) * m2.powi(2) + m1.powi(2) * m2.powi(2)),
    )) / (2.0 * m0)
}

pub fn blatt_weisskopf(m0: f64, m1: f64, m2: f64, l: usize) -> f64 {
    let q = breakup_momentum(m0, m1, m2);
    let z = q.powi(2) / f64::powi(0.1973, 2);
    match l {
        0 => 1.0,
        1 => f64::sqrt((2.0 * z) / (z + 1.0)),
        2 => f64::sqrt((13.0 * z.powi(2)) / ((z - 3.0).powi(2) + 9.0 * z)),
        3 => f64::sqrt(
            (277.0 * z.powi(3)) / (z * (z - 15.0).powi(2) + 9.0 * (2.0 * z - 5.0).powi(2)),
        ),
        4 => f64::sqrt(
            (12746.0 * z.powi(4)) / (z.powi(2) - 45.0 * z + 105.0).powi(2)
                + 25.0 * z * (2.0 * z - 21.0).powi(2),
        ),
        l => panic!("L = {l} is not yet implemented"),
    }
}
