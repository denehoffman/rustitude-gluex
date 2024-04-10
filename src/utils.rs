use factorial::Factorial;
use rustitude::prelude::*;

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

pub fn small_wigner_d_matrix(beta: f64, j: usize, m: isize, n: isize) -> f64 {
    let jpm = (j as i32 + m as i32) as u32;
    let jmm = (j as i32 - m as i32) as u32;
    let jpn = (j as i32 + n as i32) as u32;
    let jmn = (j as i32 - n as i32) as u32;
    let prefactor =
        f64::sqrt((jpm.factorial() * jmm.factorial() * jpn.factorial() * jmn.factorial()) as f64);
    let s_min = isize::max(0, n - m) as usize;
    let s_max = isize::min(jpn as isize, jmm as isize) as usize;
    let sum: f64 = (s_min..=s_max)
        .map(|s| {
            ((-1.0f64).powi(m as i32 - n as i32 + s as i32)
                * (f64::cos(beta / 2.0)
                    .powi(2 * (j as i32) + n as i32 - m as i32 - 2 * (s as i32)))
                * (f64::sin(beta / 2.0).powi(m as i32 - n as i32 + 2 * s as i32)))
                / ((jpm - s as u32).factorial()
                    * (s as u32).factorial()
                    * ((m - n + s as isize) as u32).factorial()
                    * (jmm - s as u32).factorial()) as f64
        })
        .sum();
    prefactor * sum
}

pub fn wigner_d_matrix(
    alpha: f64,
    beta: f64,
    gamma: f64,
    j: usize,
    m: isize,
    n: isize,
) -> Complex64 {
    Complex64::cis(-(m as f64) * alpha)
        * small_wigner_d_matrix(beta, j, m, n)
        * Complex64::cis(-(n as f64) * gamma)
}
