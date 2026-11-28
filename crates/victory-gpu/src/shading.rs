//! The title's coarse shading, given to the host as the console's list takes
//! it (`VICTORY_SHADING_RATE=0`: ignored, every draw shaded per pixel, as
//! before; `=turns`: given and ignored by turns of twenty seconds, to compare
//! the two in one run).
//!
//! The console's list has the PC's `RSSetShadingRate(base rate, two
//! combiners)` with the PC's values (the console interface's own header: the
//! rates 1x1, 1x2, 2x1, 2x2, 2x4, 4x2, 4x4 as 0, 1, 4, 5, 6, 9, 10; the
//! combiners passthrough, override, min, max, sum as 0 to 4). The title's
//! state flush calls it whenever the wanted rate differs from the last one
//! set, some tens of times a frame; the layer used to drop the call, so
//! draws the title asks to shade once for two or four pixels were shaded for
//! every pixel. The title sets no shading rate image.
//!
//! A new host list starts at 1x1 with both combiners at passthrough, as the
//! console's does, so [`crate::draw::State::replay`] sets the rate again on a
//! host list that continues the title's list.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::host::HostList;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Off,
    On,
    /// Given and not given by turns, to compare the two in one run at the
    /// same scenes (`VICTORY_SHADING_RATE=turns`; the log says when).
    ByTurns,
}

fn mode() -> Mode {
    static MODE: OnceLock<Mode> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_SHADING_RATE").as_deref() {
        Ok("0") => Mode::Off,
        Ok("turns") => Mode::ByTurns,
        _ => Mode::On,
    })
}

/// With `VICTORY_SHADING_RATE=turns`: this long given, this long not.
const TURN_MS: u64 = 20_000;

/// Milliseconds since the first asking.
fn now_ms() -> u64 {
    static START: OnceLock<std::time::Instant> = OnceLock::new();
    START.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// Whether a turn that began `ms` after the start is one in which the rate
/// is given (the first is).
const fn turn_gives(ms: u64) -> bool {
    ms / TURN_MS % 2 == 0
}

/// Whether the title's rate is given to the host now.
pub(crate) fn on() -> bool {
    match mode() {
        Mode::Off => false,
        Mode::On => true,
        Mode::ByTurns => turn_gives(now_ms()),
    }
}

/// By turns: a line whenever the turn changes, from the title's own calls.
fn turned() {
    if mode() != Mode::ByTurns {
        return;
    }
    /// The turn the last line was for, plus one.
    static SAID: AtomicU64 = AtomicU64::new(0);
    let turn = now_ms() / TURN_MS + 1;
    if SAID.swap(turn, Ordering::Relaxed) != turn {
        tracing::info!(
            given = turn_gives((turn - 1) * TURN_MS),
            seconds = TURN_MS / 1000,
            "the title's shading rate is given to the host by turns (VICTORY_SHADING_RATE=turns)"
        );
    }
}

/// The rates a list takes, in the order of [`COUNTS`].
const RATES: [(u32, &str); 7] =
    [(0, "1x1"), (1, "1x2"), (4, "2x1"), (5, "2x2"), (6, "2x4"), (9, "4x2"), (10, "4x4")];

/// The most a combiner can be (sum).
const COMBINERS: u32 = 4;

/// What the host list is told for the title's `rate` and `combiners`: none
/// for values no list takes; a rate over 2x2 as 2x2 in that direction on a
/// host without the larger rates; no combiners on a host that has only a
/// rate a draw (`tier` 1).
pub(crate) fn told(rate: u32, combiners: [u32; 2], tier: u32, larger: bool) -> Option<(u32, Option<[u32; 2]>)> {
    if tier == 0 || !RATES.iter().any(|r| r.0 == rate) || combiners.iter().any(|&c| c > COMBINERS) {
        return None;
    }
    // Two bits a direction: the logarithm of the pixels shaded as one.
    let (x, y) = (rate >> 2, rate & 3);
    let rate = if larger { rate } else { (x.min(1) << 2) | y.min(1) };
    Some((rate, (tier >= 2).then_some(combiners)))
}

/// Calls by rate (as [`RATES`]), calls with a combiner that is not
/// passthrough, calls with a value no list takes, and the title's draws
/// under a rate other than 1x1 and all of them.
static COUNTS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
static COMBINED: AtomicU64 = AtomicU64::new(0);
static REFUSED: AtomicU64 = AtomicU64::new(0);
static COARSE_DRAWS: AtomicU64 = AtomicU64::new(0);
static DRAWS: AtomicU64 = AtomicU64::new(0);

/// A draw of the title's is bound under `rate` (none: never set, 1x1).
#[inline]
pub(crate) fn drawn(rate: Option<[u32; 3]>) {
    DRAWS.fetch_add(1, Ordering::Relaxed);
    if rate.is_some_and(|r| r[0] != 0) {
        COARSE_DRAWS.fetch_add(1, Ordering::Relaxed);
    }
}

/// The counts as a line, about every five seconds, from the calls
/// themselves (no call, no line).
fn line() {
    static LAST: AtomicU64 = AtomicU64::new(0);
    let now = now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 5_000 || LAST.compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    let by: Vec<String> = RATES
        .iter()
        .zip(&COUNTS)
        .map(|(r, c)| (r.1, c.swap(0, Ordering::Relaxed)))
        .filter(|c| c.1 != 0)
        .map(|(name, n)| format!("{name} {n}"))
        .collect();
    tracing::info!(
        given_now = on(),
        seconds = (now - last) as f64 / 1000.0,
        calls = %by.join(", "),
        combined = COMBINED.swap(0, Ordering::Relaxed),
        refused = REFUSED.swap(0, Ordering::Relaxed),
        coarse_draws = COARSE_DRAWS.swap(0, Ordering::Relaxed),
        draws = DRAWS.swap(0, Ordering::Relaxed),
        "shading rate the title asked for since the last line (VICTORY_SHADING_RATE=0: not given to the host): calls by rate, calls with a combiner, calls with a value no list takes, the title's draws bound under a rate other than 1x1 and all of them"
    );
}

/// The title's call is counted.
pub(crate) fn asked(rate: u32, combiners: [u32; 2]) {
    match RATES.iter().position(|r| r.0 == rate) {
        Some(i) if combiners.iter().all(|&c| c <= COMBINERS) => {
            COUNTS[i].fetch_add(1, Ordering::Relaxed);
            if combiners != [0, 0] {
                COMBINED.fetch_add(1, Ordering::Relaxed);
            }
        }
        _ => {
            REFUSED.fetch_add(1, Ordering::Relaxed);
        }
    }
    turned();
    line();
}

/// `list` shades what follows at `rate` with `combiners`, where the switch is
/// on and the host can.
pub(crate) fn set(list: &HostList, rate: u32, combiners: [u32; 2]) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || {
        format!("SHADING RATE {rate:#x} combiners {combiners:?}")
    });
    if !on() {
        return;
    }
    let Some((tier, larger)) = crate::host::host().map(|h| h.shading_rates()) else {
        return;
    };
    let Some((rate, combiners)) = told(rate, combiners, tier, larger) else {
        return;
    };
    if let Ok(l5) = list.list.cast::<ID3D12GraphicsCommandList5>() {
        let combiners = combiners.map(|c| c.map(|c| D3D12_SHADING_RATE_COMBINER(c as i32)));
        // SAFETY: an open list on a device with variable-rate shading; a
        // rate and combiners of the interface's own values; the combiners
        // live across the call.
        unsafe {
            l5.RSSetShadingRate(D3D12_SHADING_RATE(rate as i32), combiners.as_ref().map(|c| c.as_ptr()));
        }
    }
}

/// `list` shades every pixel again: before a pass of the layer's own on a
/// list that may hold the title's rate.
pub(crate) fn plain(list: &ID3D12GraphicsCommandList) {
    if !on() || crate::host::host().is_none_or(|h| h.shading_rates().0 == 0) {
        return;
    }
    if let Ok(l5) = list.cast::<ID3D12GraphicsCommandList5>() {
        // SAFETY: an open list on a device with variable-rate shading.
        unsafe { l5.RSSetShadingRate(D3D12_SHADING_RATE_1X1, None) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The call as the layer makes it, on host lists: set, set again with
    /// combiners, every pixel for a pass of our own, and set again on the
    /// host list that continues the title's. The lists are closed (a list
    /// that was told something it does not take fails there), and the device
    /// still stands.
    #[test]
    fn host_lists_take_the_rate_and_take_it_again_when_continued() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let (tier, larger) = host.shading_rates();
        eprintln!("host variable-rate shading: tier {tier}, larger rates {larger}, given {}", on());
        let mut state = crate::draw::State::default();
        let first = host.new_list().unwrap();
        state.set_shading_rate(&first, 5, [0, 0]);
        state.set_shading_rate(&first, 10, [1, 2]);
        plain(&first.list);
        let second = host.new_list().unwrap();
        state.replay(&second);
        for list in [&first, &second] {
            // SAFETY: open lists of this test's own, closed once.
            unsafe { list.list.Close() }.expect("the list took what it was told");
        }
        // SAFETY: a live device.
        unsafe { host.device.GetDeviceRemovedReason() }.expect("the device stands");
    }

    #[test]
    fn turns_give_first_and_then_every_other_one() {
        assert!(turn_gives(0));
        assert!(turn_gives(TURN_MS - 1));
        assert!(!turn_gives(TURN_MS));
        assert!(!turn_gives(2 * TURN_MS - 1));
        assert!(turn_gives(2 * TURN_MS));
    }

    #[test]
    fn a_rate_is_told_as_the_title_gave_it() {
        // The seven rates with any combiners, on a host with all of it.
        for (rate, _) in RATES {
            assert_eq!(told(rate, [0, 0], 2, true), Some((rate, Some([0, 0]))));
            assert_eq!(told(rate, [1, 4], 2, true), Some((rate, Some([1, 4]))));
        }
    }

    #[test]
    fn values_no_list_takes_are_not_told() {
        for rate in [2, 3, 7, 8, 11, 0x10, u32::MAX] {
            assert_eq!(told(rate, [0, 0], 2, true), None, "rate {rate}");
        }
        assert_eq!(told(5, [5, 0], 2, true), None);
        assert_eq!(told(5, [0, 9], 2, true), None);
    }

    #[test]
    fn a_host_without_it_is_told_nothing() {
        assert_eq!(told(5, [0, 0], 0, false), None);
    }

    #[test]
    fn a_host_with_a_rate_a_draw_only_gets_no_combiners() {
        assert_eq!(told(5, [1, 1], 1, true), Some((5, None)));
    }

    #[test]
    fn larger_rates_stop_at_two_by_two_where_the_host_lacks_them() {
        // 2x4, 4x2, 4x4 -> 2x2; 1x2, 2x1, 2x2, 1x1 as they are.
        for (rate, want) in [(6, 5), (9, 5), (10, 5), (1, 1), (4, 4), (5, 5), (0, 0)] {
            assert_eq!(told(rate, [0, 0], 2, false), Some((want, Some([0, 0]))), "rate {rate}");
        }
    }
}
