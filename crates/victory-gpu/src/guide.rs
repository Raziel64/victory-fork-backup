//! The host's own ray-tracing structures for the programs that walk the
//! title's (`VICTORY_RAY_GUIDE`, `victory_shader::emit::ray_guide_level`):
//! where the layer calls [`core`], which does the work and names nothing of
//! the layer (the translator's software-adapter tests run that file itself).
//!
//! The switch's values are steps, each with all of the one before:
//!
//! - **1**: the walking programs are translated with the guided node test
//!   and given the guide table of no top level (`crate::memory::no_guide`):
//!   they walk as without the guide. Nothing here runs but a count of how
//!   often they are bound.
//! - **2**: before a walking program is given a top level of the title's,
//!   the host makes its own of it ([`tops`]): the pass
//!   `victory_shader::rays::TOP_PASS` writes the instances on the card and
//!   the driver builds from them. No bottom level is built, so every
//!   instance is a stand-in box: the hardware finds the instances a ray
//!   crosses and the title walks each bottom level itself.
//! - **3**: the title's static bottom levels are copied back, decoded and
//!   built into the pool ([`came`], [`went`], [`work`]); their instances
//!   come from the hardware as triangles.
//! - **4**: the bottom levels the title builds at run time (its moving and
//!   skinned objects) too: where its builder writes one ([`built`]) the
//!   pass `victory_shader::rays::BOTTOM_PASS` counts it on the card, and
//!   from then on writes the host's own of it and has the driver refit or
//!   build it in every list that walks, before that list's top levels.
//!
//! Where each call is recorded, and why there:
//!
//! - [`work`]: where a list first comes to title memory by address
//!   (`crate::memory::ready`): the copies back need the arenas in the state
//!   every list finds them in, before that list's own use of them. Builds
//!   and table entries go into the same list.
//! - [`tops`]: in `crate::draw::State::bind`, after the list's view heaps
//!   are set and before the walking program's own root signature: after the
//!   title's builder in the title's own order (it builds a top level before
//!   it has it walked), and in the same list as the walk, so the order of
//!   two lists never matters. Once per list and top level.
//!
//! A walking program is bound the host's table only when every top level it
//! is given was made in its own list ([`tops`]'s second answer); else the
//! table of no top level, and its walk is the console's. So nothing a list
//! could not make (no view slot, a fifth top level, a failed step) leaves a
//! walk with what another list made of another frame.
//!
//! The pool's cap is `VICTORY_RAY_GUIDE_MB` (128 unless set). A block of it
//! is made only while the card has room by the rule the arenas of title
//! memory by address go by (`crate::resource::by_address_room`), and the
//! system counts the blocks with everything else the process has on the
//! card. What finds no place stays a stand-in box.
//!
//! Memory let go here is used again or dropped by the layer's own rule for
//! what lists may still name (`crate::host::let_go_stamp`), on top of the
//! frames [`core`] counts; and nothing is made before the first list (the
//! stand-in box, the tables' first words) is known to have been submitted.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::rays;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};

mod core;

/// [`HostList::first_use`]'s keys of this module: the list has had the
/// moving bottom levels' pass; its walks go without the guide; and, by top
/// level, "made in this list" and "not made in this list".
const LIST_MOVING: usize = usize::MAX - 0x6f;
const LIST_PLAIN: usize = usize::MAX - 0x70;
fn list_top(address: u64) -> usize {
    usize::MAX - (9 << 33) - (address >> 8) as usize
}
fn list_top_failed(address: u64) -> usize {
    usize::MAX - (10 << 33) - (address >> 8) as usize
}

/// A bottom level the title builds at run time is let go when its builder
/// has not written it for so many frames.
const MOVING_IDLE: u64 = 120;

/// `VICTORY_RAY_GUIDE`'s step (0: off).
pub(crate) fn level() -> u8 {
    victory_shader::emit::ray_guide_level()
}

/// The pool's cap in MB (`VICTORY_RAY_GUIDE_MB`).
fn cap_mb() -> u64 {
    static MB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MB.get_or_init(|| std::env::var("VICTORY_RAY_GUIDE_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(128).min(2048))
}

struct State {
    guide: core::Guide,
    /// By view heap (its address) and entry of the table: the slot of the
    /// host top level's view there, and which making of it the view is of.
    views: HashMap<(usize, usize), (u32, u32)>,
    said: Said,
    /// Since the start: view slots taken; top levels not made for want of a
    /// slot; steps that failed; lists whose walks went without the guide.
    slots: u64,
    no_slot: u64,
    failed: u64,
    plain_lists: u64,
    /// The stages said once: the first list recorded, submitted, the first
    /// top level, static bottom level, count, moving bottom level.
    stages: [bool; 6],
}

/// For the report: when it was last said, and what was counted then.
struct Said {
    at: std::time::Instant,
    stats: core::Stats,
    counted: [[u32; 4]; core::TOPS],
    /// Since then: frames seen, the last one, microseconds of this module's
    /// work on the recording threads.
    frames: u64,
    frame: u64,
    micros: u64,
}

static GUIDE: Mutex<Option<Option<State>>> = Mutex::new(None);

/// Walking programs bound since the last line: with the host's table, and
/// with the table of no top level; and when the line was last said (ms).
static BOUND: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

fn make() -> Option<State> {
    let host = host()?;
    let made = (|| -> Result<core::Guide, String> {
        let device: ID3D12Device5 = host.device.cast().map_err(|e| e.message().to_string())?;
        let mut o5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        // SAFETY: the feature query writes the structure of the given size.
        let asked = unsafe { device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, (&raw mut o5).cast(), size_of_val(&o5) as u32) };
        if asked.is_err() || o5.RaytracingTier.0 < D3D12_RAYTRACING_TIER_1_1.0 {
            return Err(format!("the card has no inline ray queries (tier {})", o5.RaytracingTier.0));
        }
        let pass = crate::host::compile(rays::TOP_PASS, windows::core::s!("main"), windows::core::s!("cs_5_1")).map_err(|e| e.message().to_string())?;
        let shared = |bytes: u64| {
            let (res, data) = host.shared_buffer(bytes).ok_or("no shared buffer")?;
            Ok::<_, String>(core::Mapped { res, data, len: bytes })
        };
        // (A step that builds nothing has a pool of nothing.)
        let cfg = core::Config::of(if level() >= 3 { cap_mb() } else { 0 });
        let mut guide = core::Guide::new(&device, cfg, &pass, shared(8 << 20)?, shared(core::COUNTS_BYTES.next_multiple_of(4096))?)?;
        if level() >= 4 {
            // Without its pass the moving bottom levels stay stand-in boxes
            // and everything else goes on.
            let moving = crate::host::compile(rays::BOTTOM_PASS, windows::core::s!("main"), windows::core::s!("cs_5_1"))
                .map_err(|e| e.message().to_string())
                .and_then(|pass| guide.with_bottom_pass(&pass));
            if let Err(e) = moving {
                tracing::warn!(error = %e, "the ray guide: the pass for the bottom levels the title builds at run time is not made: they stay stand-in boxes");
            }
        }
        Ok(guide)
    })();
    match made {
        Ok(guide) => {
            for (buffer, what) in guide.buffers() {
                crate::host::name(buffer, || format!("the ray guide: {what}"));
            }
            tracing::info!(
                "the ray guide (VICTORY_RAY_GUIDE={}): the host makes its own ray-tracing structures of the title's; a pool of at most {} MB for bottom levels (VICTORY_RAY_GUIDE_MB)",
                level(),
                if level() >= 3 { cap_mb() } else { 0 }
            );
            Some(State {
                guide,
                views: HashMap::new(),
                said: Said { at: std::time::Instant::now(), stats: core::Stats::default(), counted: Default::default(), frames: 0, frame: 0, micros: 0 },
                slots: 0,
                no_slot: 0,
                failed: 0,
                plain_lists: 0,
                stages: [false; 6],
            })
        }
        Err(e) => {
            tracing::warn!(error = %e, "the ray guide is not made: the walking programs walk as without it");
            None
        }
    }
}

/// Runs `f` with the guide, made at the first call (step 2 and up).
fn with<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    if level() < 2 {
        return None;
    }
    let mut guard = GUIDE.lock().unwrap_or_else(|e| e.into_inner());
    let state = guard.get_or_insert_with(make).as_mut()?;
    let began = std::time::Instant::now();
    // What the layer waits for before it drops an object let go now, and
    // how far that wait has come: the guide's memory goes by the same.
    state.guide.clock(crate::host::let_go_stamp(), crate::host::let_go_done());
    let r = f(state);
    state.said.micros += began.elapsed().as_micros() as u64;
    Some(r)
}

/// What was recorded into `list` just now that later lists lean on (the
/// first list's box and tables, a first build): known to be there once
/// `list` is submitted.
fn arm(list: &HostList, guide: &mut core::Guide) {
    if let Some(token) = guide.take_sent() {
        list.defer_host(move || token.store(true, Ordering::Relaxed));
    }
}

/// The title's builder was given the packed structure at `address` (`len`
/// bytes unpacked, `header` its first two dwords): at a submission.
pub(crate) fn came(address: u64, len: u64, header: u64, frame: u64) {
    if level() >= 3 {
        with(|s| s.guide.came(address, len, header, frame));
    }
}

/// The title dropped its packed structure at `address`.
pub(crate) fn went(address: u64) {
    if level() >= 3 {
        with(|s| s.guide.went(address));
    }
}

/// The title's builder wrote a structure of its own at `address` (not a
/// packed one: one it builds at run time), in a list recorded in `frame`.
pub(crate) fn built(address: u64, frame: u64) {
    if level() >= 4 {
        with(|s| s.guide.seen(address, frame));
    }
}

/// Where `list` first comes to title memory by address: the first list has
/// the stand-in box built and the tables filled; then, once a frame, the
/// copies back of structures the builder has left alone and the builds of
/// those whose copies have come. `page`: a page's place on the card.
pub(crate) fn work(list: &HostList, frame: u64, page: &dyn Fn(u64) -> Option<(ID3D12Resource, u64)>) {
    with(|s| {
        let Ok(l) = list.list.cast::<ID3D12GraphicsCommandList4>() else { return };
        match s.guide.begin(&l, frame) {
            Ok(_) => arm(list, &mut s.guide),
            Err(e) => {
                s.failed += 1;
                once(&e);
                return;
            }
        }
        if level() >= 3 {
            s.guide.copy_back(&l, frame, page);
            if let Err(e) = s.guide.work(&l, frame, &mut room) {
                s.failed += 1;
                once(&e);
            }
            arm(list, &mut s.guide);
        }
        report(s, frame);
    });
}

/// A block of the pool only while the card has room, by the arenas' own
/// rule.
fn room(bytes: u64) -> bool {
    let fits = host().and_then(|h| h.video_memory()).is_none_or(|[budget, used, ..]| crate::resource::by_address_room(budget, used, bytes));
    if fits {
        GRANTED.fetch_add(bytes, Ordering::Relaxed);
    }
    fits
}

/// Bytes of the pool's blocks the card was found to have room for, so far.
static GRANTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What the pool may still take of the card (step 3 and above): its cap
/// less the blocks it was given room for. An arena of memory by address is
/// made only while the card has room for itself AND this
/// (`crate::memory`'s `card_room`): arenas come first in a run and would
/// take what the pool needs (run 425's arithmetic: `fps3.md` section 4).
pub(crate) fn reserve() -> u64 {
    reserve_of(level(), cap_mb(), GRANTED.load(Ordering::Relaxed))
}

fn reserve_of(level: u8, cap_mb: u64, granted: u64) -> u64 {
    if level >= 3 { (cap_mb << 20).saturating_sub(granted) } else { 0 }
}

#[cfg(test)]
mod room_tests {
    /// The pool's place in front of the arenas: nothing below step 3, the
    /// cap at first, less each block given room, never below nothing.
    #[test]
    fn the_pool_keeps_what_it_has_not_taken_yet() {
        use super::reserve_of;
        assert_eq!(reserve_of(2, 128, 0), 0);
        assert_eq!(reserve_of(3, 128, 0), 128 << 20);
        assert_eq!(reserve_of(4, 128, 48 << 20), 80 << 20);
        assert_eq!(reserve_of(3, 128, 200 << 20), 0);
    }
}

fn once(error: &str) {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !SAID.swap(true, std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(error, "the ray guide: a step failed (said once; counted in the guide's line); what it would have built stays a stand-in box");
    }
}

/// Before a walking program is given the title's top levels at `addresses`
/// in `list`: the host's own of each is made there (once per list and top
/// level), and before the first of a list the host's bottom levels of what
/// the title builds at run time (step 4). `heap` is the list's view heap,
/// `copies` the slot of the null view of its run of title memory's copies,
/// `directory` and `faults` what a walking program is bound for its looks.
///
/// The first answer: something was recorded; the list's compute root
/// signature and pipeline are then the pass's, and the caller binds its own
/// again. The second: every one of these top levels was made in this list,
/// so the program may be bound the host's table ([`table`]); else it must
/// be bound the table of no top level.
pub(crate) fn tops(list: &HostList, heap: &HostHeap, copies: u32, directory: u64, faults: u64, addresses: &[u64], frame: u64) -> (bool, bool) {
    with(|s| {
        let Ok(l) = list.list.cast::<ID3D12GraphicsCommandList4>() else { return (false, false) };
        if list.used(LIST_PLAIN) {
            return (false, false);
        }
        let mut recorded = false;
        let io = |slot: u32| core::TopIo { directory, faults, copies: heap.gpu_handle(copies), slot };
        if level() >= 4 && list.first_use(LIST_MOVING) {
            match s.guide.record_moving(&l, frame, io(0), MOVING_IDLE, &mut room) {
                Ok(done) => recorded = done,
                Err(e) => {
                    // What the host table says of the moving ones is not of
                    // this list: its walks are the title's own.
                    s.failed += 1;
                    s.plain_lists += 1;
                    once(&e);
                    list.first_use(LIST_PLAIN);
                    return (false, false);
                }
            }
            arm(list, &mut s.guide);
        }
        let mut guided = !addresses.is_empty();
        for &address in addresses {
            if address < 1 << 32 {
                // (No title memory by address: the node test finds no top
                // level there, with any table.)
                continue;
            }
            // Once per list and top level: the walks after the first in a
            // list find it made, or not made.
            if !list.first_use(list_top(address)) {
                guided &= !list.used(list_top_failed(address));
                continue;
            }
            let made = (|| {
                let used = match s.guide.top(address, frame) {
                    Ok(Some(used)) => used,
                    Ok(None) => return false,
                    Err(e) => {
                        s.failed += 1;
                        once(&e);
                        return false;
                    }
                };
                // Its view in this heap: a slot of its own for every making
                // of it (a list recorded earlier may still name the one
                // before). Held for the heap's life, as the copies' run is.
                let key = (std::ptr::from_ref(heap) as usize, used.index);
                let slot = match s.views.get(&key) {
                    Some(&(slot, generation)) if generation == used.generation => slot,
                    _ => {
                        let Some(slot) = heap.dynamic(1) else {
                            s.no_slot += 1;
                            return false;
                        };
                        heap.structure_at(slot, used.location);
                        s.views.insert(key, (slot, used.generation));
                        s.slots += 1;
                        slot
                    }
                };
                s.guide.record_top(&l, &used, io(slot));
                true
            })();
            if made {
                recorded = true;
            } else {
                list.first_use(list_top_failed(address));
                guided = false;
            }
        }
        (recorded, guided)
    })
    .unwrap_or((false, false))
}

/// The guide table a walking program is bound in `frame`: the host's when
/// every top level the program is given was made in its list (`guided`,
/// [`tops`]) and the first list has surely run, else the table of no top
/// level.
pub(crate) fn table(frame: u64, guided: bool) -> Option<u64> {
    // The title pulls instances up into its top levels (seen in its own
    // instance descriptions, `crate::raycounts`): the pass's "one node an
    // instance" does not hold, so no walk is given the host's table.
    let guided = guided && !crate::raycounts::pulled_up();
    let host = if guided { with(|s| s.guide.table(frame)).flatten() } else { None };
    BOUND[usize::from(host.is_none())].fetch_add(1, Ordering::Relaxed);
    if level() == 1 {
        bound_line();
    }
    host.or_else(crate::memory::no_guide)
}

/// Step 1 has no other line: every five seconds or so, how often the
/// walking programs were bound (it proves they load and run).
fn bound_line() {
    let now = crate::host::now_ms();
    let last = BOUND[2].load(Ordering::Relaxed);
    if now < last + 5000 || BOUND[2].compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed).is_err() {
        return;
    }
    let plain = BOUND[1].swap(0, Ordering::Relaxed);
    BOUND[0].store(0, Ordering::Relaxed);
    if last != 0 {
        tracing::info!(
            "the ray guide (VICTORY_RAY_GUIDE=1): programs that walk with the guided node test were bound {plain} times in {:.1} s, each with the table \
             of no top level: they walk as without the guide; nothing is built",
            (now - last) as f64 / 1000.0
        );
    }
}

/// Every five seconds or so, what the guide holds and did; and once each,
/// the stages as they are reached.
fn report(s: &mut State, frame: u64) {
    if s.said.frame != frame {
        (s.said.frame, s.said.frames) = (frame, s.said.frames + 1);
    }
    let now = s.guide.stats;
    let (born, live) = s.guide.state(frame);
    let reached = [born, live, now.tops > 0, now.built > 0, now.census_asked > 0, now.moving_builds > 0];
    const STAGES: [&str; 6] = [
        "the first list is recorded: the stand-in box's build and the two tables' first words",
        "the first list was submitted and its frames have passed: top levels are made from now on",
        "the first host top level was recorded (the pass on the card, then the driver's build)",
        "the first static bottom level was copied back, decoded and its build recorded",
        "the first count of a bottom level the title builds at run time was asked of the card",
        "the first bottom level the title builds at run time was counted and its host build recorded",
    ];
    for (k, &is) in reached.iter().enumerate() {
        if is && !s.stages[k] {
            s.stages[k] = true;
            tracing::info!("the ray guide (VICTORY_RAY_GUIDE={}), stage {} of 6 reached in frame {frame}: {}", level(), k + 1, STAGES[k]);
        }
    }
    let secs = s.said.at.elapsed().as_secs_f64();
    if secs < 5.0 {
        return;
    }
    let then = s.said.stats;
    let (built, waiting, starved, triangles) = s.guide.standing();
    let (blocks, held, other) = s.guide.bytes();
    let frames = s.said.frames.max(1) as f64;
    let tops = (now.tops - then.tops) as f64;
    // What the pass counted, a top level made: with a host structure, as
    // boxes, left out.
    let mut counted = [0f64; 3];
    let mut instances = Vec::new();
    for index in 0..core::TOPS {
        let c = s.guide.counted(index);
        for k in 0..3 {
            counted[k] += f64::from(c[k + 1].wrapping_sub(s.said.counted[index][k + 1]));
        }
        if c[0] != 0 && c[0] != !0 {
            instances.push(c[0]);
        }
        s.said.counted[index] = c;
    }
    let per_top = |x: f64| if tops > 0.0 { x / tops } else { 0.0 };
    let (bound, plain) = (BOUND[0].swap(0, Ordering::Relaxed), BOUND[1].swap(0, Ordering::Relaxed));
    let (going, buffers) = s.guide.waiting_to_go();
    if level() >= 3 {
        let all = now.opaque + now.not_opaque;
        tracing::info!(
            "the ray guide, what the static bottom levels decoded since the start hold: triangles marked opaque {} and not opaque {} ({:.1} %              opaque); box nodes of 16 bits {}, of 32 bits {}, nodes of other kinds {}; structures stored in a space of their own {}; the pool              keeps {} MB of the card in front of further arenas",
            now.opaque,
            now.not_opaque,
            if all > 0 { 100.0 * now.opaque as f64 / all as f64 } else { 0.0 },
            now.boxes[0],
            now.boxes[1],
            now.other_nodes,
            now.own_space,
            reserve() >> 20
        );
    }
    tracing::info!(
        "the ray guide (VICTORY_RAY_GUIDE={}): host bottom levels built and held {built} ({triangles} triangles, {:.1} MB of structures in {} MB of \
         blocks, the pool's cap {} MB), waiting {waiting}, turned down for room {starved}; since the start came {}, went {}, built {}, not sound {}, \
         not there {}, turned down for room {} times; since the last line {:.1} builds a frame in {} frames with a build ({:.0} triangles a build \
         frame, {} pages copied back); top levels made a frame {:.1} (walks the table had no entry for: {}), the title's instances in them now \
         {instances:?}; instances a top level made: as triangles {:.0}, as stand-in boxes {:.0}, left out {:.0}; the guide's own {:.1} MB beside \
         the pool; its work on the recording threads {:.0} microseconds a frame over {} frames; walking programs bound with the host's table \
         {bound}, with the table of no top level {plain}; since the start: view slots taken {}, top levels not made for want of a slot {}, steps \
         that failed {}, lists whose walks went without the guide {}, the first list recorded again {}, no room on the way {} times; let go and \
         waiting for its lists {going} ({buffers} buffers)",
        level(),
        held as f64 / 1048576.0,
        blocks >> 20,
        if level() >= 3 { cap_mb() } else { 0 },
        now.came,
        now.went,
        now.built,
        now.unsound,
        now.not_there,
        now.no_room,
        (now.built - then.built) as f64 / frames,
        now.build_frames - then.build_frames,
        (now.triangles - then.triangles) as f64 / (now.build_frames - then.build_frames).max(1) as f64,
        now.pages - then.pages,
        tops / frames,
        now.tops_over - then.tops_over,
        per_top(counted[0]),
        per_top(counted[1]),
        per_top(counted[2]),
        other as f64 / 1048576.0,
        s.said.micros as f64 / frames,
        s.said.frames,
        s.slots,
        s.no_slot,
        s.failed,
        s.plain_lists,
        now.begun_again,
        now.ring_full,
    );
    if level() >= 4 {
        let (held, triangles, bytes, waiting) = s.guide.moving_standing();
        let lists = (now.moving_lists - then.moving_lists).max(1) as f64;
        let taken = s.guide.counted(core::TOPS);
        tracing::info!(
            "the ray guide, bottom levels the title builds at run time: the host holds its own of {held} ({triangles} triangles, {:.1} MB of the \
             pool), {waiting} wait for their count; since the last line the pass ran in {:.1} lists a frame with {:.1} rows a list: a list \
             {:.1} refits and {:.1} builds of {:.0} triangles, {:.1} left boxes for the list's marks; since the start: counts asked {}, no \
             bottom level there {}, counted again because a triangle did not fit {}, turned down for room {}, let go {}; the pass offered {} \
             and turned down {} structures",
            bytes as f64 / 1048576.0,
            (now.moving_lists - then.moving_lists) as f64 / frames,
            (now.moving_rows - then.moving_rows) as f64 / lists,
            (now.moving_refits - then.moving_refits) as f64 / lists,
            (now.moving_builds - then.moving_builds) as f64 / lists,
            (now.moving_triangles - then.moving_triangles) as f64 / lists,
            (now.moving_over - then.moving_over) as f64 / lists,
            now.census_asked,
            now.census_refused,
            now.moving_misfit,
            now.moving_refused,
            now.moving_gone,
            taken[0],
            taken[1],
        );
    }
    s.said = Said { at: std::time::Instant::now(), stats: now, counted: s.said.counted, frames: 0, frame, micros: 0 };
}
