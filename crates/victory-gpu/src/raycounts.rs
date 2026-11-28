//! What the title's ray-tracing structures are like, counted from the PC
//! side: one line with [`crate::memory`]'s report, every 10 s
//! (`VICTORY_RAY_COUNTS=0`: off, nothing is read or counted).
//!
//! It answers what the host's own ray tracing must know before it is built:
//! whether bottom levels keep their triangles in a scaled, moved or turned
//! space of their own, whether instances are pulled up into the top level
//! as 4 or 16 nodes each, and where the structures lie.
//!
//! **Read only, and only what the layer reads already**:
//!
//! - the first 64 bytes of a packed bottom level in title memory, once, when
//!   the builder comes to it (`Memory::land`, which has just read its first
//!   8; a packed structure begins on 256 bytes and is at least 128 long, so
//!   the 64 lie inside it and inside its first page);
//! - a list's upload memory with the instance descriptions of a top-level
//!   build (`Memory::named_in_upload`, which walks the same block), one block
//!   in [`EVERY`] frames and [`SCAN`] bytes of it at most;
//! - the addresses of the structures a walking dispatch is given, in the
//!   same frames, [`WALKED`] of them at most.
//!
//! No pointer found in any of them is followed: an address is only looked up
//! in tables (the bottom levels counted here, the directory of memory by
//! address). What is on the card alone is NOT counted here, and the line
//! says so: a top level's header and instance nodes, every unpacked node
//! (16-bit or 32-bit boxes, procedural nodes), the triangles' opaque flags,
//! and bottom levels the title builds at run time.
//!
//! The header's fields (dwords): 0 the version (low half) and flags
//! (0x8000_0000 a bottom level, 0x0100_0000 packed, 0x0008_0000 and
//! 0x0001_0000 compressed, 0x0002_0000 built offline, 0x0004_0000 can be
//! updated, 0x00F0_0000 a k-DOP's size); 1 the size in 64-byte nodes less
//! one; 3 the number of elements; 15 the space of its own: 0x8000_0000
//! scaled, 0x4000_0000 moved, 0x3F00_0000 turned (which of 64 rotations).
//! An instance description is the host's: twelve floats, the id and mask,
//! the contribution and flags (the console's own: 0x40 pulled up as 4 nodes,
//! 0xC0 as 16), the structure's address.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// An instance description with the console's pull-up flags was seen.
static PULLED: AtomicBool = AtomicBool::new(false);

/// Whether the title was seen to pull instances up into a top level (as 4
/// or 16 instance nodes each). The ray guide's top-level pass takes one
/// node per instance, so from then on the guide gives no walk its table
/// (`crate::guide::table`): the walking programs walk as without it.
/// Never true with `VICTORY_RAY_COUNTS=0`: nothing is looked at then.
pub(crate) fn pulled_up() -> bool {
    PULLED.load(Ordering::Relaxed)
}

/// A header's bytes.
pub(crate) const HEADER: usize = 64;
/// An instance description's bytes.
const DESC: usize = 64;
/// One block of instance descriptions is looked at in so many frames.
const EVERY: u64 = 16;
/// Of one block so many bytes at most (16,384 descriptions).
const SCAN: usize = 1 << 20;
/// Bottom levels remembered by address, at most.
const KNOWN: usize = 1 << 15;
/// Walked structures remembered by address, at most.
const WALKED: usize = 256;
/// Different masks and flag bytes told apart, at most.
const KINDS: usize = 12;

/// Whether the counting is on.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RAY_COUNTS").is_none_or(|v| v != "0"))
}

/// Where a structure's first page is, for shaders.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Place {
    /// A page of an arena.
    Arena,
    /// A title buffer's copy (or a window of title memory).
    Buffer,
    /// No copy.
    Nowhere,
}

/// What a header says, as far as it is counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Head {
    version: u32,
    flags: u32,
    nodes: u32,
    elements: u32,
    space: u32,
}

impl Head {
    fn own_space(&self) -> bool {
        self.space & 0xFF00_0000 != 0
    }
}

/// The header in `bytes`, if they are a whole one.
fn head(bytes: &[u8]) -> Option<([u32; 16], Head)> {
    let bytes = bytes.get(..HEADER)?;
    let w: [u32; 16] = std::array::from_fn(|k| u32::from_le_bytes([bytes[4 * k], bytes[4 * k + 1], bytes[4 * k + 2], bytes[4 * k + 3]]));
    Some((w, Head { version: w[0] & 0xFFFF, flags: w[0] & 0xFFFF_0000, nodes: w[1], elements: w[3], space: w[15] }))
}

#[derive(Default)]
pub(crate) struct Counts {
    /// Bottom levels by address: whether they have a space of their own.
    known: HashMap<u64, bool>,
    /// Packed structures the builder came to; of them: not a whole header,
    /// another version than 6, no bottom level; built offline, can be
    /// updated, with a k-DOP, either compressed flag.
    packed: u64,
    short: u64,
    version: u64,
    top: u64,
    offline: u64,
    updated: u64,
    kdop: u64,
    compressed: u64,
    /// Scaled, moved, turned, any of the three.
    space: [u64; 4],
    elements: (u64, u32),
    nodes: u64,
    place: [u64; 3],
    /// The first header with a space of its own, else the last one seen.
    example: Option<([u32; 16], bool)>,
    /// Blocks of instance descriptions looked at, descriptions, of them 64
    /// bytes behind the one before; words that look like an address but
    /// have no description before them.
    next: u64,
    lists: u64,
    descs: u64,
    runs: u64,
    odd: u64,
    most: u64,
    /// Flag bits 1, 2, 4, 8; pulled up as 4, as 16; 0x80 alone; 0x10 or 0x20.
    flags: [u64; 8],
    /// Flag bytes and masks seen, with their counts.
    bytes: [HashMap<u8, u64>; 2],
    /// Their bottom levels: counted here (of them with a space of their
    /// own), not counted here by place.
    theirs: (u64, u64),
    others: [u64; 3],
    /// Structures walking dispatches were given, by place.
    walked: HashSet<u64>,
    walked_place: [u64; 3],
}

impl Counts {
    /// A packed structure the builder came to: the first [`HEADER`] bytes
    /// title memory has at `address` (zeros where it could not be read).
    pub(crate) fn structure(&mut self, address: u64, bytes: &[u8], place: Place) {
        self.packed += 1;
        let Some((words, h)) = head(bytes) else {
            self.short += 1;
            return;
        };
        if h.version != 6 {
            self.version += 1;
            return;
        }
        if h.flags & 0x8000_0000 == 0 {
            self.top += 1;
            return;
        }
        self.place[place as usize] += 1;
        self.offline += u64::from(h.flags & 0x0002_0000 != 0);
        self.updated += u64::from(h.flags & 0x0004_0000 != 0);
        self.kdop += u64::from(h.flags & 0x00F0_0000 != 0);
        self.compressed += u64::from(h.flags & 0x0009_0000 != 0);
        self.space[0] += u64::from(h.space & 0x8000_0000 != 0);
        self.space[1] += u64::from(h.space & 0x4000_0000 != 0);
        self.space[2] += u64::from(h.space & 0x3F00_0000 != 0);
        self.space[3] += u64::from(h.own_space());
        self.elements = (self.elements.0 + u64::from(h.elements), self.elements.1.max(h.elements));
        self.nodes += u64::from(h.nodes);
        if self.known.len() < KNOWN || self.known.contains_key(&address) {
            self.known.insert(address, h.own_space());
        }
        if !self.example.is_some_and(|e| e.1) {
            self.example = Some((words, h.own_space()));
        }
    }

    /// Whether a block of instance descriptions is looked at in `frame`.
    pub(crate) fn wants_list(&mut self, frame: u64) -> bool {
        if frame < self.next && self.next - frame <= EVERY {
            return false;
        }
        self.next = frame.wrapping_add(EVERY);
        true
    }

    /// A block of upload memory a builder dispatch names: the instance
    /// descriptions in it. `place`: where the structure at an address is.
    pub(crate) fn list(&mut self, block: &[u8], place: impl Fn(u64) -> Place) {
        let block = &block[..block.len().min(SCAN)];
        self.lists += 1;
        let (mut last, mut found) = (usize::MAX, 0u64);
        let mut at = 0;
        while at + 8 <= block.len() {
            let word = u64::from_le_bytes(std::array::from_fn(|k| block[at + k]));
            let here = at;
            at += 8;
            // (What `named_in_upload` takes for a structure's address.)
            if word & 0xFF != 0 || !(1 << 32..1 << 37).contains(&word) {
                continue;
            }
            // The description ends with the address.
            let Some(desc) = (here + 8).checked_sub(DESC).and_then(|from| block.get(from..here + 8)) else {
                self.odd += 1;
                continue;
            };
            let float = |k: usize| f32::from_bits(u32::from_le_bytes([desc[4 * k], desc[4 * k + 1], desc[4 * k + 2], desc[4 * k + 3]]));
            // A matrix: every number nothing or of a size a transform has
            // (addresses and counts read as floats are tiny or huge), and
            // not all nothing.
            let sized = |x: f32| x == 0.0 || (1e-20..1e20).contains(&x.abs());
            if !(0..12).all(|k| sized(float(k))) || (0..12).all(|k| float(k) == 0.0) {
                self.odd += 1;
                continue;
            }
            found += 1;
            self.runs += u64::from(last != usize::MAX && here == last + DESC);
            last = here;
            let (mask, flags) = (desc[51], desc[55]);
            for (n, bit) in [1u8, 2, 4, 8].into_iter().enumerate() {
                self.flags[n] += u64::from(flags & bit != 0);
            }
            self.flags[4] += u64::from(flags & 0xC0 == 0x40);
            self.flags[5] += u64::from(flags & 0xC0 == 0xC0);
            if flags & 0x40 != 0 && !PULLED.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "ray counts: an instance description with the pull-up flags (flags byte {flags:#04x}) was seen: a top level then has more                      instance nodes than instances, which the ray guide's pass does not know; from here on the guide gives no walk its table                      (the walking programs walk as with VICTORY_RAY_GUIDE=1)"
                );
            }
            self.flags[6] += u64::from(flags & 0xC0 == 0x80);
            self.flags[7] += u64::from(flags & 0x30 != 0);
            for (seen, byte) in self.bytes.iter_mut().zip([flags, mask]) {
                if seen.len() < KINDS || seen.contains_key(&byte) {
                    *seen.entry(byte).or_insert(0) += 1;
                }
            }
            match self.known.get(&word) {
                Some(&own) => self.theirs = (self.theirs.0 + 1, self.theirs.1 + u64::from(own)),
                None => self.others[place(word) as usize] += 1,
            }
        }
        self.descs += found;
        self.most = self.most.max(found);
    }

    /// Whether the structures walking dispatches are given are counted in
    /// `frame`.
    pub(crate) fn wants_walked(&self, frame: u64) -> bool {
        frame.is_multiple_of(EVERY) && self.walked.len() < WALKED
    }

    /// A structure a walking dispatch was given.
    pub(crate) fn walked(&mut self, address: u64, place: Place) {
        if self.walked.len() < WALKED && self.walked.insert(address) {
            self.walked_place[place as usize] += 1;
        }
    }

    /// The line, once anything was counted. Everything in it is since the
    /// run began.
    pub(crate) fn line(&self) -> Option<String> {
        if self.packed == 0 && self.lists == 0 && self.walked.is_empty() {
            return None;
        }
        let kinds = |seen: &HashMap<u8, u64>| {
            let mut all: Vec<_> = seen.iter().collect();
            all.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            let text: Vec<_> = all.iter().take(6).map(|(byte, n)| format!("{byte:#04x} x{n}")).collect();
            if text.is_empty() { "none".to_string() } else { text.join(", ") }
        };
        let example = match &self.example {
            Some((w, _)) => w.iter().map(|w| format!("{w:08x}")).collect::<Vec<_>>().join(" "),
            None => "none".to_string(),
        };
        let bottoms = self.packed - self.short - self.version - self.top;
        Some(format!(
            "ray counts (since the run began; PC side only): packed structures {} (not readable {}, another version {}, no bottom level {}); \
             bottom levels {}: built offline {}, can be updated {}, with a k-DOP {}, compressed {}; \
             IN A SPACE OF THEIR OWN {} (scaled {}, moved {}, turned {}); elements {} in all, {} at most, nodes {}; \
             their first page in an arena {}, in a title buffer {}, nowhere {}; \
             instance lists looked at {} (one in {} frames): {} instances, {} at most in one, {} in runs of 64 bytes, {} address-like words turned down; \
             flags: no culling {}, counter-clockwise {}, force opaque {}, force not opaque {}, PULLED UP AS 4 NODES {}, AS 16 NODES {}, 0x80 alone {}, 0x10 or 0x20 {}; \
             flag bytes {}; masks {}; \
             their bottom levels: packed ones counted here {} (in a space of their own {}), others in an arena {}, in a title buffer {}, nowhere {}; \
             structures given to walking dispatches {}: in an arena {}, in a title buffer {}, nowhere {}; \
             on the card alone, not counted: top-level headers and instance nodes, node kinds, opaque flags, run-time bottom levels; \
             a header: {}",
            self.packed,
            self.short,
            self.version,
            self.top,
            bottoms,
            self.offline,
            self.updated,
            self.kdop,
            self.compressed,
            self.space[3],
            self.space[0],
            self.space[1],
            self.space[2],
            self.elements.0,
            self.elements.1,
            self.nodes,
            self.place[0],
            self.place[1],
            self.place[2],
            self.lists,
            EVERY,
            self.descs,
            self.most,
            self.runs,
            self.odd,
            self.flags[0],
            self.flags[1],
            self.flags[2],
            self.flags[3],
            self.flags[4],
            self.flags[5],
            self.flags[6],
            self.flags[7],
            kinds(&self.bytes[0]),
            kinds(&self.bytes[1]),
            self.theirs.0,
            self.theirs.1,
            self.others[0],
            self.others[1],
            self.others[2],
            self.walked.len(),
            self.walked_place[0],
            self.walked_place[1],
            self.walked_place[2],
            example
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(flags: u32, nodes: u32, elements: u32, space: u32) -> Vec<u8> {
        let mut w = [0u32; 16];
        (w[0], w[1], w[3], w[15]) = (flags, nodes, elements, space);
        w.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn desc(flags: u8, mask: u8, structure: u64) -> Vec<u8> {
        let mut w = [0u32; 16];
        (w[0], w[5], w[10]) = (1f32.to_bits(), 1f32.to_bits(), 1f32.to_bits());
        (w[12], w[13]) = (7 | u32::from(mask) << 24, 3 | u32::from(flags) << 24);
        (w[14], w[15]) = (structure as u32, (structure >> 32) as u32);
        w.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    const A: u64 = 0x2_4000_0100;
    const B: u64 = 0x2_4100_0000;

    #[test]
    fn headers_are_counted_by_what_they_say() {
        let mut c = Counts::default();
        // As the title ships them: a packed bottom level built offline.
        c.structure(A, &header(0x8102_0006, 0x9a1a, 1200, 0), Place::Arena);
        // One scaled and moved, one turned (rotation 5).
        c.structure(B, &header(0x8102_0006, 40, 12, 0xC07F_7F7F), Place::Arena);
        c.structure(B + 0x100, &header(0x8106_0006, 40, 12, 0x0500_0000), Place::Buffer);
        // Not whole, another version, a top level: counted as such, and no more.
        c.structure(B + 0x200, &header(0x8102_0006, 40, 12, 0)[..63], Place::Arena);
        c.structure(B + 0x300, &[], Place::Arena);
        c.structure(B + 0x400, &header(0x8102_0005, 40, 12, 0x8000_0000), Place::Arena);
        c.structure(B + 0x500, &header(0x0102_0006, 40, 12, 0x8000_0000), Place::Arena);
        assert_eq!((c.packed, c.short, c.version, c.top), (7, 2, 1, 1));
        assert_eq!((c.offline, c.updated, c.kdop, c.compressed), (3, 1, 0, 0));
        assert_eq!(c.space, [1, 1, 1, 2]);
        assert_eq!((c.elements, c.nodes), ((1224, 1200), 0x9a1a + 80));
        assert_eq!(c.place, [2, 1, 0]);
        // The example is the first with a space of its own, and stays.
        assert_eq!(c.example.unwrap().0[15], 0xC07F_7F7F);
        assert_eq!(c.known.len(), 3);
        let line = c.line().unwrap();
        assert!(line.contains("IN A SPACE OF THEIR OWN 2 (scaled 1, moved 1, turned 1)"), "{line}");
        assert!(line.contains("a header: 81020006 00000028"), "{line}");
    }

    #[test]
    fn instances_are_found_by_their_last_eight_bytes() {
        let mut c = Counts::default();
        c.structure(A, &header(0x8102_0006, 40, 12, 0x8000_0000), Place::Arena);
        let unknown = 0x3_0000_0000u64;
        let mut block = vec![0u8; 40];
        block.extend(desc(0, 0xff, A));
        block.extend(desc(0x41, 0xda, A));
        block.extend(desc(0xC4, 0x58, unknown));
        block.extend(desc(0x80, 0x01, unknown + 0x1_0000));
        // An address with no room for a description before it, garbage
        // floats before another, and bytes that are no whole word.
        let mut early = vec![0u8; 16];
        early[8..16].copy_from_slice(&A.to_le_bytes());
        let mut bad = desc(0, 0xff, A);
        bad[0..4].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        let mut all = early;
        all.extend(&block);
        all.extend(bad);
        all.extend([0xffu8; 5]);
        c.list(&all, |a| if a == unknown { Place::Buffer } else { Place::Nowhere });
        assert_eq!((c.lists, c.descs, c.runs, c.odd, c.most), (1, 4, 3, 2, 4));
        assert_eq!(c.flags, [1, 0, 1, 0, 1, 1, 1, 0]);
        assert_eq!(c.theirs, (2, 2));
        assert_eq!(c.others, [0, 1, 1]);
        assert_eq!(c.bytes[1][&0xda], 1);
        let line = c.line().unwrap();
        assert!(line.contains("PULLED UP AS 4 NODES 1, AS 16 NODES 1, 0x80 alone 1"), "{line}");
        // The guard's source: a pulled-up instance was seen (0x41, 0xC4), and stays seen.
        assert!(pulled_up());
    }

    #[test]
    fn nothing_is_read_past_a_block_or_past_the_limit() {
        let mut c = Counts::default();
        // Every length from nothing to three descriptions, cut anywhere.
        let whole: Vec<u8> = [desc(1, 1, A), desc(2, 2, A), desc(4, 4, A)].concat();
        for len in 0..=whole.len() {
            c.list(&whole[..len], |_| Place::Nowhere);
        }
        // A description is counted from the length that holds all of it.
        assert_eq!(c.descs, (0..=whole.len()).map(|len| (len / DESC) as u64).sum::<u64>());
        // Words that are all addresses: none has a description's floats.
        let mut c = Counts::default();
        let addresses: Vec<u8> = (0..4096).flat_map(|_| A.to_le_bytes()).collect();
        c.list(&addresses, |_| Place::Arena);
        assert_eq!((c.descs, c.odd), (0, 4096));
        // Only the first megabyte of a block is looked at.
        let mut c = Counts::default();
        let long: Vec<u8> = (0..SCAN / DESC + 100).flat_map(|_| desc(0, 0xff, A)).collect();
        c.list(&long, |_| Place::Arena);
        assert_eq!(c.descs, (SCAN / DESC) as u64);
        // The tables do not grow without end.
        let mut c = Counts::default();
        for n in 0..(KNOWN as u64 + 10) {
            c.structure(A + n * 0x100, &header(0x8102_0006, 1, 1, 0), Place::Arena);
        }
        assert_eq!(c.known.len(), KNOWN);
        for n in 0..1000u64 {
            c.walked(B + n * 0x100, Place::Arena);
        }
        assert_eq!((c.walked.len(), c.walked_place[0]), (WALKED, WALKED as u64));
        let many: Vec<u8> = (0..=255u8).flat_map(|k| desc(k, k, A)).collect();
        c.list(&many, |_| Place::Arena);
        assert!(c.bytes[0].len() <= KINDS && c.bytes[1].len() <= KINDS);
    }

    #[test]
    fn one_list_in_sixteen_frames_and_nothing_to_say_before_anything_came() {
        let mut c = Counts::default();
        assert!(c.line().is_none());
        let taken: Vec<u64> = (100..140).filter(|&f| c.wants_list(f)).collect();
        assert_eq!(taken, [100, 116, 132]);
        // A frame counter that starts again is taken at once.
        assert!(c.wants_list(3));
        assert_eq!((0..64u64).filter(|&f| c.wants_walked(f)).count(), 4);
        c.walked(A, Place::Buffer);
        c.walked(A, Place::Buffer);
        assert!(c.line().unwrap().contains("structures given to walking dispatches 1: in an arena 0, in a title buffer 1, nowhere 0"));
    }

    /// What one look costs at the most: a whole megabyte of descriptions
    /// (16,384 instances; the title's lists are a few thousand), every one
    /// looked up. Printed with `--nocapture`; one look in 16 frames.
    #[test]
    fn a_look_at_a_full_megabyte_takes_under_two_milliseconds_in_release() {
        let mut c = Counts::default();
        for n in 0..2000u64 {
            c.structure(A + n * 0x1_0000, &header(0x8102_0006, 100, 50, 0), Place::Arena);
        }
        let block: Vec<u8> = (0..(SCAN / DESC) as u64).flat_map(|n| desc(0, 0xff, A + (n % 4000) * 0x1_0000)).collect();
        let arenas: Vec<u64> = (0..16).collect();
        let place = |a: u64| if arenas.iter().any(|&n| n == (a >> 26) % 64) { Place::Arena } else { Place::Nowhere };
        c.list(&block, place);
        let start = std::time::Instant::now();
        for _ in 0..20 {
            c.list(&block, place);
        }
        let micros = start.elapsed().as_micros() as f64 / 20.0;
        println!("raycounts: one look at {} instances: {micros:.0} microseconds = {:.4} ms a frame at one look in {EVERY} frames", SCAN / DESC, micros / 1000.0 / EVERY as f64);
        assert_eq!(c.descs, 21 * (SCAN / DESC) as u64);
        if !cfg!(debug_assertions) {
            assert!(micros < 2000.0, "{micros}");
        }
    }
}
