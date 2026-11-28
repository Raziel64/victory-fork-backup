//! The host's own ray-tracing structures, built from the title's
//! (`VICTORY_RAY_GUIDE`): everything that needs only a device and a command
//! list. No part of the layer is named here, so the same code runs in the
//! translator's software-adapter tests (`victory-shader`
//! `emit/rays_tests.rs` takes this file in by path) as in the game, where
//! `guide.rs` calls it.
//!
//! - [`Pool`]: the card's memory for host bottom levels, in blocks, under a
//!   cap in bytes; a block is made only while the caller's `room` says the
//!   card has it. A structure that finds no place is not built: its
//!   instances stay stand-in boxes, which the title walks itself as without
//!   the guide.
//! - the queue of the title's static bottom levels ([`Guide::came`],
//!   [`Guide::copy_back`], [`Guide::work`]): a structure the title's builder
//!   has unpacked is copied back to the PC, decoded
//!   ([`rays::bottom`]), checked, and built by the driver, so many triangles
//!   a frame at most; let go when the title drops its own ([`Guide::went`]).
//! - a host top level per title top level ([`Guide::top`],
//!   [`Guide::record_top`]): the pass [`rays::TOP_PASS`] writes its instance
//!   descriptions on the card and the driver builds from them, every time a
//!   walking program is about to be given that top level.
//! - the two tables: the guide table translated programs read
//!   (`victory_shader::emit::SLOT_GUIDE`, [`rays::guide_table`]) and the host
//!   table the pass reads ([`rays::hosts_table`]).
//!
//! # Order
//!
//! Lists are recorded on many threads and run in the order they are
//! submitted, which is not the order they are recorded in. Nothing here
//! leans on the order of two lists:
//!
//! - what shaders read of the tables is changed by copies recorded in a
//!   list, never by the CPU: a build and the table entries that name it are
//!   in one list, the build first;
//! - memory that was let go (a structure's, a list's, a buffer outgrown) is
//!   not given to anything else for [`Config::quarantine`] frames;
//! - every list leaves the buffers of this module in the state it found
//!   them in;
//! - the stand-in box is built in the first list that comes here, and no
//!   top level is made before those frames have passed.
//!
//! # What the layer adds to that (round 3)
//!
//! Frames alone do not say that a list has run. In the layer three more
//! things hold, none of which the software-adapter tests need:
//!
//! - [`Guide::clock`]: memory let go is used again, and a buffer dropped,
//!   only when the submissions that could name it have finished as well
//!   (the layer's own rule for everything it retires);
//! - [`Guide::take_sent`]: the caller sets the token when the list it
//!   recorded into is submitted. Nothing is made before the first list
//!   (stand-in box, tables) was submitted, and a structure is refitted only
//!   once a build of it was: a list that is recorded and never sent leaves
//!   nothing behind that a later list leans on;
//! - the tables start as zeros in the state shaders read them in, and the
//!   first list's words may be recorded again: the pass looks into no table
//!   of zeros, and a walking program finds no top level in one.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use super::rays;

type List = ID3D12GraphicsCommandList4;

/// A page of title memory.
pub const PAGE: u64 = 1 << 16;
/// The top levels guided at a time ([`rays::GUIDE_TOPS`]).
pub const TOPS: usize = rays::GUIDE_TOPS as usize;
/// The most instances a host top level is made for.
pub const MOST: u32 = 1 << 16;
/// A dispatch of the pass is given so many counts (dwords).
const COUNTS: u64 = 8;
/// Where [`rays::BOTTOM_PASS`]'s counts begin in the counts' buffer, and
/// the counts of structures nobody described ([`rays::census_job`]): the
/// byte of the first and how many at a time. The buffer holds
/// [`COUNTS_BYTES`] when the pass is used.
const MOVING_AT: u64 = 4 * COUNTS * TOPS as u64;
const CENSUS_AT: u64 = 8192;
const CENSUS_SLOTS: usize = 64;
pub const COUNTS_BYTES: u64 = CENSUS_AT + (4 * rays::CENSUS_DWORDS * CENSUS_SLOTS) as u64;

fn err(e: windows::core::Error) -> String {
    e.message().to_string()
}

/// How much the guide may take and do.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// The pool's cap, and a block of it, in bytes.
    pub cap: u64,
    pub block: u64,
    /// Triangles given to the driver's builds in a frame, at most (the
    /// structure that passes the mark is still built whole).
    pub triangles_a_frame: u32,
    /// Pages copied back in a frame, at most.
    pub pages_a_frame: u32,
    /// A structure is copied back when the title's builder has left it
    /// alone for so many frames.
    pub settle: u64,
    /// Memory let go is not used again for so many frames; nor is a top
    /// level made before so many frames after the first list.
    pub quarantine: u64,
    /// The guide table's dwords, and the host table's places.
    pub table: u32,
    pub places: u32,
    /// The instances a top level is first made for.
    pub most: u32,
    /// Bytes of the ring that carries vertices and table words to the card.
    pub upload: u64,
    /// Bottom levels the title builds at run time (step 4), in one list:
    /// the triangles refitted at most, and built anew at most (the first
    /// structure of a list is taken whatever its size; what passes the mark
    /// is a stand-in box in that list).
    pub moving_refit: u32,
    pub moving_build: u32,
    /// The pool's bytes they may hold in all.
    pub moving_cap: u64,
    /// One the title's builder wrote within so many frames is written anew
    /// by the pass in every list that walks.
    pub moving_fresh: u64,
    /// The host's structure is built anew after so many refits.
    pub moving_rebuild: u32,
}

impl Config {
    /// For a pool of `mb` MB.
    pub fn of(mb: u64) -> Self {
        let cap = mb << 20;
        Config {
            cap,
            block: (16 << 20).min(cap.max(1 << 16)),
            // (Measured on the software adapter's rig, a release build: the
            // recording thread's part of a build is about a quarter of a
            // microsecond a triangle, so half a millisecond in a frame that
            // builds; the card's own time for the build is not known.)
            triangles_a_frame: 2000,
            pages_a_frame: 32,
            settle: 12,
            quarantine: 16,
            // (A triangle takes four bytes of table and, of the pool, more
            // than 32: a full table before a full pool is not expected.)
            table: ((cap / 32) as u32).clamp(1 << 14, 1 << 23),
            places: 4096,
            most: 1024,
            upload: 4 << 20,
            // (Not measured on the card. The software adapter, a CPU, takes
            // 1.2 ms a thousand triangles to refit and 1.8 to build; a card
            // of this class is commonly quoted at 100 to 300 million
            // triangles a second refitted and a fifth of that built: these
            // marks would be 1 to 2 ms and 1 to 2.5 ms.)
            moving_refit: 200_000,
            moving_build: 50_000,
            moving_cap: cap / 4,
            moving_fresh: 4,
            moving_rebuild: 32,
        }
    }
}

/// Runs of a line that are free: for a block of the pool, the guide
/// table's lists, the pages copies come back to.
#[derive(Debug, Clone)]
pub struct Ranges {
    /// Start and length, in order, never touching.
    free: Vec<(u64, u64)>,
    total: u64,
}

impl Ranges {
    pub fn new(total: u64) -> Self {
        Ranges { free: vec![(0, total)], total }
    }

    /// The first free run of `len` whose start is a multiple of `align`.
    pub fn take(&mut self, len: u64, align: u64) -> Option<u64> {
        if len == 0 {
            return None;
        }
        for i in 0..self.free.len() {
            let (start, run) = self.free[i];
            let at = start.next_multiple_of(align);
            if at + len > start + run {
                continue;
            }
            self.free.remove(i);
            if at + len < start + run {
                self.free.insert(i, (at + len, start + run - at - len));
            }
            if at > start {
                self.free.insert(i, (start, at - start));
            }
            return Some(at);
        }
        None
    }

    /// `len` from `start` is free again.
    pub fn give(&mut self, start: u64, len: u64) {
        let i = self.free.partition_point(|r| r.0 < start);
        self.free.insert(i, (start, len));
        if i + 1 < self.free.len() && self.free[i].0 + self.free[i].1 == self.free[i + 1].0 {
            self.free[i].1 += self.free[i + 1].1;
            self.free.remove(i + 1);
        }
        if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == self.free[i].0 {
            self.free[i - 1].1 += self.free[i].1;
            self.free.remove(i);
        }
    }

    pub fn held(&self) -> u64 {
        self.total - self.free.iter().map(|r| r.1).sum::<u64>()
    }
}

/// A buffer on the card and its address there.
struct Buffer {
    res: ID3D12Resource,
    at: u64,
}

fn buffer(device: &ID3D12Device5, bytes: u64, heap: D3D12_HEAP_TYPE, state: D3D12_RESOURCE_STATES) -> Result<Buffer, String> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes.max(4).next_multiple_of(256),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if heap == D3D12_HEAP_TYPE_DEFAULT { D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS } else { D3D12_RESOURCE_FLAG_NONE },
        ..Default::default()
    };
    let mut made: Option<ID3D12Resource> = None;
    // SAFETY: the descriptions outlive the call.
    unsafe {
        device
            .CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() }, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut made)
            .map_err(|e| format!("a buffer of {bytes} bytes: {}", err(e)))?;
    }
    let res = made.ok_or("no buffer")?;
    // SAFETY: a live buffer.
    let at = unsafe { res.GetGPUVirtualAddress() };
    Ok(Buffer { res, at })
}

fn transition(r: &ID3D12Resource, before: D3D12_RESOURCE_STATES, after: D3D12_RESOURCE_STATES) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(r) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// What was written by shaders or builds before is seen by what comes
/// after.
fn settled() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER { pResource: std::mem::ManuallyDrop::new(None) }),
        },
    }
}

const STRUCTURE: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
const READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
const WRITTEN: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

/// A buffer the CPU reads and writes and the card copies to and shaders
/// write (the layer's shared buffers: memory of the PC): the caller makes
/// them.
pub struct Mapped {
    pub res: ID3D12Resource,
    pub data: *mut u8,
    pub len: u64,
}

/// The card's memory for host bottom levels.
pub struct Pool {
    cap: u64,
    block: u64,
    blocks: Vec<Option<(Buffer, Ranges)>>,
}

/// A structure's place in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Place {
    block: usize,
    offset: u64,
    len: u64,
    /// Its address on the card.
    pub at: u64,
}

/// Why a structure found no place.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoPlace {
    /// The pool is at its cap.
    Cap,
    /// The card has no room for another block.
    Card,
    /// Larger than a block.
    Large,
}

impl Pool {
    pub fn new(cap: u64, block: u64) -> Self {
        Pool { cap, block, blocks: Vec::new() }
    }

    /// The bytes of its blocks, and of the structures in them.
    pub fn bytes(&self) -> (u64, u64) {
        let blocks = self.blocks.iter().flatten().count() as u64 * self.block;
        (blocks, self.blocks.iter().flatten().map(|b| b.1.held()).sum())
    }

    /// A place of `len` bytes: in a block there is, else in a new one while
    /// the pool is under its cap and `room` has the card take the block.
    fn place(&mut self, device: &ID3D12Device5, len: u64, room: &mut dyn FnMut(u64) -> bool) -> Result<Place, NoPlace> {
        // (Structures lie on multiples of 256 bytes.)
        let len = len.next_multiple_of(256);
        if len > self.block {
            return Err(NoPlace::Large);
        }
        for (n, b) in self.blocks.iter_mut().enumerate() {
            if let Some((buffer, ranges)) = b
                && let Some(offset) = ranges.take(len, 256)
            {
                return Ok(Place { block: n, offset, len, at: buffer.at + offset });
            }
        }
        if self.bytes().0 + self.block > self.cap {
            return Err(NoPlace::Cap);
        }
        if !room(self.block) {
            return Err(NoPlace::Card);
        }
        let Ok(made) = buffer(device, self.block, D3D12_HEAP_TYPE_DEFAULT, STRUCTURE) else {
            return Err(NoPlace::Card);
        };
        let mut ranges = Ranges::new(self.block);
        let offset = ranges.take(len, 256).ok_or(NoPlace::Large)?;
        let at = made.at + offset;
        let n = match self.blocks.iter().position(Option::is_none) {
            Some(n) => n,
            None => {
                self.blocks.push(None);
                self.blocks.len() - 1
            }
        };
        self.blocks[n] = Some((made, ranges));
        Ok(Place { block: n, offset, len, at })
    }

    /// The place is free again; a block with nothing left in it goes back
    /// to the card (returned, for the caller to hold until no list names
    /// it).
    fn give(&mut self, place: Place) -> Option<ID3D12Resource> {
        let (_, ranges) = self.blocks[place.block].as_mut()?;
        ranges.give(place.offset, place.len);
        if ranges.held() != 0 {
            return None;
        }
        self.blocks[place.block].take().map(|b| b.0.res)
    }
}

/// The host table on the CPU: what the card's copy holds once the copies
/// recorded so far have run.
struct Hosts {
    words: Vec<u32>,
}

impl Hosts {
    fn new(places: u32) -> Self {
        let mut words = vec![0u32; (rays::HOSTS_HEADER + 4 * places) as usize];
        words[0] = places;
        Hosts { words }
    }

    fn at(&self, key: u32, probe: u32) -> usize {
        let places = self.words[0];
        (rays::HOSTS_HEADER + 4 * ((rays::hosts_place(key, places) + probe) & (places - 1))) as usize
    }

    /// Puts a bottom level in: the dword its place starts at.
    fn put(&mut self, bottom: u64, list: u32, structure: u64) -> Option<usize> {
        let key = rays::hosts_key(bottom);
        if key == 0 || key == rays::HOSTS_GONE {
            return None;
        }
        for probe in 0..rays::HOSTS_PROBES {
            let at = self.at(key, probe);
            if self.words[at] == 0 || self.words[at] == rays::HOSTS_GONE {
                self.words[at..at + 4].copy_from_slice(&[key, list, structure as u32, (structure >> 32) as u32]);
                return Some(at);
            }
        }
        None
    }

    /// Takes a bottom level out: the dword its place starts at.
    fn take(&mut self, bottom: u64) -> Option<usize> {
        let key = rays::hosts_key(bottom);
        for probe in 0..rays::HOSTS_PROBES {
            let at = self.at(key, probe);
            match self.words[at] {
                0 => return None,
                k if k == key => {
                    self.words[at..at + 4].copy_from_slice(&[rays::HOSTS_GONE, 0, 0, 0]);
                    return Some(at);
                }
                _ => {}
            }
        }
        None
    }
}

/// Bytes on their way to the card: a ring whose places are not used again
/// before the lists that read them have run.
struct Ring {
    buffer: Buffer,
    data: *mut u8,
    len: u64,
    at: u64,
    /// The frame, the submissions' stamp and the bytes of what was taken,
    /// oldest first.
    live: VecDeque<(u64, u64, u64, u64)>,
}

impl Ring {
    /// A place for `bytes`, filled with them: its offset.
    fn put(&mut self, bytes: &[u32], frame: u64, quarantine: u64, now: (u64, u64)) -> Option<u64> {
        while self.live.front().is_some_and(|l| l.0 + quarantine <= frame && l.1 <= now.1) {
            self.live.pop_front();
        }
        let len = (4 * bytes.len() as u64).next_multiple_of(16);
        if len > self.len {
            return None;
        }
        let start = if self.at + len > self.len { 0 } else { self.at };
        if self.live.iter().any(|&(_, _, from, to)| start < to && from < start + len) {
            return None;
        }
        // SAFETY: inside the mapping, which lives as long as the buffer.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.data.add(start as usize).cast::<u32>(), bytes.len()) };
        self.live.push_back((frame, now.0, start, start + len));
        self.at = start + len;
        Some(start)
    }

    /// Other words for a place [`Ring::put`] gave.
    fn write(&mut self, at: u64, words: &[u32]) {
        // SAFETY: a place of at least that many bytes inside the mapping.
        unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), self.data.add(at as usize).cast::<u32>(), words.len()) };
    }
}

/// Why a structure of the title's has no host structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// No place in the pool; it is asked for again when room comes.
    Room(NoPlace),
    /// The guide table or the host table is full.
    Tables,
    /// What was copied back is no bottom level of the title's format, or
    /// one whose triangles do not lie in its bounds: not built, for good.
    Unsound,
    /// Its pages were not on the card, or the copy never came.
    NotThere,
}

/// Where a structure of the title's is on its way.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Stage {
    /// The title's builder was given it; it waits to be left alone.
    Came,
    /// Its pages were copied back in this frame, to these pages of the
    /// buffer copies come back to.
    Copied { frame: u64, first: u64, pages: u64 },
    Built { place: Place, list: (u64, u64), triangles: u32 },
    Refused(Refused),
}

struct Job {
    len: u64,
    header: u64,
    /// The frame the builder last named it in.
    last: u64,
    tries: u32,
    stage: Stage,
}

/// What was let go, until it may be used again.
enum Freed {
    Place(Place),
    List(u64, u64),
    Buffer(#[allow(dead_code)] ID3D12Resource),
}

/// A host top level.
struct Top {
    address: u64,
    used: u64,
    most: u32,
    generation: u32,
    descs: Buffer,
    tlas: Buffer,
}

/// A host top level for a walk: which of the table's entries, which making
/// of it (another one: another view), and its address on the card.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TopUse {
    pub index: usize,
    pub generation: u32,
    pub location: u64,
}

/// What a dispatch of the pass is given beside the top level.
#[derive(Clone, Copy)]
pub struct TopIo {
    /// The directory of title memory by address, on the card, and the
    /// notes of pages found without a copy (the walking programs' half).
    pub directory: u64,
    pub faults: u64,
    /// The table of the copies' views, from the null view on.
    pub copies: D3D12_GPU_DESCRIPTOR_HANDLE,
    /// The view-heap slot the host top level's view is in.
    pub slot: u32,
}

/// A bottom level the title builds itself at run time, as the title
/// described it to its builder: where it will lie, and the triangles of
/// each of its geometries (up to [`rays::MOVING_GEOMETRIES`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Moving {
    pub address: u64,
    pub triangles: [u32; rays::MOVING_GEOMETRIES],
}

/// The host's structure of one.
struct Moved {
    triangles: u32,
    bases: [u32; rays::MOVING_GEOMETRIES],
    place: Place,
    list: (u64, u64),
    hosts_at: usize,
    /// The frame its structure was first built in (none: not yet), and the
    /// frame the title last built or refitted its own.
    built: Option<u64>,
    last: u64,
    /// The token of the list its last build without a proven one before it
    /// was recorded in; and whether a build of it is known to have been
    /// submitted: from then on a refit finds a structure there whatever
    /// order the lists since ran in.
    sent: Option<Arc<AtomicBool>>,
    proven: bool,
    /// Refits since the last build; the frame the pass last wrote it in;
    /// whether that pass offered it (else: over the list's marks, a box).
    refits: u32,
    hosted: u64,
    on: bool,
    /// Whether the title's builder call described it ([`Guide::moving`]),
    /// else the pass counted it ([`Guide::seen`]).
    described: bool,
}

/// A structure the title's builder wrote that nobody described: counted on
/// the card before the host builds its own of it.
struct Census {
    /// The frame the builder last wrote it in.
    last: u64,
    /// The triangle nodes the next count looks at.
    cap: u32,
    /// A count on its way: its slot, the pass's stamp, the frame.
    flight: Option<(usize, u32, u64)>,
    /// Not counted again before this frame.
    wait: u64,
}

/// Counted since the start.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    /// Structures of the title's that came, went, were built, and were
    /// refused for good as not sound or not there.
    pub came: u64,
    pub went: u64,
    pub built: u64,
    pub unsound: u64,
    pub not_there: u64,
    /// Builds turned down for room (a structure may be turned down again).
    pub no_room: u64,
    pub triangles: u64,
    /// Of the static bottom levels decoded on the PC side (one decoded
    /// again counts again): triangles marked opaque and not, box nodes of
    /// 16 and of 32 bits, nodes of other kinds, and structures stored in a
    /// space of their own.
    pub opaque: u64,
    pub not_opaque: u64,
    pub boxes: [u64; 2],
    pub other_nodes: u64,
    pub own_space: u64,
    /// Pages copied back; frames with a build; top levels made for walks.
    pub pages: u64,
    pub build_frames: u64,
    pub tops: u64,
    /// Walks of a top level the table had no entry for.
    pub tops_over: u64,
    /// Bottom levels the title builds at run time: host builds, host
    /// refits, their triangles, those turned down (no room, or more
    /// geometries than the pass takes), those let go after frames unused.
    pub moving_builds: u64,
    pub moving_refits: u64,
    pub moving_triangles: u64,
    pub moving_refused: u64,
    pub moving_gone: u64,
    /// Of them: left a stand-in box in a list for the list's marks; found
    /// by the pass not to fit their count (counted again); counts asked of
    /// the card, and those that found no bottom level at the address; rows
    /// of the pass; lists it was recorded in.
    pub moving_over: u64,
    pub moving_misfit: u64,
    pub census_asked: u64,
    pub census_refused: u64,
    pub moving_rows: u64,
    pub moving_lists: u64,
    /// The first list's words recorded again (its list was never sent).
    pub begun_again: u64,
    /// Places of the ring that were not to be had (a step waited a frame).
    pub ring_full: u64,
}

/// The host's side of the guide.
pub struct Guide {
    device: ID3D12Device5,
    cfg: Config,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    pool: Pool,
    stand_in: Buffer,
    scratch: Buffer,
    scratch_len: u64,
    table: Buffer,
    lists: Ranges,
    hosts: Buffer,
    hosts_cpu: Hosts,
    ring: Ring,
    back: Mapped,
    back_pages: Ranges,
    counts: Mapped,
    jobs: HashMap<u64, Job>,
    /// Their addresses in the order they came.
    order: Vec<u64>,
    /// Structures let go whose table entries are still to be taken out.
    leaving: Vec<(u64, Place, (u64, u64))>,
    /// With the frame and the submissions' stamp it was let go at.
    freed: VecDeque<(u64, u64, Freed)>,
    tops: [Option<Top>; TOPS],
    /// Bottom levels the title builds at run time, by address; the pass
    /// for them; and their vertices on the card (none: not made yet).
    moved: HashMap<u64, Moved>,
    /// Places of the host table to mark gone with the next pass.
    moving_gone: Vec<(u64, usize)>,
    bottom_pass: Option<(ID3D12RootSignature, ID3D12PipelineState)>,
    moving_vertices: Option<(Buffer, u64)>,
    /// The pool's bytes they hold.
    moving_held: u64,
    /// Those nobody described, by address; the free slots of their counts
    /// (a bit each); the frame counts were last asked in; the passes' stamp.
    census: HashMap<u64, Census>,
    census_free: u64,
    census_frame: u64,
    stamp: u32,
    /// The submissions' clock ([`Guide::clock`]): the stamp of what is let
    /// go now, and the newest stamp whose submissions have all finished.
    now: (u64, u64),
    /// Whether tokens are waited for ([`Guide::take_sent`]); the token of
    /// what was recorded since it was last taken; the first list's.
    wait_sent: bool,
    sent: Option<Arc<AtomicBool>>,
    begun: Option<Arc<AtomicBool>>,
    came_free: bool,
    /// The frame of the first list, and of the last copies and builds.
    born: Option<u64>,
    copied: u64,
    worked: u64,
    /// Whether a build was turned down for room since room last came.
    starved: bool,
    pub stats: Stats,
}

// SAFETY: the mappings are of buffers the struct holds; the caller keeps it
// behind a lock.
unsafe impl Send for Guide {}

impl Guide {
    /// The guide for `device`: `pass` is [`rays::TOP_PASS`] compiled
    /// (`cs_5_1`); `back` (at least a few MB) is where copies come back to
    /// and `counts` (at least 256 bytes) where the pass counts.
    pub fn new(device: &ID3D12Device5, cfg: Config, pass: &[u8], back: Mapped, counts: Mapped) -> Result<Self, String> {
        assert!(cfg.places.is_power_of_two(), "{} places", cfg.places);
        let root = pass_root(device)?;
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: pass.as_ptr().cast(), BytecodeLength: pass.len() },
            ..Default::default()
        };
        // SAFETY: the description points at live data.
        let pso: ID3D12PipelineState = unsafe { device.CreateComputePipelineState(&desc) }.map_err(|e| format!("the pass: {}", err(e)))?;
        let ring = {
            let b = buffer(device, cfg.upload, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ)?;
            let mut data = std::ptr::null_mut();
            // SAFETY: an upload buffer, mapped for its life.
            unsafe { b.res.Map(0, None, Some(&mut data)) }.map_err(err)?;
            Ring { buffer: b, data: data.cast(), len: cfg.upload, at: 0, live: VecDeque::new() }
        };
        let scratch_len = 1 << 20;
        // SAFETY: the caller's mapping of at least that many bytes.
        unsafe { std::ptr::write_bytes(counts.data, 0, (4 * COUNTS as usize * TOPS).min(counts.len as usize)) };
        Ok(Guide {
            device: device.clone(),
            cfg,
            root,
            pso,
            pool: Pool::new(cfg.cap, cfg.block),
            stand_in: buffer(device, 1 << 16, D3D12_HEAP_TYPE_DEFAULT, STRUCTURE)?,
            scratch: buffer(device, scratch_len, D3D12_HEAP_TYPE_DEFAULT, WRITTEN)?,
            scratch_len,
            table: buffer(device, 4 * u64::from(cfg.table), D3D12_HEAP_TYPE_DEFAULT, READ)?,
            lists: {
                let mut lists = Ranges::new(u64::from(cfg.table));
                // (The header: the top levels' entries.)
                lists.take(u64::from(rays::GUIDE_HEADER), 1);
                lists
            },
            hosts: buffer(device, 4 * u64::from(rays::HOSTS_HEADER + 4 * cfg.places), D3D12_HEAP_TYPE_DEFAULT, READ)?,
            hosts_cpu: Hosts::new(cfg.places),
            ring,
            back_pages: Ranges::new(back.len / PAGE),
            back,
            counts,
            jobs: HashMap::new(),
            order: Vec::new(),
            leaving: Vec::new(),
            freed: VecDeque::new(),
            tops: std::array::from_fn(|_| None),
            moved: HashMap::new(),
            moving_gone: Vec::new(),
            bottom_pass: None,
            moving_vertices: None,
            moving_held: 0,
            census: HashMap::new(),
            census_free: !0,
            census_frame: 0,
            stamp: 0,
            now: (0, 0),
            wait_sent: false,
            sent: None,
            begun: None,
            came_free: false,
            born: None,
            copied: 0,
            worked: 0,
            starved: false,
            stats: Stats::default(),
        })
    }

    /// The buffers made so far, for the caller's books.
    pub fn buffers(&self) -> Vec<(&ID3D12Resource, &'static str)> {
        let mut all = vec![
            (&self.stand_in.res, "the stand-in box"),
            (&self.scratch.res, "the builds' scratch memory"),
            (&self.table.res, "the guide table"),
            (&self.hosts.res, "the host table"),
            (&self.ring.buffer.res, "vertices and table words on their way"),
        ];
        all.extend(self.pool.blocks.iter().flatten().map(|b| (&b.0.res, "a block of host bottom levels")));
        for t in self.tops.iter().flatten() {
            all.push((&t.descs.res, "a host top level's instances"));
            all.push((&t.tlas.res, "a host top level"));
        }
        all
    }

    /// The guide table's address on the card, what a walking program's
    /// `SLOT_GUIDE` is bound to: once the first list has had its frames.
    pub fn table(&self, frame: u64) -> Option<u64> {
        self.live(frame).then_some(self.table.at)
    }

    /// The bytes of the pool's blocks and of the structures in them; and
    /// the bytes of everything else here that lies on the card.
    pub fn bytes(&self) -> (u64, u64, u64) {
        let (blocks, held) = self.pool.bytes();
        let size = |b: &Buffer| {
            // SAFETY: a live buffer.
            unsafe { b.res.GetDesc().Width }
        };
        let tops: u64 = self.tops.iter().flatten().map(|t| size(&t.descs) + size(&t.tlas)).sum();
        (blocks, held, tops + size(&self.stand_in) + size(&self.scratch) + size(&self.table) + size(&self.hosts))
    }

    /// Structures built and held now; waiting (to be left alone, to be
    /// copied back, to be decoded); turned down for room; and the triangles
    /// of those held.
    pub fn standing(&self) -> (usize, usize, usize, u64) {
        let mut n = (0, 0, 0, 0);
        for j in self.jobs.values() {
            match j.stage {
                Stage::Built { triangles, .. } => {
                    n.0 += 1;
                    n.3 += u64::from(triangles);
                }
                Stage::Came | Stage::Copied { .. } => n.1 += 1,
                Stage::Refused(Refused::Room(_) | Refused::Tables) => n.2 += 1,
                Stage::Refused(_) => {}
            }
        }
        n
    }

    /// What the pass counted for entry `index` of the table: the title's
    /// instances at its last dispatch (all ones: no header), and since the
    /// start those given a host structure, a stand-in box, left out.
    pub fn counted(&self, index: usize) -> [u32; 4] {
        // SAFETY: inside the caller's mapping; shaders write it, so the
        // dwords are only numbers to say.
        std::array::from_fn(|k| unsafe { self.counts.data.cast::<u32>().add(COUNTS as usize * index + k).read_volatile() })
    }

    fn live(&self, frame: u64) -> bool {
        self.born.is_some_and(|born| frame >= born + self.cfg.quarantine)
            && (!self.wait_sent || self.begun.as_ref().is_some_and(|b| b.load(Ordering::Relaxed)))
    }

    /// Whether the first list has been recorded, and whether everything is
    /// made from now on ([`Guide::begin`]'s frames have passed and, where
    /// tokens are waited for, its list was submitted).
    pub fn state(&self, frame: u64) -> (bool, bool) {
        (self.born.is_some(), self.live(frame))
    }

    fn put(&mut self, words: &[u32], frame: u64) -> Option<u64> {
        let at = self.ring.put(words, frame, self.cfg.quarantine, self.now);
        self.stats.ring_full += u64::from(at.is_none());
        at
    }

    /// The submissions' clock, before a call that may let memory go or use
    /// it again: `stamp` is what the layer would wait for to drop an object
    /// let go now (every submission begun or queued), `done` the newest
    /// stamp whose wait is over. From the first call on, the frames'
    /// quarantine alone frees nothing, and tokens are waited for
    /// ([`Guide::take_sent`]). Never called by the software-adapter tests
    /// that go by frames.
    pub fn clock(&mut self, stamp: u64, done: u64) {
        (self.now, self.wait_sent) = ((stamp, done), true);
    }

    /// The token of what was recorded since this was last called (none:
    /// nothing that a later list leans on): the caller sets it when the
    /// list it recorded into is submitted.
    pub fn take_sent(&mut self) -> Option<Arc<AtomicBool>> {
        self.sent.take()
    }

    fn token(&mut self) -> Arc<AtomicBool> {
        self.sent.get_or_insert_with(|| Arc::new(AtomicBool::new(false))).clone()
    }

    fn let_go(&mut self, frame: u64, what: Freed) {
        self.freed.push_back((frame, self.now.0, what));
    }

    /// What was let go long enough ago is free again: by frames and, where
    /// the caller keeps the clock, by submissions.
    fn reclaim(&mut self, frame: u64) {
        while self.freed.front().is_some_and(|f| f.0 + self.cfg.quarantine <= frame && f.1 <= self.now.1) {
            match self.freed.pop_front().map(|f| f.2) {
                Some(Freed::Place(place)) => {
                    self.came_free = true;
                    if let Some(block) = self.pool.give(place) {
                        // (No structure of it is named any more: the block
                        // itself goes after the same wait.)
                        self.let_go(frame, Freed::Buffer(block));
                    }
                }
                Some(Freed::List(start, len)) => self.lists.give(start, len),
                Some(Freed::Buffer(_)) | None => {}
            }
        }
    }

    /// Memory let go and not yet free again: entries, and of them buffers.
    pub fn waiting_to_go(&self) -> (usize, usize) {
        (self.freed.len(), self.freed.iter().filter(|f| matches!(f.2, Freed::Buffer(_))).count())
    }

    /// The first list that comes here: the stand-in box is built in it and
    /// the two tables are given their first contents. True when that was
    /// recorded now.
    pub fn begin(&mut self, list: &List, frame: u64) -> Result<bool, String> {
        self.reclaim(frame);
        if let Some(born) = self.born {
            // Its list was never submitted (the title reset it): the same
            // words again, in this list. Nothing was made meanwhile.
            let lost = self.wait_sent && frame >= born + 4 * self.cfg.quarantine && self.begun.as_ref().is_none_or(|b| !b.load(Ordering::Relaxed));
            if !lost {
                return Ok(false);
            }
            self.stats.begun_again += 1;
        }
        let cube = self.put(&rays::STAND_IN.map(f32::to_bits), frame).ok_or("no room for the first words")?;
        let (header, _) = rays::guide_table(&[], &[]);
        let header_at = self.put(&header, frame).ok_or("no room for the first words")?;
        let hosts = self.hosts_cpu.words.clone();
        let hosts_at = self.put(&hosts, frame).ok_or("no room for the first words")?;
        let geometry = D3D12_RAYTRACING_GEOMETRY_DESC {
            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS,
            Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_NONE,
            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                AABBs: D3D12_RAYTRACING_GEOMETRY_AABBS_DESC {
                    AABBCount: 1,
                    AABBs: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE { StartAddress: self.ring.buffer.at + cube, StrideInBytes: 24 },
                },
            },
        };
        let inputs = D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: 1,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { pGeometryDescs: &geometry },
        };
        let (result, scratch) = self.sizes(&inputs);
        if result > 1 << 16 || result == 0 {
            return Err(format!("the stand-in box takes {result} bytes"));
        }
        self.grow_scratch(scratch, frame)?;
        // SAFETY: an open list; live buffers of the sizes the driver asked
        // for, each in the state it was made in.
        unsafe {
            list.BuildRaytracingAccelerationStructure(
                &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                    DestAccelerationStructureData: self.stand_in.at,
                    Inputs: inputs,
                    SourceAccelerationStructureData: 0,
                    ScratchAccelerationStructureData: self.scratch.at,
                },
                None,
            );
            // (The tables rest in the state shaders read them in, as every
            // list finds and leaves them: these words may be recorded twice.)
            list.ResourceBarrier(&[
                settled(),
                transition(&self.table.res, READ, D3D12_RESOURCE_STATE_COPY_DEST),
                transition(&self.hosts.res, READ, D3D12_RESOURCE_STATE_COPY_DEST),
            ]);
            list.CopyBufferRegion(&self.table.res, 0, &self.ring.buffer.res, header_at, 4 * header.len() as u64);
            list.CopyBufferRegion(&self.hosts.res, 0, &self.ring.buffer.res, hosts_at, 4 * hosts.len() as u64);
            list.ResourceBarrier(&[
                transition(&self.table.res, D3D12_RESOURCE_STATE_COPY_DEST, READ),
                transition(&self.hosts.res, D3D12_RESOURCE_STATE_COPY_DEST, READ),
            ]);
        }
        self.born = Some(frame);
        self.begun = Some(self.token());
        Ok(true)
    }

    fn sizes(&self, inputs: &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS) -> (u64, u64) {
        let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        // SAFETY: `inputs` points at descriptions the caller keeps.
        unsafe { self.device.GetRaytracingAccelerationStructurePrebuildInfo(inputs, &mut info) };
        (info.ResultDataMaxSizeInBytes, info.ScratchDataSizeInBytes.max(info.UpdateScratchDataSizeInBytes))
    }

    /// The builds' scratch memory holds `bytes`: a larger one is made, and
    /// the one outgrown held until no list names it.
    fn grow_scratch(&mut self, bytes: u64, frame: u64) -> Result<(), String> {
        if bytes <= self.scratch_len {
            return Ok(());
        }
        let len = bytes.next_power_of_two();
        let made = buffer(&self.device, len, D3D12_HEAP_TYPE_DEFAULT, WRITTEN)?;
        let old = std::mem::replace(&mut self.scratch, made);
        self.scratch_len = len;
        self.let_go(frame, Freed::Buffer(old.res));
        Ok(())
    }

    /// The title's builder was given the packed structure at `address`
    /// (`len` bytes unpacked, `header` its first two dwords) in `frame`.
    pub fn came(&mut self, address: u64, len: u64, header: u64, frame: u64) {
        if let Some(job) = self.jobs.get_mut(&address) {
            if job.header == header {
                // Still at it: what is copied back must wait. (One built
                // already is unpacked to the same bytes again.)
                job.last = frame;
                match job.stage {
                    Stage::Copied { first, pages, .. } => {
                        self.back_pages.give(first, pages);
                        job.stage = Stage::Came;
                    }
                    // (It may be there now.)
                    Stage::Refused(Refused::NotThere) => (job.stage, job.tries) = (Stage::Came, 0),
                    _ => {}
                }
                return;
            }
            self.went(address);
        }
        // (One the title built at run time at this address is no more, nor
        // a count of one on its way.)
        self.forget(address, frame);
        if let Some((slot, ..)) = self.census.remove(&address).and_then(|c| c.flight) {
            self.census_free |= 1 << slot;
        }
        self.jobs.insert(address, Job { len, header, last: frame, tries: 0, stage: Stage::Came });
        self.order.push(address);
        self.stats.came += 1;
    }

    /// The title dropped its structure at `address`: so does the host. Its
    /// table entries go with the next builds' list; its memory is free a
    /// while after.
    pub fn went(&mut self, address: u64) {
        let Some(job) = self.jobs.remove(&address) else { return };
        self.order.retain(|&a| a != address);
        self.stats.went += 1;
        match job.stage {
            Stage::Built { place, list, .. } => self.leaving.push((address, place, list)),
            Stage::Copied { first, pages, .. } => self.back_pages.give(first, pages),
            Stage::Came | Stage::Refused(_) => {}
        }
    }

    /// Records the copies back of structures the builder has left alone:
    /// `page` gives a page of title memory's place on the card (a buffer in
    /// the common state that this list has not used, and the byte in it).
    /// Once a frame; at most [`Config::pages_a_frame`] pages.
    pub fn copy_back(&mut self, list: &List, frame: u64, page: &dyn Fn(u64) -> Option<(ID3D12Resource, u64)>) {
        if !self.live(frame) || self.copied == frame {
            return;
        }
        self.copied = frame;
        let mut left = u64::from(self.cfg.pages_a_frame);
        // To, from, the byte there.
        let mut copies: Vec<(u64, ID3D12Resource, u64)> = Vec::new();
        // The newest first: what came last is where the player is going.
        for n in (0..self.order.len()).rev() {
            let address = self.order[n];
            let Some(job) = self.jobs.get_mut(&address) else { continue };
            if job.stage != Stage::Came || frame < job.last + self.cfg.settle {
                continue;
            }
            let (first, end) = (address / PAGE, (address + job.len).div_ceil(PAGE));
            let pages = end - first;
            if pages > self.back_pages.total {
                job.stage = Stage::Refused(Refused::NotThere);
                self.stats.not_there += 1;
                continue;
            }
            if pages > left {
                continue;
            }
            let from: Vec<(ID3D12Resource, u64)> = (first..end).map_while(page).collect();
            if from.len() as u64 != pages {
                // A page of it has no copy on the card (yet).
                job.tries += 1;
                if job.tries >= 6 {
                    job.stage = Stage::Refused(Refused::NotThere);
                    self.stats.not_there += 1;
                }
                job.last = frame;
                continue;
            }
            let Some(to) = self.back_pages.take(pages, 1) else { break };
            // What the copy must write over before it is taken for come.
            // SAFETY: inside the caller's mapping.
            unsafe { std::ptr::write_bytes(self.back.data.add((to * PAGE + address % PAGE) as usize), 0, 8) };
            for (k, (res, at)) in from.into_iter().enumerate() {
                copies.push(((to + k as u64) * PAGE, res, at));
            }
            job.stage = Stage::Copied { frame, first: to, pages };
            left -= pages;
            self.stats.pages += pages;
        }
        if copies.is_empty() {
            return;
        }
        let mut sources: Vec<&ID3D12Resource> = Vec::new();
        for (_, res, _) in &copies {
            if !sources.iter().any(|s| *s == res) {
                sources.push(res);
            }
        }
        let turn = |before, after| -> Vec<D3D12_RESOURCE_BARRIER> { sources.iter().map(|s| transition(s, before, after)).collect() };
        // SAFETY: an open list; live buffers; the pages lie inside them.
        unsafe {
            list.ResourceBarrier(&turn(D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE));
            for (to, res, at) in &copies {
                list.CopyBufferRegion(&self.back.res, *to, res, *at, PAGE);
            }
            list.ResourceBarrier(&turn(D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON));
        }
    }

    /// The structure of a job whose copy has come, decoded and checked.
    fn decode(&self, address: u64, job: &Job, first: u64, pages: u64) -> Result<rays::Bottom, Option<Refused>> {
        let at = (first * PAGE) as usize;
        // SAFETY: pages of the caller's mapping, on a multiple of 64 KiB,
        // that nothing writes while the lock is held but a copy long done.
        let words = unsafe { std::slice::from_raw_parts(self.back.data.add(at).cast::<u32>(), (pages * PAGE / 4) as usize) };
        let base = address / PAGE * PAGE;
        let head = ((address - base) / 4) as usize;
        if words[head] == 0 && words[head + 1] == 0 {
            // The copy has not run yet.
            return Err(None);
        }
        let memory = rays::Span { base, words };
        let bottom = match rays::bottom(&memory, address) {
            Ok(b) => b,
            // Not unpacked yet: looked at again.
            Err(rays::Error::Packed(_)) => return Err(None),
            Err(_) => return Err(Some(Refused::Unsound)),
        };
        // The structure the builder was given: its size, and triangles that
        // lie where it says it is.
        let sound = (u64::from(bottom.header.nodes) + 1) * 64 <= job.len.max(64)
            && !bottom.triangles.is_empty()
            && bottom.triangles.len() < 1 << 22
            && rays::stand_in_bounds(&bottom.header.bounds).is_some_and(|b| {
                // (Vertices brought back from a space of the structure's own
                // are a rounding off the bounds the builder took of the
                // given ones: 0.004 at 200,000 in builds looked at.)
                let slack = |x: f32| if bottom.header.own_space() { 1e-6 * x.abs() + 1e-4 } else { 0.0 };
                bottom.triangles.iter().flat_map(|t| t.vertices).all(|v| (0..3).all(|a| v[a] >= b[a] - slack(b[a]) && v[a] <= b[3 + a] + slack(b[3 + a])))
            });
        if sound { Ok(bottom) } else { Err(Some(Refused::Unsound)) }
    }

    /// Records the builds of structures whose copies have come, and what
    /// the tables gain and lose: once a frame, at most
    /// [`Config::triangles_a_frame`] triangles. `room`: whether the card
    /// has so many bytes for another block of the pool.
    pub fn work(&mut self, list: &List, frame: u64, room: &mut dyn FnMut(u64) -> bool) -> Result<(), String> {
        if !self.live(frame) || self.worked == frame {
            return Ok(());
        }
        self.worked = frame;
        self.reclaim(frame);
        let came_free = std::mem::take(&mut self.came_free);
        // Room came: those turned down for it are asked for again.
        if came_free && self.starved {
            self.starved = false;
            for job in self.jobs.values_mut() {
                if matches!(job.stage, Stage::Refused(Refused::Room(_) | Refused::Tables)) {
                    (job.stage, job.last) = (Stage::Came, frame);
                }
            }
        }
        // What the tables are to hold: (the table's dword, the ring's byte,
        // dwords) for the guide table and for the host table.
        let mut table_words: Vec<(u64, u64, u64)> = Vec::new();
        let mut hosts_words: Vec<(u64, u64, u64)> = Vec::new();
        for (address, place, list_at) in std::mem::take(&mut self.leaving) {
            let Some(from) = self.put(&[rays::HOSTS_GONE, 0, 0, 0], frame) else {
                // (Said again with the next list.)
                self.leaving.push((address, place, list_at));
                continue;
            };
            if let Some(at) = self.hosts_cpu.take(address) {
                hosts_words.push((at as u64, from, 4));
            }
            self.let_go(frame, Freed::Place(place));
            self.let_go(frame, Freed::List(list_at.0, list_at.1));
        }
        // The builds: destination, vertices, triangles.
        let mut builds: Vec<(u64, u64, u32)> = Vec::new();
        let mut triangles = 0u32;
        for n in (0..self.order.len()).rev() {
            if triangles >= self.cfg.triangles_a_frame {
                break;
            }
            let address = self.order[n];
            let Some(job) = self.jobs.get(&address) else { continue };
            let Stage::Copied { frame: copied, first, pages } = job.stage else { continue };
            if frame < copied + 2 {
                continue;
            }
            let bottom = match self.decode(address, job, first, pages) {
                Ok(b) => b,
                Err(why) => {
                    // Not come or not unpacked: waited for, then asked for
                    // again, a few times.
                    let tries = job.tries;
                    let stage = match why {
                        Some(why) => Some(Stage::Refused(why)),
                        None if frame < copied + 2 * self.cfg.settle => None,
                        None if tries >= 3 => Some(Stage::Refused(Refused::NotThere)),
                        None => Some(Stage::Came),
                    };
                    if let Some(stage) = stage {
                        self.back_pages.give(first, pages);
                        match stage {
                            Stage::Refused(Refused::Unsound) => self.stats.unsound += 1,
                            Stage::Refused(_) => self.stats.not_there += 1,
                            _ => {}
                        }
                        if let Some(job) = self.jobs.get_mut(&address) {
                            (job.stage, job.last, job.tries) = (stage, frame, tries + 1);
                        }
                    }
                    continue;
                }
            };
            self.back_pages.give(first, pages);
            let count = bottom.triangles.len() as u32;
            let opaque = bottom.triangles.iter().filter(|t| t.opaque).count() as u64;
            self.stats.opaque += opaque;
            self.stats.not_opaque += u64::from(count) - opaque;
            self.stats.boxes[0] += u64::from(bottom.boxes[0]);
            self.stats.boxes[1] += u64::from(bottom.boxes[1]);
            self.stats.other_nodes += u64::from(bottom.others);
            self.stats.own_space += u64::from(bottom.header.own_space());
            let made = self.build(&bottom, frame, room);
            let stage = match made {
                Ok((place, list_at, vertices, table, hosts)) => {
                    builds.push((place.at, vertices, count));
                    table_words.push(table);
                    hosts_words.push(hosts);
                    triangles += count;
                    self.stats.built += 1;
                    self.stats.triangles += u64::from(count);
                    Stage::Built { place, list: list_at, triangles: count }
                }
                // (The ring is full for this frame: looked at again.)
                Err(None) => Stage::Came,
                Err(Some(why)) => {
                    self.stats.no_room += 1;
                    self.starved = true;
                    Stage::Refused(why)
                }
            };
            if let Some(job) = self.jobs.get_mut(&address) {
                (job.stage, job.last) = (stage, frame);
            }
        }
        if builds.is_empty() && table_words.is_empty() && hosts_words.is_empty() {
            return Ok(());
        }
        self.stats.build_frames += u64::from(!builds.is_empty());
        // SAFETY: an open list; live buffers in the states this module
        // keeps them in; every range lies inside its buffer.
        unsafe {
            for &(to, vertices, count) in &builds {
                let geometry = triangles_of(self.ring.buffer.at + vertices, count);
                list.BuildRaytracingAccelerationStructure(
                    &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                        DestAccelerationStructureData: to,
                        Inputs: bottom_inputs(&geometry),
                        SourceAccelerationStructureData: 0,
                        ScratchAccelerationStructureData: self.scratch.at,
                    },
                    None,
                );
                // (The next build takes the same scratch memory.)
                list.ResourceBarrier(&[settled()]);
            }
            // The entries after the builds they name.
            list.ResourceBarrier(&[
                transition(&self.table.res, READ, D3D12_RESOURCE_STATE_COPY_DEST),
                transition(&self.hosts.res, READ, D3D12_RESOURCE_STATE_COPY_DEST),
            ]);
            for &(to, from, dwords) in &table_words {
                list.CopyBufferRegion(&self.table.res, 4 * to, &self.ring.buffer.res, from, 4 * dwords);
            }
            for &(to, from, dwords) in &hosts_words {
                list.CopyBufferRegion(&self.hosts.res, 4 * to, &self.ring.buffer.res, from, 4 * dwords);
            }
            list.ResourceBarrier(&[
                transition(&self.table.res, D3D12_RESOURCE_STATE_COPY_DEST, READ),
                transition(&self.hosts.res, D3D12_RESOURCE_STATE_COPY_DEST, READ),
            ]);
        }
        Ok(())
    }

    /// A place, a list, the vertices and the table words of `bottom`: its
    /// place in the pool, its list (start, dwords), the ring's byte of its
    /// vertices, and the two tables' words to copy. None: no room in the
    /// ring now.
    #[allow(clippy::type_complexity)]
    fn build(
        &mut self,
        bottom: &rays::Bottom,
        frame: u64,
        room: &mut dyn FnMut(u64) -> bool,
    ) -> Result<(Place, (u64, u64), u64, (u64, u64, u64), (u64, u64, u64)), Option<Refused>> {
        let count = bottom.triangles.len() as u32;
        let geometry = triangles_of(self.ring.buffer.at, count);
        let (result, scratch) = self.sizes(&bottom_inputs(&geometry));
        if result == 0 || self.grow_scratch(scratch, frame).is_err() {
            return Err(Some(Refused::Room(NoPlace::Card)));
        }
        // Its list in the guide table: the root's key, then a triangle's
        // pointer per host primitive.
        let dwords = 1 + u64::from(count);
        let Some(list_at) = self.lists.take(dwords, 1).filter(|at| at + dwords < 1 << 24) else {
            return Err(Some(Refused::Tables));
        };
        let place = match self.pool.place(&self.device, result, room) {
            Ok(place) => place,
            Err(why) => {
                self.lists.give(list_at, dwords);
                return Err(Some(Refused::Room(why)));
            }
        };
        let back_out = |g: &mut Self, why| {
            g.lists.give(list_at, dwords);
            if let Some(block) = g.pool.give(place) {
                g.let_go(frame, Freed::Buffer(block));
            }
            why
        };
        let address = bottom.header.address;
        let vertices: Vec<u32> = bottom.triangles.iter().flat_map(|t| t.vertices).flatten().map(f32::to_bits).collect();
        let mut list_words = Vec::with_capacity(dwords as usize);
        list_words.push(rays::guide_key(address + u64::from(bottom.header.root >> 3) * 64));
        list_words.extend(bottom.triangles.iter().map(|t| t.pointer));
        // The ring first: the host table is touched only once what says so
        // to the card has its place.
        let rung = (self.put(&vertices, frame), self.put(&list_words, frame), self.put(&[0; 4], frame));
        let (Some(vertices_at), Some(list_from), Some(hosts_from)) = rung else {
            return Err(back_out(self, None));
        };
        let Some(hosts_at) = self.hosts_cpu.put(address, list_at as u32, place.at) else {
            return Err(back_out(self, Some(Refused::Tables)));
        };
        let hosts_words: [u32; 4] = std::array::from_fn(|k| self.hosts_cpu.words[hosts_at + k]);
        self.ring.write(hosts_from, &hosts_words);
        Ok((place, (list_at, dwords), vertices_at, (list_at, list_from, dwords), (hosts_at as u64, hosts_from, 4)))
    }

    /// The pass for bottom levels the title builds at run time
    /// ([`rays::BOTTOM_PASS`] compiled, `cs_5_1`): without it
    /// [`Guide::moving`] and [`Guide::seen`] take nothing. The counts'
    /// buffer must hold [`COUNTS_BYTES`].
    pub fn with_bottom_pass(&mut self, pass: &[u8]) -> Result<(), String> {
        if self.counts.len < COUNTS_BYTES {
            return Err(format!("the counts' buffer holds {} bytes of {COUNTS_BYTES}", self.counts.len));
        }
        let root = bottom_root(&self.device)?;
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
            // SAFETY: a borrowed pointer the description never releases.
            pRootSignature: unsafe { std::mem::transmute_copy(&root) },
            CS: D3D12_SHADER_BYTECODE { pShaderBytecode: pass.as_ptr().cast(), BytecodeLength: pass.len() },
            ..Default::default()
        };
        // SAFETY: the description points at live data.
        let pso: ID3D12PipelineState = unsafe { self.device.CreateComputePipelineState(&desc) }.map_err(|e| format!("the bottom pass: {}", err(e)))?;
        // SAFETY: inside the caller's mapping.
        unsafe { std::ptr::write_bytes(self.counts.data.add(MOVING_AT as usize), 0, (COUNTS_BYTES - MOVING_AT) as usize) };
        self.bottom_pass = Some((root, pso));
        Ok(())
    }

    /// The title's builder wrote a structure at `address` in `frame`, and
    /// the layer knows no more of it (it sees the builder's dispatches, not
    /// the title's call). The pass counts it on the card ([`rays::census_job`]:
    /// whether a bottom level is there, and the triangles of each of its
    /// geometries); once the count has come back the host builds its own as
    /// if the title had described it ([`Guide::moving`]). A structure the
    /// host has already is only noted as written again.
    pub fn seen(&mut self, address: u64, frame: u64) {
        if self.bottom_pass.is_none() || address & 0xff != 0 || address >> 40 != 0 || address < 1 << 32 {
            return;
        }
        if let Some(had) = self.moved.get_mut(&address) {
            had.last = frame;
            return;
        }
        // (A static one is the queue's.)
        if self.jobs.contains_key(&address) {
            return;
        }
        if !self.census.contains_key(&address) && self.census.len() >= 4096 {
            return;
        }
        self.census.entry(address).or_insert(Census { last: frame, cap: 1 << 16, flight: None, wait: 0 }).last = frame;
    }

    /// The title is building or refitting a bottom level of its own in
    /// `frame`, with what it described to its builder: the host builds or
    /// refits its own of it in every list that walks from now on
    /// ([`Guide::record_moving`]). False when the host takes none of it (no
    /// room, over [`Config::moving_cap`], no triangles): its instances stay
    /// stand-in boxes. (The layer does not stand in front of the title's
    /// builder call yet: it goes by [`Guide::seen`]. The tests call this.)
    #[allow(dead_code)]
    pub fn moving(&mut self, what: Moving, frame: u64, room: &mut dyn FnMut(u64) -> bool) -> bool {
        self.take_moving(what, frame, room, true)
    }

    fn take_moving(&mut self, what: Moving, frame: u64, room: &mut dyn FnMut(u64) -> bool, described: bool) -> bool {
        if !self.live(frame) || self.bottom_pass.is_none() {
            return false;
        }
        let total: u64 = what.triangles.iter().map(|&t| u64::from(t)).sum();
        if total == 0 || total >= 1 << 22 || what.address & 0xff != 0 || what.address >> 40 != 0 {
            self.stats.moving_refused += 1;
            return false;
        }
        let total = total as u32;
        if self.jobs.contains_key(&what.address) {
            // (What was only counted does not take a static one's place.)
            if !described {
                return false;
            }
            // (The title describes another at this address: the static one
            // is the title's no more.)
            self.went(what.address);
        }
        if let Some((slot, ..)) = self.census.remove(&what.address).and_then(|c| c.flight) {
            self.census_free |= 1 << slot;
        }
        let mut bases = [!0u32; rays::MOVING_GEOMETRIES];
        let mut next = 0;
        for (g, &t) in what.triangles.iter().enumerate() {
            if t != 0 {
                (bases[g], next) = (next, next + t);
            }
        }
        if let Some(had) = self.moved.get_mut(&what.address)
            && had.triangles == total
            && had.bases == bases
        {
            had.last = frame;
            return true;
        }
        // Other triangles: another host structure, in the same place of the
        // host table, which the pass writes anew in the next list (the
        // structure before is offered by no list recorded from now on).
        let before = self.moved.remove(&what.address);
        let kept = before.as_ref().map(|m| m.hosts_at);
        if let Some(old) = before {
            self.moving_held -= old.place.len;
            self.let_go(frame, Freed::Place(old.place));
            self.let_go(frame, Freed::List(old.list.0, old.list.1));
        }
        let geometry = triangles_of(self.ring.buffer.at, total);
        let (result, scratch) = self.sizes(&moving_inputs(&geometry, false));
        let dwords = 1 + u64::from(total);
        let made = (|| {
            if result == 0 || self.moving_held + result > self.cfg.moving_cap || self.grow_scratch(scratch, frame).is_err() {
                return None;
            }
            let list = self.lists.take(dwords, 1).filter(|at| at + dwords < 1 << 24)?;
            let Ok(place) = self.pool.place(&self.device, result, room) else {
                self.lists.give(list, dwords);
                return None;
            };
            let Some(hosts_at) = kept.or_else(|| self.hosts_cpu.put(what.address, list as u32, place.at)) else {
                self.lists.give(list, dwords);
                if let Some(block) = self.pool.give(place) {
                    self.let_go(frame, Freed::Buffer(block));
                }
                return None;
            };
            Some(Moved {
                triangles: total,
                bases,
                place,
                list: (list, dwords),
                hosts_at,
                built: None,
                last: frame,
                sent: None,
                proven: false,
                // (Not every structure is built anew in the same frame.)
                refits: (what.address >> 8) as u32 % self.cfg.moving_rebuild.max(1),
                hosted: 0,
                on: false,
                described,
            })
        })();
        match made {
            Some(moved) => {
                self.moving_held += moved.place.len;
                self.moved.insert(what.address, moved);
                true
            }
            None => {
                // The place of the one before is marked gone on the card by
                // the next pass (a job of no triangles), and is free here.
                if let Some(at) = kept {
                    self.moving_gone.push((what.address, at));
                    self.hosts_cpu.words[at..at + 4].copy_from_slice(&[rays::HOSTS_GONE, 0, 0, 0]);
                }
                self.stats.moving_refused += 1;
                false
            }
        }
    }

    /// The host's structure of the title's run-time bottom level at
    /// `address` is let go: its table entries with the next list, its memory
    /// a while after.
    fn forget(&mut self, address: u64, _frame: u64) {
        if let Some(moved) = self.moved.remove(&address) {
            self.moving_held -= moved.place.len;
            self.leaving.push((address, moved.place, moved.list));
            self.stats.moving_gone += 1;
        }
    }

    /// Bottom levels the title builds at run time: held now, their
    /// triangles, the pool's bytes they hold, and those waiting for a count.
    pub fn moving_standing(&self) -> (usize, u64, u64, usize) {
        (self.moved.len(), self.moved.values().map(|m| u64::from(m.triangles)).sum(), self.moving_held, self.census.len())
    }

    /// The counts that have come back: a structure counted is taken as if
    /// the title had described it.
    fn counted_moving(&mut self, frame: u64, room: &mut dyn FnMut(u64) -> bool) {
        let mut found = Vec::new();
        for (&address, c) in &mut self.census {
            let Some((slot, stamp, asked)) = c.flight else { continue };
            // SAFETY: a slot inside the caller's mapping; shaders write it,
            // so the dwords are only numbers.
            let w: [u32; rays::CENSUS_DWORDS] =
                std::array::from_fn(|k| unsafe { self.counts.data.add(CENSUS_AT as usize + 4 * (rays::CENSUS_DWORDS * slot + k)).cast::<u32>().read_volatile() });
            if w[0] != stamp {
                // Not run yet; or its list never was.
                if frame > asked + 4 * self.cfg.quarantine {
                    (c.flight, c.wait) = (None, frame + 8);
                    self.census_free |= 1 << slot;
                }
                continue;
            }
            c.flight = None;
            self.census_free |= 1 << slot;
            let counts: [u32; rays::MOVING_GEOMETRIES] = std::array::from_fn(|g| w[2 + g]);
            if w[1] == !0 || w[11] != 0 || counts.iter().all(|&t| t == 0) {
                // No bottom level there (a top level, scratch memory), or
                // more geometries than the pass takes: looked at again much
                // later.
                self.stats.census_refused += 1;
                c.wait = frame + 600;
            } else if w[1] > c.cap {
                c.cap = w[1].next_power_of_two().min(1 << 22);
            } else if w[10] != 0 {
                // A page of it had no copy.
                c.wait = frame + 8;
            } else {
                found.push(Moving { address, triangles: counts });
            }
        }
        for what in found {
            let last = self.census.get(&what.address).map_or(frame, |c| c.last);
            if self.take_moving(what, frame, room, false) {
                if let Some(m) = self.moved.get_mut(&what.address) {
                    m.last = last;
                }
            } else {
                // (No room: asked again when much has changed.)
                self.census.insert(what.address, Census { last, cap: 1 << 16, flight: None, wait: frame + 120 });
            }
        }
        // Structures the pass found not to fit their count: one the title
        // described stays as it is (the pass turns it down in every list);
        // one that was counted is counted again.
        for k in 0..16 {
            // SAFETY: inside the caller's mapping.
            let at = unsafe { self.counts.data.add(MOVING_AT as usize + 8 + 4 * k).cast::<u32>() };
            // SAFETY: as above.
            let key = unsafe { at.read_volatile() };
            if key == 0 {
                continue;
            }
            // SAFETY: as above.
            unsafe { at.write_volatile(0) };
            let address = u64::from(key) << 8;
            if self.moved.get(&address).is_some_and(|m| !m.described && m.built.is_some_and(|b| b + 2 <= frame)) {
                let last = self.moved.get(&address).map_or(frame, |m| m.last);
                self.forget(address, frame);
                self.census.insert(address, Census { last, cap: 1 << 16, flight: None, wait: frame + 2 });
                self.stats.moving_misfit += 1;
            }
        }
    }

    /// Records, in a list that walks, the pass [`rays::BOTTOM_PASS`] and the
    /// driver's builds for the bottom levels the title builds at run time:
    /// before the list's top levels are made. Once a list.
    ///
    /// Every structure the title's builder wrote within
    /// [`Config::moving_fresh`] frames gets a row in every such list: the
    /// pass writes it anew from the title's structure as it is at that
    /// point of the list, and offers it to the top level's pass only then.
    /// So whatever order the lists of a frame are recorded and run in, a
    /// list walks a host structure that was made in that list after
    /// everything that ran before it. One the title has left alone longer
    /// stands as it was last written.
    ///
    /// The driver refits where a build of the structure is known to have
    /// been submitted and it was refitted fewer than
    /// [`Config::moving_rebuild`] times since; else it builds. A list takes
    /// at most [`Config::moving_refit`] triangles of refits and
    /// [`Config::moving_build`] of builds: what passes the marks is marked
    /// gone for that list (a stand-in box, the title walks it itself).
    ///
    /// Also asks the card for the counts of structures nobody described
    /// (once a frame), takes the counts that have come back, and lets go
    /// those the title has not written for `idle` frames. The list's view
    /// heap must be set; its compute root signature and pipeline are this
    /// module's afterwards when true is returned. An error: nothing was
    /// recorded, and what earlier lists left in the host table is not of
    /// this list: the caller has this list's walks go without the guide.
    pub fn record_moving(&mut self, list: &List, frame: u64, io: TopIo, idle: u64, room: &mut dyn FnMut(u64) -> bool) -> Result<bool, String> {
        if !self.live(frame) {
            return Ok(false);
        }
        let Some((root, pso)) = self.bottom_pass.clone() else { return Ok(false) };
        self.counted_moving(frame, room);
        let stale: Vec<u64> = self.moved.iter().filter(|(_, m)| m.last + idle < frame).map(|(&a, _)| a).collect();
        for address in stale {
            self.forget(address, frame);
        }
        self.census.retain(|_, c| c.flight.is_some() || c.last + idle >= frame);
        let gone = std::mem::take(&mut self.moving_gone);
        let mut addresses: Vec<u64> = self.moved.keys().copied().collect();
        addresses.sort_unstable();
        let most_rows = rays::MOVING_ROWS as usize - CENSUS_SLOTS - gone.len().min(256);
        let (fresh, quarantine, wait_sent) = (self.cfg.moving_fresh, self.cfg.quarantine, self.wait_sent);
        let (mut refit_left, mut build_left) = (i64::from(self.cfg.moving_refit), i64::from(self.cfg.moving_build));
        let (mut refits, mut builds_n) = (0u32, 0u32);
        let mut table: Vec<u32> = Vec::new();
        // What a row decided: the address, and (a refit, the vertices' byte)
        // or none for a row that marks it gone.
        let mut rows: Vec<(u64, Option<(bool, u64)>)> = Vec::new();
        let (mut at, mut most) = (0u64, 0u32);
        let mut over = Vec::new();
        for address in addresses {
            let Some(m) = self.moved.get(&address) else { continue };
            if m.last + fresh <= frame && m.on && m.hosted >= m.last + 2 {
                continue;
            }
            if rows.len() >= most_rows {
                over.push(address);
                continue;
            }
            let proven = m.proven
                || if wait_sent { m.sent.as_ref().is_some_and(|t| t.load(Ordering::Relaxed)) } else { m.built.is_some_and(|b| b + quarantine <= frame) };
            let t = i64::from(m.triangles);
            let build = !proven || m.refits >= self.cfg.moving_rebuild;
            let decided = if build && (builds_n == 0 || build_left >= t) {
                Some(false)
            } else if proven && (refits == 0 || refit_left >= t) {
                Some(true)
            } else {
                None
            };
            match decided {
                Some(refit) => {
                    if refit {
                        (refit_left, refits) = (refit_left - t, refits + 1);
                    } else {
                        (build_left, builds_n) = (build_left - t, builds_n + 1);
                    }
                    table.extend(rays::moving_job(address, m.triangles, at as u32, m.list.0 as u32, 4 * m.hosts_at as u32, m.place.at, m.bases));
                    rows.push((address, Some((refit, at))));
                    at += 36 * u64::from(m.triangles);
                    most = most.max(m.triangles);
                }
                None => {
                    table.extend(rays::moving_job(address, 0, 0, 0, 4 * m.hosts_at as u32, 0, [!0; rays::MOVING_GEOMETRIES]));
                    rows.push((address, None));
                }
            }
        }
        // (More of them than the pass has rows: the last are let go.)
        for address in over {
            self.forget(address, frame);
        }
        // (A job of no triangles: the pass marks its place gone.)
        for &(address, place) in gone.iter().take(256) {
            table.extend(rays::moving_job(address, 0, 0, 0, 4 * place as u32, 0, [!0; rays::MOVING_GEOMETRIES]));
        }
        // The counts asked for in this frame.
        let stamp = self.stamp.wrapping_add(1).max(1);
        let mut asked: Vec<(u64, usize)> = Vec::new();
        let mut most_counted = 0u32;
        if self.census_frame != frame {
            let mut wanted: Vec<(u64, u32)> = self
                .census
                .iter()
                .filter(|(_, c)| c.flight.is_none() && c.wait <= frame && c.last + fresh > frame)
                .map(|(&a, c)| (a, c.cap))
                .collect();
            wanted.sort_unstable();
            for (address, cap) in wanted.into_iter().take(16) {
                if self.census_free == 0 {
                    break;
                }
                let slot = self.census_free.trailing_zeros() as usize;
                self.census_free &= !(1 << slot);
                // SAFETY: a slot inside the caller's mapping.
                unsafe { std::ptr::write_bytes(self.counts.data.add(CENSUS_AT as usize + 4 * rays::CENSUS_DWORDS * slot), 0, 4 * rays::CENSUS_DWORDS) };
                table.extend(rays::census_job(address, cap, (CENSUS_AT as usize + 4 * rays::CENSUS_DWORDS * slot) as u32));
                asked.push((address, slot));
                most_counted = most_counted.max(cap);
            }
        }
        let give_back = |g: &mut Self, gone: Vec<(u64, usize)>, asked: &[(u64, usize)]| {
            g.moving_gone = gone;
            for &(_, slot) in asked {
                g.census_free |= 1 << slot;
            }
        };
        let jobs = (table.len() / 16) as u32;
        if jobs == 0 {
            return Ok(false);
        }
        // The vertices of all of them, one after another.
        if self.moving_vertices.as_ref().is_none_or(|v| v.1 < at) {
            let len = at.next_power_of_two().max(1 << 20);
            let made = match buffer(&self.device, len, D3D12_HEAP_TYPE_DEFAULT, WRITTEN) {
                Ok(made) => made,
                Err(e) => {
                    give_back(self, gone, &asked);
                    return Err(e);
                }
            };
            if let Some((old, _)) = self.moving_vertices.replace((made, len)) {
                self.let_go(frame, Freed::Buffer(old.res));
            }
        }
        let Some(vertices) = self.moving_vertices.as_ref().map(|v| (v.0.res.clone(), v.0.at)) else {
            give_back(self, gone, &asked);
            return Err("no vertex buffer".into());
        };
        let Some(table_at) = self.put(&table, frame) else {
            give_back(self, gone, &asked);
            return Err("no room on the way for the jobs of the bottom levels' pass".into());
        };
        // From here on the list has the pass: the books follow.
        self.stamp = stamp;
        if !asked.is_empty() {
            self.census_frame = frame;
        }
        for &(address, slot) in &asked {
            if let Some(c) = self.census.get_mut(&address) {
                c.flight = Some((slot, stamp, frame));
            }
        }
        self.stats.census_asked += asked.len() as u64;
        self.stats.moving_rows += u64::from(jobs);
        self.stats.moving_lists += 1;
        let token = rows.iter().any(|r| matches!(r.1, Some((false, _)))).then(|| self.token());
        // The builds: destination, vertices, triangles, a refit.
        let mut builds = Vec::with_capacity(rows.len());
        for &(address, decided) in &rows {
            let Some(m) = self.moved.get_mut(&address) else { continue };
            m.hosted = frame;
            m.on = decided.is_some();
            let Some((refit, at)) = decided else {
                self.stats.moving_over += 1;
                continue;
            };
            if !m.proven && m.sent.as_ref().is_some_and(|t| t.load(Ordering::Relaxed)) {
                m.proven = true;
            }
            if refit {
                m.refits += 1;
                self.stats.moving_refits += 1;
            } else {
                m.refits = 0;
                m.built.get_or_insert(frame);
                if !m.proven {
                    m.sent = token.clone();
                }
                self.stats.moving_builds += 1;
            }
            self.stats.moving_triangles += u64::from(m.triangles);
            builds.push((m.place.at, vertices.1 + at, m.triangles, refit));
        }
        let constants = |phase: u32| [jobs, MOVING_AT as u32, phase, stamp];
        // SAFETY: an open list with its view heap set; this module's root
        // signature and pipeline; live buffers in the states it keeps them.
        unsafe {
            list.ResourceBarrier(&[transition(&self.table.res, READ, WRITTEN), transition(&self.hosts.res, READ, WRITTEN)]);
            list.SetComputeRootSignature(&root);
            list.SetPipelineState(&pso);
            list.SetComputeRootShaderResourceView(1, io.directory);
            list.SetComputeRootShaderResourceView(2, self.ring.buffer.at + table_at);
            list.SetComputeRootUnorderedAccessView(3, vertices.1);
            list.SetComputeRootUnorderedAccessView(4, self.hosts.at);
            list.SetComputeRootUnorderedAccessView(5, self.table.at);
            list.SetComputeRootUnorderedAccessView(6, io.faults);
            list.SetComputeRootUnorderedAccessView(7, self.counts.res.GetGPUVirtualAddress());
            list.SetComputeRootDescriptorTable(8, io.copies);
            // Clear, fill, commit: each sees what the one before wrote.
            if most != 0 {
                list.SetComputeRoot32BitConstants(0, 4, constants(0).as_ptr().cast(), 0);
                list.Dispatch(most.div_ceil(64), jobs, 1);
                list.ResourceBarrier(&[settled()]);
            }
            list.SetComputeRoot32BitConstants(0, 4, constants(1).as_ptr().cast(), 0);
            list.Dispatch(most.max(most_counted).div_ceil(64).max(1), jobs, 1);
            list.ResourceBarrier(&[settled()]);
            list.SetComputeRoot32BitConstants(0, 4, constants(2).as_ptr().cast(), 0);
            list.Dispatch(1, jobs, 1);
            list.ResourceBarrier(&[
                settled(),
                transition(&self.table.res, WRITTEN, READ),
                transition(&self.hosts.res, WRITTEN, READ),
                transition(&vertices.0, WRITTEN, READ),
            ]);
            for &(to, from, triangles, refit) in &builds {
                let geometry = triangles_of(from, triangles);
                list.BuildRaytracingAccelerationStructure(
                    &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                        DestAccelerationStructureData: to,
                        Inputs: moving_inputs(&geometry, refit),
                        SourceAccelerationStructureData: if refit { to } else { 0 },
                        ScratchAccelerationStructureData: self.scratch.at,
                    },
                    None,
                );
                list.ResourceBarrier(&[settled()]);
            }
            list.ResourceBarrier(&[transition(&vertices.0, READ, WRITTEN)]);
        }
        Ok(true)
    }

    /// The host top level for the title's at `address`, about to be walked
    /// in `frame`: an entry of the table (the one it had, a free one, or the
    /// one longest unused that no walk of this frame has: its buffers serve
    /// the new top level as they are, every use builds it anew), made for as
    /// many instances as the pass last counted there. None before
    /// [`Guide::begin`]'s frames have passed, or with four other top levels
    /// walked in this frame: that walk is the title's own.
    pub fn top(&mut self, address: u64, frame: u64) -> Result<Option<TopUse>, String> {
        if !self.live(frame) {
            return Ok(None);
        }
        let found = self.tops.iter().position(|t| t.as_ref().is_some_and(|t| t.address == address));
        let index = found
            .or_else(|| self.tops.iter().position(Option::is_none))
            .or_else(|| (0..TOPS).filter(|&n| self.tops[n].as_ref().is_some_and(|t| t.used < frame)).min_by_key(|&n| self.tops[n].as_ref().map(|t| t.used)));
        let Some(index) = index else {
            self.stats.tops_over += 1;
            return Ok(None);
        };
        // As many instances as the title's top level had when it was last
        // walked, and a quarter more.
        let counted = if found.is_some() { self.counted(index)[0] } else { 0 };
        let had = self.tops[index].as_ref().map(|t| t.most);
        let most = match had {
            Some(most) if counted == !0 || counted <= most => most,
            Some(_) => (counted.saturating_add(counted / 4)).next_power_of_two().min(MOST),
            None => self.cfg.most,
        };
        if had != Some(most) {
            let inputs = top_inputs(most, 0);
            let (result, scratch) = self.sizes(&inputs);
            if result == 0 {
                return Err("the driver builds no top level".into());
            }
            self.grow_scratch(scratch, frame)?;
            let made = Top {
                address,
                used: frame,
                most,
                generation: self.tops[index].as_ref().map_or(0, |t| t.generation) + 1,
                descs: buffer(&self.device, 64 * u64::from(most), D3D12_HEAP_TYPE_DEFAULT, WRITTEN)?,
                tlas: buffer(&self.device, result, D3D12_HEAP_TYPE_DEFAULT, STRUCTURE)?,
            };
            if let Some(old) = self.tops[index].replace(made) {
                self.let_go(frame, Freed::Buffer(old.descs.res));
                self.let_go(frame, Freed::Buffer(old.tlas.res));
            }
            // SAFETY: inside the caller's mapping.
            unsafe { self.counts.data.cast::<u32>().add(COUNTS as usize * index).write_volatile(0) };
        }
        let Some(top) = self.tops[index].as_mut() else { return Ok(None) };
        (top.address, top.used) = (address, frame);
        Ok(Some(TopUse { index, generation: top.generation, location: top.tlas.at }))
    }

    /// Records the making of a host top level before a walk: the pass
    /// writes its instances and its entry of the guide table from the
    /// title's top level as the card holds it at that point of the list,
    /// and the driver builds from them. The list's view heap must be set
    /// (`io.copies` is in it); its compute root signature and pipeline are
    /// this module's afterwards.
    pub fn record_top(&mut self, list: &List, used: &TopUse, io: TopIo) {
        let Some(top) = self.tops[used.index].as_ref().filter(|t| t.generation == used.generation) else { return };
        let constants = [
            top.address as u32,
            (top.address >> 32) as u32,
            top.most,
            self.stand_in.at as u32,
            (self.stand_in.at >> 32) as u32,
            used.index as u32,
            io.slot,
            (4 * COUNTS as usize * used.index) as u32,
        ];
        // SAFETY: an open list with its view heap set; this module's root
        // signature and pipeline; live buffers in the states it keeps them.
        unsafe {
            list.ResourceBarrier(&[transition(&self.table.res, READ, WRITTEN)]);
            list.SetComputeRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            list.SetComputeRoot32BitConstants(0, constants.len() as u32, constants.as_ptr().cast(), 0);
            list.SetComputeRootShaderResourceView(1, io.directory);
            list.SetComputeRootShaderResourceView(2, self.hosts.at);
            list.SetComputeRootUnorderedAccessView(3, top.descs.at);
            list.SetComputeRootUnorderedAccessView(4, self.counts.res.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(5, self.table.at);
            list.SetComputeRootUnorderedAccessView(6, io.faults);
            list.SetComputeRootDescriptorTable(7, io.copies);
            list.Dispatch(top.most.div_ceil(64), 1, 1);
            list.ResourceBarrier(&[transition(&self.table.res, WRITTEN, READ), transition(&top.descs.res, WRITTEN, READ)]);
            list.BuildRaytracingAccelerationStructure(
                &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                    DestAccelerationStructureData: top.tlas.at,
                    Inputs: top_inputs(top.most, top.descs.at),
                    SourceAccelerationStructureData: 0,
                    ScratchAccelerationStructureData: self.scratch.at,
                },
                None,
            );
            list.ResourceBarrier(&[settled(), transition(&top.descs.res, READ, WRITTEN)]);
        }
        self.stats.tops += 1;
    }
}

/// The driver's inputs of a bottom level of `count` triangles whose
/// vertices (three of three floats each) lie at `vertices`.
fn triangles_of(vertices: u64, count: u32) -> D3D12_RAYTRACING_GEOMETRY_DESC {
    D3D12_RAYTRACING_GEOMETRY_DESC {
        Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
        Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_NONE,
        Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
            Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                Transform3x4: 0,
                IndexFormat: DXGI_FORMAT_UNKNOWN,
                VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                IndexCount: 0,
                VertexCount: 3 * count,
                IndexBuffer: 0,
                VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE { StartAddress: vertices, StrideInBytes: 12 },
            },
        },
    }
}

fn bottom_inputs(geometry: &D3D12_RAYTRACING_GEOMETRY_DESC) -> D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
        Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
        NumDescs: 1,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { pGeometryDescs: geometry },
    }
}

/// A bottom level the title builds at run time: built to be refitted.
fn moving_inputs(geometry: &D3D12_RAYTRACING_GEOMETRY_DESC, refit: bool) -> D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
    let mut flags = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD | D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    if refit {
        flags |= D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;
    }
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
        Flags: flags,
        NumDescs: 1,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { pGeometryDescs: geometry },
    }
}

/// [`rays::BOTTOM_PASS`]'s root signature: 0 its constants, 1 the directory,
/// 2 the jobs, 3 the vertices, 4 the host table, 5 the guide table, 6 the
/// notes of pages without a copy, 7 the counts, 8 the copies' views.
fn bottom_root(device: &ID3D12Device5) -> Result<ID3D12RootSignature, String> {
    let range = D3D12_DESCRIPTOR_RANGE1 {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        NumDescriptors: 4096,
        BaseShaderRegister: 0,
        RegisterSpace: 1,
        Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
        OffsetInDescriptorsFromTableStart: 0,
    };
    let view = |kind, register| D3D12_ROOT_PARAMETER1 {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 { ShaderRegister: register, RegisterSpace: 0, Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let params = [
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 4 } },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        view(D3D12_ROOT_PARAMETER_TYPE_SRV, 0),
        view(D3D12_ROOT_PARAMETER_TYPE_SRV, 1),
        view(D3D12_ROOT_PARAMETER_TYPE_UAV, 0),
        view(D3D12_ROOT_PARAMETER_TYPE_UAV, 1),
        view(D3D12_ROOT_PARAMETER_TYPE_UAV, 2),
        view(D3D12_ROOT_PARAMETER_TYPE_UAV, 3),
        view(D3D12_ROOT_PARAMETER_TYPE_UAV, 4),
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 { NumDescriptorRanges: 1, pDescriptorRanges: &range },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
            },
        },
    };
    // SAFETY: the description points at live arrays.
    unsafe {
        let mut blob = None;
        D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).map_err(err)?;
        let blob = blob.ok_or("no root signature blob")?;
        let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        device.CreateRootSignature(0, bytes).map_err(err)
    }
}

/// A top level of `most` instances, built anew at every use.
fn top_inputs(most: u32, descriptions: u64) -> D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
        Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
        Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_BUILD,
        NumDescs: most,
        DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
        Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { InstanceDescs: descriptions },
    }
}

/// The pass's root signature ([`rays::TOP_PASS`]'s bindings): 0 its eight
/// constants, 1 the directory, 2 the host table, 3 the descriptions, 4 the
/// counts, 5 the guide table, 6 the notes of pages without a copy, 7 the
/// copies' views.
fn pass_root(device: &ID3D12Device5) -> Result<ID3D12RootSignature, String> {
    let range = D3D12_DESCRIPTOR_RANGE1 {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
        NumDescriptors: 4096,
        BaseShaderRegister: 0,
        RegisterSpace: 1,
        Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
        OffsetInDescriptorsFromTableStart: 0,
    };
    let view = |kind, register| D3D12_ROOT_PARAMETER1 {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 { ShaderRegister: register, RegisterSpace: 0, Flags: D3D12_ROOT_DESCRIPTOR_FLAG_DATA_VOLATILE },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let params = [
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 { Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 8 } },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        view(D3D12_ROOT_PARAMETER_TYPE_SRV, 0),
        view(D3D12_ROOT_PARAMETER_TYPE_SRV, 1),
        view(D3D12_ROOT_PARAMETER_TYPE_UAV, 0),
        view(D3D12_ROOT_PARAMETER_TYPE_UAV, 1),
        view(D3D12_ROOT_PARAMETER_TYPE_UAV, 2),
        view(D3D12_ROOT_PARAMETER_TYPE_UAV, 3),
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE1 { NumDescriptorRanges: 1, pDescriptorRanges: &range },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
        Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
        Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
            Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                NumStaticSamplers: 0,
                pStaticSamplers: std::ptr::null(),
                Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
            },
        },
    };
    // SAFETY: the description points at live arrays.
    unsafe {
        let mut blob = None;
        D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).map_err(err)?;
        let blob = blob.ok_or("no root signature blob")?;
        let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        device.CreateRootSignature(0, bytes).map_err(err)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_are_taken_and_given_back_whole() {
        let mut r = Ranges::new(1000);
        let a = r.take(100, 1).unwrap();
        let b = r.take(300, 256).unwrap();
        let c = r.take(200, 1).unwrap();
        assert_eq!((a, b, c), (0, 256, 556));
        // (The hole before the aligned run is found again.)
        assert_eq!(r.take(156, 1), Some(100));
        assert_eq!(r.held(), 756);
        assert_eq!(r.take(300, 1), None);
        r.give(b, 300);
        r.give(a, 100);
        r.give(100, 156);
        assert_eq!(r.take(556, 1), Some(0));
        r.give(0, 556);
        r.give(c, 200);
        assert_eq!((r.held(), r.take(1000, 8)), (0, Some(0)));
        assert_eq!(r.take(0, 1), None);
    }

    #[test]
    fn the_host_table_takes_out_without_losing_its_neighbours() {
        let mut h = Hosts::new(64);
        let bottoms: Vec<u64> = (1..=40u64).map(|n| 0xD_0000_0000 + n * 0x1_0100).collect();
        for (n, &b) in bottoms.iter().enumerate() {
            assert!(h.put(b, n as u32, 0x7_0000_0000 + 256 * n as u64).is_some());
        }
        for &b in bottoms.iter().step_by(2) {
            assert!(h.take(b).is_some());
        }
        for (n, &b) in bottoms.iter().enumerate() {
            let want = (n % 2 == 1).then_some((n as u32, 0x7_0000_0000 + 256 * n as u64));
            assert_eq!(rays::hosts_find(&h.words, b), want, "{n}");
        }
        // A place let go is taken again.
        assert!(h.put(bottoms[0], 99, 0x1000).is_some());
        assert_eq!(rays::hosts_find(&h.words, bottoms[0]), Some((99, 0x1000)));
        assert_eq!(h.take(0xE_0000_0000), None);
    }

    #[test]
    fn the_pool_stops_at_its_cap() {
        let mut ranges = Ranges::new(1 << 16);
        assert!(ranges.take(1 << 16, 256).is_some());
        let pool = Pool::new(3 << 16, 1 << 16);
        assert_eq!(pool.bytes(), (0, 0));
    }
}
