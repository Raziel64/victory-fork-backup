//! The title's ray-tracing structures read back out of title memory, for the
//! host's own ray-tracing hardware (`VICTORY_RAY_GUIDE`, `emit/intersect.rs`):
//!
//! - [`bottom`] and [`top`]: a structure's bytes decoded into plain triangles
//!   and instances, each with the title's node it came from;
//! - [`walk`]: the walk a title program makes over the structures, on the
//!   CPU, turn by turn, with [`crate::bvh::intersect`] for every node test
//!   (the arithmetic the emitted code does). It is the reference host
//!   structures built from the decoded data are compared against, and it
//!   counts the turns of the title's loop;
//! - [`guide_table`]: the buffer the guided node test reads
//!   ([`crate::emit::SLOT_GUIDE`]), and [`Guided`], that node test on the CPU
//!   over any source of candidates.
//!
//! # Formats
//!
//! Nodes are 64 bytes; a node pointer is the node's number from the
//! structure's first byte times 8 plus a kind ([`crate::bvh`]). Box and
//! triangle nodes are the instruction's (`bvh.rs`). What the title keeps in
//! the bytes the instruction does not read, as its walking programs read
//! them (f607780aa8e502d0 and f22a76fcb967ee59):
//!
//! - **header** (node 0), dwords: 0 the version (6, low half) and marks
//!   (high half: [`MARK_BOTTOM`], [`MARK_PACKED`]); 1 the size in nodes; 3 the
//!   number of elements; 4-9 the bounds; 12 the root's pointer; 13 the byte
//!   offset of the elements; 14 the byte offset of the parent table.
//! - **triangle node**: vertices 0-3 as three floats each from byte 0;
//!   dword 12: bit 0 the geometry's opaque flag, bit 7 "a second triangle
//!   follows" (the walk goes from triangle 0 to triangle 1 of the node),
//!   bits 31:8 the geometry index; dwords 13 and 14 the primitive index of
//!   triangle 0 and 1; dword 15 the instruction's vertex order bytes.
//! - **instance node** (128 bytes): dwords 0-11 the object-to-world matrix
//!   (three rows of four); 12 the id (24 bits) and the mask; 13 the
//!   contribution to the hit group index (24 bits) and the flags; 14 the
//!   bottom level's root pointer (bit 31: no rotation, bit 30: static; the
//!   low 30 bits all set: none); 15 the bottom level's address in 256-byte
//!   units; 16-18 the world-to-object move; 19-27 the world-to-object rows.
//!   The first 64 bytes are a host instance description as they are, but for
//!   the structure's address.
//!
//! A packed structure (as the title ships its static ones) is not decoded:
//! the title's own programs unpack it in place on the card first.

use std::collections::{HashMap, HashSet};

use crate::bvh::{self, NONE, Ray};

/// The header's version.
pub const VERSION: u32 = 6;
/// Header mark: a bottom level.
pub const MARK_BOTTOM: u32 = 0x8000;
/// Header mark: still packed.
pub const MARK_PACKED: u32 = 0x100;
/// Header dword 15: the nodes are scaled, moved, turned (which of 64 turns).
pub const SPACE_SCALED: u32 = 0x8000_0000;
pub const SPACE_MOVED: u32 = 0x4000_0000;
pub const SPACE_TURNED: u32 = 0x3F00_0000;
/// The instance flags a host instance may carry: no culling, front faces
/// counter-clockwise, force opaque, force not opaque. The console has more
/// of its own in the same byte (0x40 and 0xC0: the instance is pulled up
/// into the top level as 4 or 16 nodes), which a host build must not see.
pub const HOST_FLAGS: u8 = 0x0F;
/// An instance's root pointer (30 bits) that names no bottom level.
pub const NO_ROOT: u32 = 0x3fff_ffff;
/// Instance flags, as the host's: no culling of this instance's triangles.
pub const FLAG_CULL_DISABLE: u8 = 1;
/// Instance flags: front faces run counter-clockwise.
pub const FLAG_FRONT_CCW: u8 = 2;
/// The page title memory is kept in.
pub const PAGE: u64 = 1 << 16;

/// Why a structure was not decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// No copy of the memory at this address.
    Unmapped(u64),
    /// No header of this format at this address.
    NoHeader(u64),
    /// The structure at this address is still packed.
    Packed(u64),
    /// A bottom level where a top level was asked for, or the other way.
    WrongLevel(u64),
    /// More nodes are reachable than the header says the structure has.
    Loop(u64),
}

/// Title memory by address.
pub trait Memory {
    /// `words.len()` dwords from `address` (a multiple of 4); `false` when
    /// any of them has no copy.
    fn read(&self, address: u64, words: &mut [u32]) -> bool;
}

/// One run of title memory.
#[derive(Debug, Clone, Copy)]
pub struct Span<'a> {
    pub base: u64,
    pub words: &'a [u32],
}

impl Memory for Span<'_> {
    fn read(&self, address: u64, words: &mut [u32]) -> bool {
        let Some(at) = address.checked_sub(self.base) else {
            return false;
        };
        let at = (at / 4) as usize;
        match self.words.get(at..at + words.len()) {
            Some(w) if address % 4 == 0 => {
                words.copy_from_slice(w);
                true
            }
            _ => false,
        }
    }
}

/// Pages of title memory, each wherever it is kept.
#[derive(Debug, Default, Clone)]
pub struct Pages {
    pages: HashMap<u64, Vec<u32>>,
}

impl Pages {
    /// Puts `words` at `address` (a multiple of 4), over what was there.
    pub fn write(&mut self, address: u64, words: &[u32]) {
        for (n, &w) in words.iter().enumerate() {
            let a = address + 4 * n as u64;
            let page = self.pages.entry(a / PAGE).or_insert_with(|| vec![0; (PAGE / 4) as usize]);
            page[(a % PAGE / 4) as usize] = w;
        }
    }

    /// The pages held, by number, in order.
    pub fn numbers(&self) -> Vec<u64> {
        let mut n: Vec<u64> = self.pages.keys().copied().collect();
        n.sort_unstable();
        n
    }

    /// The dwords of page `number`.
    pub fn page(&self, number: u64) -> Option<&[u32]> {
        self.pages.get(&number).map(Vec::as_slice)
    }
}

impl Memory for Pages {
    fn read(&self, address: u64, words: &mut [u32]) -> bool {
        if address % 4 != 0 {
            return false;
        }
        for (n, w) in words.iter_mut().enumerate() {
            let a = address + 4 * n as u64;
            match self.pages.get(&(a / PAGE)) {
                Some(page) => *w = page[(a % PAGE / 4) as usize],
                None => return false,
            }
        }
        true
    }
}

/// A dump a diagnostics build wrote (`VICTORY_MEMORY_DUMP`, `victory-gpu`
/// `memory.rs`): `pages.txt` and the arenas as the card held them; and, of
/// a structure the walking programs were given that lies in a title
/// buffer, that buffer's pages as title memory had them (`title.bin`, lines
/// `title <address> <offset>`; the stage-0 build writes them).
#[derive(Debug, Default)]
pub struct Dump {
    /// The pages, read whole.
    pub pages: Pages,
    /// The structures the walking programs were given (top levels).
    pub walked: Vec<u64>,
    /// The packed (static) structures the builder came to.
    pub packed: Vec<u64>,
    /// Pages `pages.txt` lists that no file holds.
    pub absent: u32,
}

impl Dump {
    /// Reads the dump in `folder`: every page its `pages.txt` lists.
    pub fn open(folder: &std::path::Path) -> std::io::Result<Self> {
        use std::io::{Read, Seek, SeekFrom};
        let text = std::fs::read_to_string(folder.join("pages.txt"))?;
        let mut dump = Dump::default();
        let hex = |s: &str| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok();
        // (None: an arena that came after the copies were taken has no
        // file; its pages stay out.)
        let mut files: HashMap<u64, Option<std::fs::File>> = HashMap::new();
        let mut bytes = vec![0u8; PAGE as usize];
        for line in text.lines() {
            let w: Vec<&str> = line.split_whitespace().collect();
            match w[..] {
                ["walked", a] => dump.walked.extend(hex(a)),
                ["packed", a, ..] => dump.packed.extend(hex(a)),
                ["title", address, offset] => {
                    let (Some(address), Some(offset)) = (hex(address), hex(offset)) else { continue };
                    let Ok(mut file) = std::fs::File::open(folder.join("title.bin")) else { continue };
                    file.seek(SeekFrom::Start(offset))?;
                    if file.read_exact(&mut bytes).is_err() {
                        continue;
                    }
                    let words: Vec<u32> =
                        bytes.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                    // (An arena's page of the same address, if there is
                    // one, is the card's: it comes later in the file or
                    // came before, and is not written over.)
                    if dump.pages.page(address / PAGE).is_none() {
                        dump.pages.write(address / PAGE * PAGE, &words);
                    }
                }
                [address, arena, offset] if address.starts_with("0x") => {
                    let (Some(address), Ok(arena), Some(offset)) = (hex(address), arena.parse::<u64>(), hex(offset)) else {
                        continue;
                    };
                    let file = files
                        .entry(arena)
                        .or_insert_with(|| std::fs::File::open(folder.join(format!("arena-{arena}.bin"))).ok());
                    let Some(file) = file else {
                        dump.absent += 1;
                        continue;
                    };
                    file.seek(SeekFrom::Start(offset))?;
                    // (A page past the end of a shortened dump stays out.)
                    if file.read_exact(&mut bytes).is_err() {
                        dump.absent += 1;
                        continue;
                    }
                    let words: Vec<u32> =
                        bytes.chunks_exact(4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
                    dump.pages.write(address / PAGE * PAGE, &words);
                }
                _ => {}
            }
        }
        Ok(dump)
    }
}

/// A structure's header.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Header {
    pub address: u64,
    pub marks: u32,
    /// Size in 64-byte nodes.
    pub nodes: u32,
    pub elements: u32,
    pub bounds: [f32; 6],
    /// The root's pointer.
    pub root: u32,
    /// Byte offset of the elements (triangle nodes, instance nodes).
    pub elements_at: u32,
    /// Byte offset of the parent table.
    pub parents_at: u32,
    /// The space the bottom level's boxes and vertices are stored in
    /// (dword 15): [`SPACE_SCALED`], [`SPACE_MOVED`], [`SPACE_TURNED`], and
    /// three scale exponents in the low 24 bits.
    pub space: u32,
}

impl Header {
    pub fn bottom(&self) -> bool {
        self.marks & MARK_BOTTOM != 0
    }

    /// Whether the nodes are in a space of the structure's own, not the
    /// object's (the title's builder may scale a bottom level by powers of
    /// two, move it by its bounds' centre and turn it, so that its boxes fit
    /// 16-bit floats). An instance's world-to-object matrix at byte 64
    /// leads into that space; its object-to-world matrix at byte 0 does not.
    pub fn own_space(&self) -> bool {
        self.space & (SPACE_SCALED | SPACE_MOVED | SPACE_TURNED) != 0
    }

    /// A stored vertex in object space: times the inverse scale, plus the
    /// centre, turned back (the order the title's own code undoes it in).
    pub fn to_object(&self, v: [f32; 3]) -> [f32; 3] {
        if !self.own_space() {
            return v;
        }
        let mut v = v;
        if self.space & SPACE_SCALED != 0 {
            // The inverse of 2^(e - 127) per axis, by its exponent byte.
            let packed = 0x00FE_FEFEu32.wrapping_sub(self.space & 0x00FF_FFFF);
            for (a, x) in v.iter_mut().enumerate() {
                *x *= f32::from_bits((packed >> (8 * a) & 0xFF) << 23);
            }
        }
        if self.space & SPACE_MOVED != 0 {
            for (a, x) in v.iter_mut().enumerate() {
                let (low, high) = (self.bounds[a], self.bounds[3 + a]);
                if low > 0.0 || high < 0.0 {
                    *x += (low + high) * 0.5;
                }
            }
        }
        let turn = (self.space & SPACE_TURNED) >> 24;
        if turn != 0 {
            // Three quarter-turn-in-four-steps angles, about x, y and z;
            // the matrix is z * y * x and its inverse the transpose.
            const COS: [f32; 5] = [1.0, 0.923_879_53, 0.707_106_78, 0.382_683_43, 0.0];
            let (x, y, z) = ((turn & 3) as usize, (turn >> 2 & 3) as usize, (turn >> 4 & 3) as usize);
            let mx = [[1.0, 0.0, 0.0], [0.0, COS[x], COS[4 - x]], [0.0, -COS[4 - x], COS[x]]];
            let my = [[COS[y], 0.0, -COS[4 - y]], [0.0, 1.0, 0.0], [COS[4 - y], 0.0, COS[y]]];
            let mz = [[COS[z], COS[4 - z], 0.0], [-COS[4 - z], COS[z], 0.0], [0.0, 0.0, 1.0]];
            let mul = |a: [[f32; 3]; 3], b: [[f32; 3]; 3]| -> [[f32; 3]; 3] {
                std::array::from_fn(|i| std::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]))
            };
            let m = mul(mz, mul(my, mx));
            v = std::array::from_fn(|i| m[0][i] * v[0] + m[1][i] * v[1] + m[2][i] * v[2]);
        }
        v
    }
}

/// The header of the structure at `address`.
pub fn header(mem: &impl Memory, address: u64) -> Result<Header, Error> {
    let mut w = [0u32; 16];
    if !mem.read(address, &mut w) {
        return Err(Error::Unmapped(address));
    }
    let marks = w[0] >> 16;
    // (The size the layer takes for one: `victory-gpu` `memory.rs`.)
    if w[0] & 0xffff != VERSION || w[1] == 0 || w[1] >= 1 << 21 {
        return Err(Error::NoHeader(address));
    }
    if marks & MARK_PACKED != 0 {
        return Err(Error::Packed(address));
    }
    Ok(Header {
        address,
        marks,
        nodes: w[1],
        elements: w[3],
        bounds: std::array::from_fn(|k| f32::from_bits(w[4 + k])),
        root: w[12],
        elements_at: w[13],
        parents_at: w[14],
        space: w[15],
    })
}

/// A triangle of a bottom level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle {
    /// The vertices in the order the node test takes them: the one neither
    /// barycentric weighs, then I's, then J's ([`bvh::triangle_vertices`]).
    /// In OBJECT space ([`Header::to_object`]): what a host structure is
    /// built of, under an instance's object-to-world matrix. The node holds
    /// them in the structure's own space where it has one.
    pub vertices: [[f32; 3]; 3],
    /// The address of the triangle's node.
    pub node: u64,
    /// The triangle's pointer in its structure: the node's number times 8
    /// plus the triangle's slot in the node (0 or 1).
    pub pointer: u32,
    pub opaque: bool,
    pub geometry: u32,
    pub primitive: u32,
}

/// A bottom level decoded.
#[derive(Debug, Clone, PartialEq)]
pub struct Bottom {
    pub header: Header,
    /// The triangles, in the order the walk from the root finds them.
    pub triangles: Vec<Triangle>,
    /// Box nodes of 64 and of 128 bytes, and nodes of other kinds
    /// (procedural primitives: not decoded).
    pub boxes: [u32; 2],
    pub others: u32,
    /// Whether the triangle nodes are one run of nodes.
    pub one_run: bool,
    /// The lowest triangle node's number and how many triangle nodes there
    /// are. (A pass on the card could write a host bottom level's vertices
    /// without a walk from the root if they are the header's elements: one
    /// run from its elements offset. The dump test counts for how many.)
    pub leaves: (u32, u32),
}

/// An instance of a top level.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Instance {
    /// Object to world, three rows of four: a host instance's transform.
    pub matrix: [f32; 12],
    pub id: u32,
    pub mask: u8,
    pub contribution: u32,
    pub flags: u8,
    /// The bottom level's address (0 with no root).
    pub bottom: u64,
    /// The bottom level's root pointer, or [`NO_ROOT`].
    pub root: u32,
    /// The address of the instance's node.
    pub node: u64,
    /// The instance's pointer in its structure (kind 6).
    pub pointer: u32,
    /// World to object as the title's programs apply it: the move, and the
    /// rows (none: the identity's).
    pub to_object: ([f32; 3], Option<[f32; 9]>),
}

/// A top level decoded.
#[derive(Debug, Clone, PartialEq)]
pub struct Top {
    pub header: Header,
    /// The instances, in the order the walk from the root finds them.
    pub instances: Vec<Instance>,
    pub boxes: [u32; 2],
    pub others: u32,
}

fn node(mem: &impl Memory, base: u64, pointer: u32, words: &mut [u32]) -> Result<(), Error> {
    let at = base + u64::from(pointer >> 3) * 64;
    if mem.read(at, words) { Ok(()) } else { Err(Error::Unmapped(at)) }
}

/// Every node reachable from the root of the structure `h`: `leaf` for each
/// pointer that is not a box's, in the order found.
fn reach(mem: &impl Memory, h: &Header, mut leaf: impl FnMut(u32) -> Result<(), Error>) -> Result<[u32; 2], Error> {
    let mut boxes = [0u32; 2];
    let mut seen = HashSet::new();
    let mut stack = vec![h.root & NO_ROOT];
    while let Some(p) = stack.pop() {
        if p == NO_ROOT || !seen.insert(p) {
            continue;
        }
        if seen.len() as u64 > 4 * u64::from(h.nodes) + 4 {
            return Err(Error::Loop(h.address));
        }
        match p & 7 {
            kind @ (4 | 5) => {
                let mut w = [0u32; 4];
                node(mem, h.address, p, &mut w)?;
                boxes[(kind - 4) as usize] += 1;
                // In child order: the last pushed is walked first.
                stack.extend(w.iter().rev().filter(|&&c| c != NONE).map(|&c| c & NO_ROOT));
            }
            _ => leaf(p)?,
        }
    }
    Ok(boxes)
}

/// The bottom level at `address`: its triangles.
pub fn bottom(mem: &impl Memory, address: u64) -> Result<Bottom, Error> {
    let h = header(mem, address)?;
    if !h.bottom() {
        return Err(Error::WrongLevel(address));
    }
    let mut triangles = Vec::new();
    let mut others = 0;
    let mut nodes = Vec::new();
    let mut done = HashSet::new();
    let boxes = reach(mem, &h, |p| {
        if p & 7 > 3 {
            others += 1;
            return Ok(());
        }
        let mut w = [0u32; 28];
        node(mem, address, p, &mut w[..16])?;
        // The walk tests the triangle the pointer names and, from triangle
        // 0, the node's second one where the node says there is one.
        let pair = w[12] & 0x80 != 0;
        let slots = if p & 7 == 0 && pair { 0..2 } else { p & 7..(p & 7) + 1 };
        for slot in slots {
            let pointer = p & !7 | slot;
            if !done.insert(pointer) {
                continue;
            }
            triangles.push(Triangle {
                vertices: bvh::triangle_vertices(&w, slot).map(|v| h.to_object(v)),
                node: address + u64::from(p >> 3) * 64,
                pointer,
                opaque: w[12] & 1 != 0,
                geometry: w[12] >> 8,
                primitive: w[(13 + slot.min(1)) as usize],
            });
        }
        nodes.push(p >> 3);
        Ok(())
    })?;
    nodes.sort_unstable();
    nodes.dedup();
    let one_run = match (nodes.first(), nodes.last()) {
        (Some(&first), Some(&last)) => (last - first) as usize + 1 == nodes.len(),
        _ => true,
    };
    let leaves = (nodes.first().copied().unwrap_or(0), nodes.len() as u32);
    Ok(Bottom { header: h, triangles, boxes, others, one_run, leaves })
}

/// The top level at `address`: its instances.
pub fn top(mem: &impl Memory, address: u64) -> Result<Top, Error> {
    let h = header(mem, address)?;
    if h.bottom() {
        return Err(Error::WrongLevel(address));
    }
    let mut instances = Vec::new();
    let mut others = 0;
    let boxes = reach(mem, &h, |p| {
        if p & 7 != 6 {
            others += 1;
            return Ok(());
        }
        let mut w = [0u32; 28];
        node(mem, address, p, &mut w)?;
        let f = |k: usize| f32::from_bits(w[k]);
        let root = w[14] & NO_ROOT;
        instances.push(Instance {
            matrix: std::array::from_fn(f),
            id: w[12] & 0xff_ffff,
            mask: (w[12] >> 24) as u8,
            contribution: w[13] & 0xff_ffff,
            flags: (w[13] >> 24) as u8,
            bottom: if root == NO_ROOT { 0 } else { u64::from(w[15]) << 8 },
            root,
            node: address + u64::from(p >> 3) * 64,
            pointer: p,
            to_object: ([f(16), f(17), f(18)], (w[14] >> 31 == 0).then(|| std::array::from_fn(|k| f(19 + k)))),
        });
        Ok(())
    })?;
    Ok(Top { header: h, instances, boxes, others })
}

/// What a walking program asks of a trace, beside the ray.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rules {
    /// An instance is entered when its mask has one of these bits.
    pub mask: u8,
    /// 0: both faces count. -1: front faces are left out, +1: back faces
    /// (as the ray's flags say; an instance's flags turn it round or off).
    pub cull: i32,
    /// The walk ends at the first triangle it accepts (shadow rays).
    pub first_hit: bool,
}

/// The node tests of one walk, by the kind of the node.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Turns {
    pub boxes16: u32,
    pub boxes32: u32,
    pub triangles: u32,
    pub instances: u32,
    /// The most entries the walk's stack held (the title's has 16).
    pub deepest: u32,
}

impl Turns {
    /// The turns of the title's loop: it tests one node a turn.
    pub fn total(&self) -> u32 {
        self.boxes16 + self.boxes32 + self.triangles + self.instances
    }
}

/// The triangle a walk ended with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub t: f32,
    /// The barycentric weights of the triangle's second and third vertex.
    pub i: f32,
    pub j: f32,
    pub front: bool,
    /// The instance's pointer in the top level.
    pub instance: u32,
    /// The bottom level's address and the triangle's pointer in it.
    pub bottom: u64,
    pub triangle: u32,
}

/// One node test of a walk: the structure's address, the node's pointer in
/// it, and the ray as the walk has it there (an instance's, inside one).
pub trait NodeTest {
    fn test(&mut self, base: u64, pointer: u32, ray: &Ray) -> Result<[u32; 4], Error>;
}

/// The node test as the console does it ([`bvh::intersect`]).
pub struct Software<'a, M: Memory>(pub &'a M);

impl<M: Memory> NodeTest for Software<'_, M> {
    fn test(&mut self, base: u64, pointer: u32, ray: &Ray) -> Result<[u32; 4], Error> {
        let mut w = [0u32; 28];
        let n = match pointer & 7 {
            5 => 28,
            _ => 16,
        };
        node(self.0, base, pointer, &mut w[..n])?;
        Ok(bvh::intersect(&w, pointer & 7, ray, true))
    }
}

/// The walk of a title program over the top level at `top`, with the
/// console's node test.
pub fn walk(mem: &impl Memory, top: u64, ray: &Ray, rules: Rules) -> Result<(Option<Hit>, Turns), Error> {
    walk_with(mem, &mut Software(mem), top, ray, rules)
}

/// The walk of a title program over the top level at `top`: its loop, turn
/// by turn, as f607780aa8e502d0 has it (the other programs of the 64-bit
/// form run the same template with more code per accepted triangle).
///
/// A turn tests one node. A box node's children that were hit are taken
/// triangles first, then boxes and instances, each group nearest first; the
/// first is the next node, the others wait on a stack. An instance whose
/// mask passes moves the ray into its space and goes to its bottom level's
/// root; the pop that takes a waiting node of the top level again brings
/// the world's ray back. A triangle is accepted when its distance is not
/// negative, under the ray's extent, and its face is not culled; then the
/// extent is its distance.
///
/// Not modelled: the title's stack is a ring of 16 entries a lane and goes
/// on by parent pointers when it overflows ([`Turns::deepest`] says whether
/// a walk came there), and its cap on turns.
pub fn walk_with(
    mem: &impl Memory,
    test: &mut impl NodeTest,
    top: u64,
    ray: &Ray,
    rules: Rules,
) -> Result<(Option<Hit>, Turns), Error> {
    let h = header(mem, top)?;
    let world = (ray.origin, ray.direction);
    let (mut origin, mut direction) = world;
    let mut extent = ray.extent;
    let mut base = top;
    let mut next = h.root as i32;
    let mut pending = [NONE; 3];
    let mut stack: Vec<u32> = Vec::new();
    // The stack's depth when the instance was entered.
    let mut entered: Option<usize> = None;
    let (mut instance, mut cull) = (0u32, 0i32);
    let (mut hit, mut turns) = (None, Turns::default());
    loop {
        if next < 1 {
            let depth = stack.len();
            let Some(p) = stack.pop() else { break };
            if entered == Some(depth) {
                (base, origin, direction, entered, instance) = (top, world.0, world.1, None, 0);
            }
            next = p as i32;
        }
        let p = next as u32;
        let kind = p & 7;
        let here = Ray { extent, origin, direction, inverse: direction.map(|d| 1.0 / d) };
        let r = test.test(base, p, &here)?;
        match kind {
            4 => turns.boxes16 += 1,
            5 => turns.boxes32 += 1,
            6 | 7 => turns.instances += 1,
            _ => turns.triangles += 1,
        }
        // What the node keeps past the instruction's bytes.
        let mut w = [0u32; 4];
        if kind & 6 != 4 {
            let at = base + u64::from(p >> 3) * 64 + 48;
            if !mem.read(at, &mut w) {
                return Err(Error::Unmapped(at));
            }
            if kind == 6 {
                (origin, direction, instance) = (world.0, world.1, p);
            }
        }
        // The children of the box before that wait: farthest first.
        pending.sort_unstable();
        for &c in pending.iter().rev().filter(|&&c| c != NONE) {
            stack.push(c & 0x1fff_ffff);
        }
        turns.deepest = turns.deepest.max(stack.len() as u32);
        pending = [NONE; 3];
        match kind {
            6 if (w[0] >> 24) as u8 & rules.mask != 0 => {
                entered = Some(stack.len());
                let at = top + u64::from(p >> 3) * 64 + 64;
                let mut m = [0u32; 12];
                if !mem.read(at, &mut m) {
                    return Err(Error::Unmapped(at));
                }
                let m = m.map(f32::from_bits);
                if w[2] >> 31 == 0 {
                    // Rows at dwords 3-11, as the title's code sums them.
                    let turn = |v: [f32; 3]| -> [f32; 3] {
                        std::array::from_fn(|a| m[3 + 3 * a] * v[0] + m[4 + 3 * a] * v[1] + m[5 + 3 * a] * v[2])
                    };
                    (origin, direction) = (turn(world.0), turn(world.1));
                }
                origin = std::array::from_fn(|a| origin[a] + m[a]);
                let flags = (w[1] >> 24) as u8;
                cull = rules.cull
                    * match (flags & FLAG_CULL_DISABLE != 0, flags & FLAG_FRONT_CCW != 0) {
                        (true, _) => 0,
                        (false, false) => 1,
                        (false, true) => -1,
                    };
                let root = w[2] & NO_ROOT;
                next = if root == NO_ROOT { -1 } else { root as i32 };
                base = u64::from(w[3]) << 8;
            }
            6 | 7 => next = -1,
            0..=3 => {
                next = if kind == 0 && w[0] & 0x80 != 0 { (p | 1) as i32 } else { -1 };
                let [numerator, denominator, i, j] = r.map(f32::from_bits);
                let scale = 1.0 / denominator;
                let t = scale * numerator;
                // The sign of the denominator's upper half: the face.
                let face = i32::from((denominator.to_bits() >> 16) as i16);
                if !(t < 0.0 || face.signum() * cull < 0 || t >= extent) {
                    extent = t;
                    hit = Some(Hit {
                        t,
                        i: scale * i,
                        j: scale * j,
                        front: denominator > 0.0,
                        instance,
                        bottom: base,
                        triangle: p,
                    });
                    if rules.first_hit {
                        break;
                    }
                }
            }
            _ => {
                // Triangles before boxes and instances, each in the order
                // the instruction gave them.
                let mut keys: [u32; 4] = std::array::from_fn(|n| match r[n] {
                    NONE => NONE,
                    c => c & 0x1fff_ffff | (c & 4) << 29 | (n as u32) << 29,
                });
                keys.sort_unstable();
                next = if keys[0] == NONE { -1 } else { (keys[0] & 0x1fff_ffff) as i32 };
                pending = [keys[1], keys[2], keys[3]];
            }
        }
    }
    Ok((hit, turns))
}

/// The guide table's dwords before its lists: [`GUIDE_TOPS`] entries of
/// four from dword 4.
pub const GUIDE_HEADER: u32 = 4 + 4 * GUIDE_TOPS;
/// The top levels one table guides.
pub const GUIDE_TOPS: u32 = 4;
/// A guided top level's entry that guides nothing: no node is there.
pub const GUIDE_NO_TOP: [u32; 4] = [!0, !0, 0, 0];

/// What the guided node test knows a node by: its address in 64-byte units
/// (the low 32 bits: addresses below 256 GiB).
pub fn guide_key(node_address: u64) -> u32 {
    (node_address >> 6) as u32
}

/// A top level the guided node test takes to the host's hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GuidedTop {
    /// The address of the title's top level and its root's pointer.
    pub address: u64,
    pub root: u32,
    /// The view-heap slot of the host's top level built from it.
    pub slot: u32,
}

/// The buffer the guided node test reads ([`crate::emit::SLOT_GUIDE`]), for
/// these top levels and the bottom levels the host built structures of.
/// Returns the buffer's dwords and, per bottom level, where its list
/// starts: what its host instances carry as their instance id.
///
/// - dword 0: the number of top levels guided (the shader does not read it);
/// - dwords 4 + 4k .. 8 + 4k, k below [`GUIDE_TOPS`]: the address of top
///   level k's root node (low dword, bits from 32 up), the root's pointer in
///   its structure, the heap slot of the host's top level ([`GUIDE_NO_TOP`]
///   for none);
/// - a list per bottom level: [`guide_key`] of its root node, then per host
///   primitive the pointer of the title's triangle ([`Triangle::pointer`]).
///
/// A host instance of a guided bottom level carries the list's first dword
/// as its instance id and the number of the title's instance node (its
/// pointer without the kind) as its contribution to the hit group index;
/// both are 24 bits ([`host_instance`]).
pub fn guide_table(tops: &[GuidedTop], bottoms: &[&Bottom]) -> (Vec<u32>, Vec<u32>) {
    assert!(tops.len() <= GUIDE_TOPS as usize, "more top levels than the table guides");
    let mut words = vec![0u32; GUIDE_HEADER as usize];
    words[0] = tops.len() as u32;
    for k in 0..GUIDE_TOPS as usize {
        let entry = match tops.get(k) {
            Some(t) => {
                let root = t.address + u64::from(t.root >> 3) * 64;
                [root as u32, (root >> 32) as u32, t.root, t.slot]
            }
            None => GUIDE_NO_TOP,
        };
        words[4 + 4 * k..8 + 4 * k].copy_from_slice(&entry);
    }
    let mut starts = Vec::with_capacity(bottoms.len());
    for b in bottoms {
        starts.push(words.len() as u32);
        words.push(guide_key(b.header.address + u64::from(b.header.root >> 3) * 64));
        words.extend(b.triangles.iter().map(|t| t.pointer));
    }
    (words, starts)
}

/// The two 24-bit numbers the host instance of the title's instance `i`
/// carries for the guided node test: its instance id (`list`, where its
/// bottom level's list starts in the guide table) and its contribution to
/// the hit group index. None when one does not fit: such an instance cannot
/// be guided.
pub fn host_instance(i: &Instance, list: u32) -> Option<(u32, u32)> {
    let number = i.pointer >> 3;
    (list < 1 << 24 && number < 1 << 24).then_some((list, number))
}

/// What the host has of the bottom level a title instance names.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Hosted {
    /// A structure of its triangles at this address on the card, and where
    /// its list starts in the guide table.
    Triangles { list: u32, structure: u64 },
    /// No structure (one the title rebuilds every frame; a static one not
    /// built yet): the host's one box primitive ([`STAND_IN`], the structure
    /// at this address) takes the bottom level's bounds. The hardware then
    /// says "the ray crosses this instance" and the title walks the bottom
    /// level itself, as without the guide.
    Bounds { bounds: [f32; 6], structure: u64 },
}

/// The box primitive every stand-in instance is an instance of: the unit
/// cube.
pub const STAND_IN: [f32; 6] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

/// A bottom level's bounds as its stand-in takes them: a little larger
/// (a thousandth of each side, and of the coordinates' size), so that
/// neither a flat structure nor a triangle on a face falls out by rounding.
/// None for bounds that are none (not numbers, inside out).
pub fn stand_in_bounds(bounds: &[f32; 6]) -> Option<[f32; 6]> {
    let mut grown = [0f32; 6];
    for a in 0..3 {
        let (low, high) = (bounds[a], bounds[3 + a]);
        if !(low.is_finite() && high.is_finite() && low <= high) {
            return None;
        }
        let pad = 1e-3 * (high - low) + 1e-4 * low.abs().max(high.abs()) + 1e-6;
        (grown[a], grown[3 + a]) = (low - pad, high + pad);
    }
    Some(grown)
}

/// The title's object-to-world matrix `m` with the unit cube moved onto
/// `bounds` first ([`stand_in_bounds`] of a bottom level's): the matrix of
/// a stand-in instance.
pub fn stand_in_matrix(m: &[f32; 12], bounds: &[f32; 6]) -> Option<[f32; 12]> {
    let b = stand_in_bounds(bounds)?;
    let mut out = [0f32; 12];
    for r in 0..3 {
        for c in 0..3 {
            out[4 * r + c] = m[4 * r + c] * (b[3 + c] - b[c]);
        }
        // (In this order of sums: the pass on the card does the same.)
        out[4 * r + 3] = m[4 * r + 3] + m[4 * r] * b[0] + m[4 * r + 1] * b[1] + m[4 * r + 2] * b[2];
    }
    out.iter().all(|x| x.is_finite()).then_some(out)
}

/// The host's instance description (16 dwords, as the host's top-level
/// build reads them) of the title's instance `i`: the title's matrix, mask
/// and flags as they are; the two numbers the guided node test reads
/// ([`host_instance`]); the host's structure. None for an instance that
/// cannot be guided (it is then left out: [`no_instance_words`]).
///
/// The on-card pass of the layer ([`TOP_PASS`]) writes the same from the
/// title's instance nodes as the card holds them.
pub fn host_instance_words(i: &Instance, hosted: Hosted) -> Option<[u32; 16]> {
    // (The driver's build is given no number that is not finite.)
    if !i.matrix.iter().all(|m| m.is_finite()) {
        return None;
    }
    let (matrix, list, structure) = match hosted {
        Hosted::Triangles { list, structure } => (i.matrix, list, structure),
        Hosted::Bounds { bounds, structure } => (stand_in_matrix(&i.matrix, &bounds)?, 0, structure),
    };
    let (id, number) = host_instance(i, list)?;
    let mut w = [0u32; 16];
    for k in 0..12 {
        w[k] = matrix[k].to_bits();
    }
    (w[12], w[13]) = (id | u32::from(i.mask) << 24, number | u32::from(i.flags & HOST_FLAGS) << 24);
    (w[14], w[15]) = (structure as u32, (structure >> 32) as u32);
    Some(w)
}

/// The description of no instance, for a place of a host top level's
/// descriptions that has none (past the title's instances, an instance of
/// nothing, one whose memory is not there): the stand-in box where it is,
/// with a mask no ray has a bit of. (Not a description of zeros: a host
/// instance of no structure is allowed, but the software adapter's build
/// does not survive one. MEASURED, 2026-11-27: an access violation.)
pub fn no_instance_words(stand_in: u64) -> [u32; 16] {
    let mut w = [0u32; 16];
    (w[0], w[5], w[10]) = (1f32.to_bits(), 1f32.to_bits(), 1f32.to_bits());
    (w[14], w[15]) = (stand_in as u32, (stand_in >> 32) as u32);
    w
}

/// The host table's dwords before its places.
pub const HOSTS_HEADER: u32 = 4;
/// The places a key is looked for in, from its own on.
pub const HOSTS_PROBES: u32 = 16;

/// What the host table knows a bottom level by: its address in 256-byte
/// units, as the title's instance nodes name it (dword 15).
pub fn hosts_key(bottom: u64) -> u32 {
    (bottom >> 8) as u32
}

/// A place of the host table whose bottom level the host let go: no key,
/// and not the end of a run of places either.
pub const HOSTS_GONE: u32 = !0;

/// The place a key is looked for first, of `places` (a power of two).
pub fn hosts_place(key: u32, places: u32) -> u32 {
    (key.wrapping_mul(0x9E37_79B1) >> 16) & (places - 1)
}

/// The host table: which of the title's bottom levels the host has a
/// structure of, for the pass that writes a host top level's instances on
/// the card ([`TOP_PASS`]). `built`: a bottom level's address, where its
/// list starts in the guide table, its host structure's address on the card.
///
/// - dword 0: the number of places (a power of two);
/// - four dwords a place from dword [`HOSTS_HEADER`]: the key
///   ([`hosts_key`]; 0: free), the list's start, the structure's address
///   (low, high). A key lies within [`HOSTS_PROBES`] places of its own.
///
/// None when `places` are too few for that.
pub fn hosts_table(built: &[(u64, u32, u64)], places: u32) -> Option<Vec<u32>> {
    assert!(places.is_power_of_two(), "{places} places");
    let mut words = vec![0u32; (HOSTS_HEADER + 4 * places) as usize];
    words[0] = places;
    'next: for &(bottom, list, structure) in built {
        let key = hosts_key(bottom);
        if key == 0 {
            return None;
        }
        for probe in 0..HOSTS_PROBES {
            let at = (HOSTS_HEADER + 4 * ((hosts_place(key, places) + probe) & (places - 1))) as usize;
            if words[at] == 0 || words[at] == key {
                words[at..at + 4].copy_from_slice(&[key, list, structure as u32, (structure >> 32) as u32]);
                continue 'next;
            }
        }
        return None;
    }
    Some(words)
}

/// What the host table has of the bottom level at `bottom`: its list's
/// start and its structure's address. As [`TOP_PASS`] looks it up.
pub fn hosts_find(table: &[u32], bottom: u64) -> Option<(u32, u64)> {
    let (key, places) = (hosts_key(bottom), table[0]);
    for probe in 0..HOSTS_PROBES {
        let at = (HOSTS_HEADER + 4 * ((hosts_place(key, places) + probe) & (places - 1))) as usize;
        match table[at] {
            0 => return None,
            k if k == key => return Some((table[at + 1], u64::from(table[at + 3]) << 32 | u64::from(table[at + 2]))),
            _ => {}
        }
    }
    None
}

/// The instance descriptions of a host top level, as the layer's pass on
/// the card ([`TOP_PASS`]) writes them for the title's top level `top`:
/// one of 16 dwords per instance node, in the nodes' order, `most` in all
/// ([`no_instance_words`] where there is none). The same on the CPU,
/// for the tests; `bounds`: a bottom level's bounds as its header has them,
/// or none where there is no header to read.
pub fn top_pass_words(
    top: &Top,
    most: u32,
    hosts: &[u32],
    stand_in: u64,
    bounds: impl Fn(u64) -> Option<[f32; 6]>,
) -> Vec<u32> {
    let mut words: Vec<u32> = (0..most).flat_map(|_| no_instance_words(stand_in)).collect();
    for i in &top.instances {
        // The instance's place among the instance nodes: two nodes each
        // from the header's elements offset.
        let Some(k) = (i.node - top.header.address).checked_sub(u64::from(top.header.elements_at)) else { continue };
        if k % 128 != 0 || k / 128 >= u64::from(most).min(u64::from(top.header.elements)) || i.bottom == 0 {
            continue;
        }
        let hosted = match hosts_find(hosts, i.bottom) {
            Some((list, structure)) => Some(Hosted::Triangles { list, structure }),
            None => bounds(i.bottom).map(|bounds| Hosted::Bounds { bounds, structure: stand_in }),
        };
        if let Some(w) = hosted.and_then(|h| host_instance_words(i, h)) {
            let at = 16 * (k / 128) as usize;
            words[at..at + 16].copy_from_slice(&w);
        }
    }
    words
}

/// The layer's pass that writes a host top level's instance descriptions
/// ON THE CARD, from the title's instance nodes as the card holds them (the
/// title's builder leaves a top level on the card alone; it is built anew
/// every frame, so a copy back to the PC would be frames late). A compute
/// program of ours, one thread an instance node; the host's top-level build
/// then reads what it wrote. Plain HLSL of shader model 5.1, which the
/// system's compiler takes (as `victory-gpu` compiles its own passes).
///
/// It reads title memory the way translated programs do: the directory
/// (two dwords a 64 KiB page: the copy's number, 0 for none, and the page's
/// byte offset in the copy) and the copies' views (a table from the null
/// view of the run on). A page without a copy reads as "not there": the
/// instance is left out, as the title's own walk finds nothing there.
///
/// Bindings: `b0` the title top level's address (low dword, bits from 32
/// up), the most instances written, the stand-in structure's address (low,
/// high), the top level's entry in the guide table, the view-heap slot of
/// the host's top level, and where its counts go (a byte offset); `t0` the
/// directory; `t1` the host table ([`hosts_table`]); `u0` the descriptions
/// (64 bytes each, [`top_pass_words`]); `u1` four counts (dword 0: the
/// title's instances, or all ones for no header; then, added up over the
/// passes: with a host structure, as stand-in boxes, left out); `u2` the
/// guide table, whose entry of this top level the pass writes (the root
/// node's address, the root's pointer, the slot: only the card knows the
/// root; [`GUIDE_NO_TOP`] when there is no header to read or the title has
/// more instances than `most`: that top level is then walked as without
/// the guide); `u3` the notes of pages found without a copy, left as walking
/// programs leave theirs (a guided walk no longer touches every node, so the
/// pass asks for the pages of the instance nodes and of the bottom levels'
/// headers itself); `u0, space1`: the copies' 4,096 views (the run
/// `victory-gpu` `memory.rs` keeps in every view heap).
///
/// It takes the instance nodes to be one run of 128 bytes each from the
/// header's elements offset. That is how the title's walking programs
/// address them (READ in f22a76fcb967ee59; not yet checked on a dump).
pub const TOP_PASS: &str = r"
cbuffer Top : register(b0) { uint topLow; uint topHigh; uint most; uint boxLow; uint boxHigh; uint entry; uint slot; uint countsAt; };
ByteAddressBuffer directory : register(t0);
ByteAddressBuffer hosts : register(t1);
RWByteAddressBuffer descs : register(u0);
RWByteAddressBuffer counts : register(u1);
RWByteAddressBuffer guide : register(u2);
RWByteAddressBuffer faults : register(u3);
RWByteAddressBuffer copies[4096] : register(u0, space1);

static const uint PAGES = 1u << 21;

bool mem(uint low, uint high, uint offset, out uint value)
{
    uint a = low + offset;
    uint h = high + (a < low ? 1u : 0u);
    uint page = (h << 16) | (a >> 16);
    value = 0;
    if (h >= 0x20u || page >= PAGES) return false;
    uint2 e = directory.Load2(page * 8u);
    uint number = e.x & 0x7fffffffu;
    if (number == 0u)
    {
        // Its number at a place by the number, for the host to make the
        // page (title memory lies above 4 GiB).
        if (page >= 0x10000u) faults.Store(4u * (page % 4095u + 1u), page);
        return false;
    }
    value = copies[NonUniformResourceIndex(number)].Load(e.y + (a & 0xffffu));
    return true;
}

bool finite(float x) { return (asuint(x) & 0x7f800000u) != 0x7f800000u; }

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint k = id.x;
    if (k >= most) return;
    // No instance: the stand-in box where it is, with a mask of no bits.
    uint d[16];
    uint n;
    for (n = 0u; n < 16u; n++) d[n] = 0u;
    d[0] = 0x3f800000u;
    d[5] = 0x3f800000u;
    d[10] = 0x3f800000u;
    d[14] = boxLow;
    d[15] = boxHigh;
    uint head, elements, at;
    bool ok = mem(topLow, topHigh, 0u, head);
    ok = mem(topLow, topHigh, 12u, elements) && ok;
    ok = mem(topLow, topHigh, 52u, at) && ok;
    // Version 6, a top level, not packed.
    ok = ok && (head & 0xffffu) == 6u && (head & 0x81000000u) == 0u;
    if (k == 0u)
    {
        counts.Store(countsAt, ok ? elements : 0xffffffffu);
        // The top level's entry in the guide table: its root node, or no
        // top level where the walk must stay the title's own.
        uint root;
        bool rooted = mem(topLow, topHigh, 48u, root);
        uint4 e = uint4(0xffffffffu, 0xffffffffu, 0u, 0u);
        if (ok && rooted && elements <= most)
        {
            uint low = topLow + (root >> 3) * 64u;
            uint high = topHigh + (low < topLow ? 1u : 0u);
            e = uint4(low, high, root, slot);
        }
        if (entry < 4u) guide.Store4(16u + 16u * entry, e);
    }
    if (ok && k < elements)
    {
        uint node = at + 128u * k;
        uint w[16];
        bool there = true;
        for (n = 0u; n < 16u; n++)
        {
            uint v;
            there = mem(topLow, topHigh, node + 4u * n, v) && there;
            w[n] = v;
        }
        uint number = node >> 6;
        uint key = w[15];
        bool made = false;
        if (there && (w[14] & 0x3fffffffu) != 0x3fffffffu && number < (1u << 24) && key != 0u)
        {
            uint places = hosts.Load(0u);
            uint place = (key * 0x9E3779B1u) >> 16;
            uint4 e = uint4(0u, 0u, 0u, 0u);
            bool found = false;
            // (A table whose first words have not come yet is zeros: no
            // place of it is looked into, and every instance is a box.)
            bool open = places != 0u && places <= 0x10000u && (places & (places - 1u)) == 0u;
            for (uint probe = 0u; probe < 16u; probe++)
            {
                uint4 x = hosts.Load4(16u + 16u * ((place + probe) & (places - 1u)));
                if (open && x.x == key) { e = x; found = true; open = false; }
                if (open && x.x == 0u) open = false;
            }
            if (found)
            {
                // (No number that is not finite goes to the driver's build.)
                bool sane = true;
                for (n = 0u; n < 12u; n++) sane = sane && (w[n] & 0x7f800000u) != 0x7f800000u;
                if (e.y < (1u << 24) && sane)
                {
                    for (n = 0u; n < 12u; n++) d[n] = w[n];
                    d[12] = e.y | (w[12] & 0xff000000u);
                    d[13] = number | (w[13] & 0x0f000000u);
                    d[14] = e.z;
                    d[15] = e.w;
                    made = true;
                    counts.InterlockedAdd(countsAt + 4u, 1u);
                }
            }
            else
            {
                // No host structure: the stand-in box takes the bottom
                // level's bounds, a little larger.
                uint bottomLow = key << 8;
                uint bottomHigh = key >> 24;
                uint b[6];
                uint bhead;
                bool fine = mem(bottomLow, bottomHigh, 0u, bhead);
                for (n = 0u; n < 6u; n++)
                {
                    uint v;
                    fine = mem(bottomLow, bottomHigh, 16u + 4u * n, v) && fine;
                    b[n] = v;
                }
                // Version 6, a bottom level, not packed.
                fine = fine && (bhead & 0xffffu) == 6u && (bhead & 0x81000000u) == 0x80000000u;
                precise float low[3];
                precise float size[3];
                for (n = 0u; n < 3u; n++)
                {
                    precise float lo = asfloat(b[n]);
                    precise float hi = asfloat(b[3u + n]);
                    fine = fine && finite(lo) && finite(hi) && lo <= hi;
                    precise float side = hi - lo;
                    precise float pad = 1e-3f * side;
                    pad = pad + 1e-4f * max(abs(lo), abs(hi));
                    pad = pad + 1e-6f;
                    precise float grownLow = lo - pad;
                    precise float grownHigh = hi + pad;
                    low[n] = grownLow;
                    size[n] = grownHigh - grownLow;
                }
                precise float m[12];
                for (n = 0u; n < 12u; n++) m[n] = asfloat(w[n]);
                precise float o[12];
                for (n = 0u; n < 3u; n++)
                {
                    o[4u * n] = m[4u * n] * size[0];
                    o[4u * n + 1u] = m[4u * n + 1u] * size[1];
                    o[4u * n + 2u] = m[4u * n + 2u] * size[2];
                    precise float moved = m[4u * n] * low[0];
                    moved = m[4u * n + 3u] + moved;
                    precise float second = m[4u * n + 1u] * low[1];
                    moved = moved + second;
                    precise float third = m[4u * n + 2u] * low[2];
                    moved = moved + third;
                    o[4u * n + 3u] = moved;
                }
                for (n = 0u; n < 12u; n++) fine = fine && finite(o[n]);
                if (fine)
                {
                    for (n = 0u; n < 12u; n++) d[n] = asuint(o[n]);
                    d[12] = w[12] & 0xff000000u;
                    d[13] = number | (w[13] & 0x0f000000u);
                    d[14] = boxLow;
                    d[15] = boxHigh;
                    made = true;
                    counts.InterlockedAdd(countsAt + 8u, 1u);
                }
            }
        }
        if (!made) counts.InterlockedAdd(countsAt + 12u, 1u);
    }
    for (n = 0u; n < 4u; n++) descs.Store4(64u * k + 16u * n, uint4(d[4u * n], d[4u * n + 1u], d[4u * n + 2u], d[4u * n + 3u]));
}
";

/// A pointer of the guide table's lists that names no node: the title's
/// loop takes nothing for it.
pub const NO_POINTER: u32 = NONE;

/// The geometries of a bottom level [`BOTTOM_PASS`] takes.
pub const MOVING_GEOMETRIES: usize = 8;

/// A job of [`BOTTOM_PASS`], 16 dwords: a bottom level the title's builder
/// has just built or refitted on the card, of which the host builds its own.
///
/// - 0, 1: the title structure's address (low dword, bits from 32 up);
/// - 2: its triangles, as the title described them to its builder or as
///   [`census_job`] counted them (all its geometries together): the host
///   structure has that many primitives. 0: the structure's place in the
///   host table is marked gone and nothing else is done;
/// - 3: the byte of its first vertex in the pass's vertex buffer;
/// - 4: where its list starts in the guide table (a dword index);
/// - 5: the byte of its place in the host table;
/// - 6, 7: the host structure's address on the card;
/// - 8 + g, g below [`MOVING_GEOMETRIES`]: the host primitive of geometry g's
///   first triangle (all ones: no such geometry).
///
/// A triangle of the title's geometry g with primitive index p is the host's
/// primitive `base[g] + p`: the host structure's order is the order the
/// title described its triangles in, whatever tree the title's builder made
/// of them. So a refit of the title's is a refit of the host's, and so is a
/// rebuild that keeps the triangle count.
pub fn moving_job(address: u64, triangles: u32, vertices_at: u32, list: u32, hosts_at: u32, structure: u64, bases: [u32; MOVING_GEOMETRIES]) -> [u32; 16] {
    let mut w = [0u32; 16];
    w[..8].copy_from_slice(&[address as u32, (address >> 32) as u32, triangles, vertices_at, list, hosts_at, structure as u32, (structure >> 32) as u32]);
    w[8..].copy_from_slice(&bases);
    w
}

/// A job of [`BOTTOM_PASS`] that only counts: the structure at `address`,
/// which the title's builder wrote and nobody described. At most `nodes`
/// triangle nodes are looked at; the counts go to the 12 dwords at byte
/// `slot` of the pass's counts ([`CENSUS_DWORDS`]):
///
/// - 0: the pass's stamp, written last (the CPU zeroed the slot before);
/// - 1: the header's triangle nodes, or all ones where there is no bottom
///   level of the format at the address (a top level, scratch memory);
/// - 2 + g: one more than the highest primitive index of geometry g;
/// - 10: not 0 where a node's page had no copy (counted again later);
/// - 11: not 0 where a node names a geometry past [`MOVING_GEOMETRIES`].
pub fn census_job(address: u64, nodes: u32, slot: u32) -> [u32; 16] {
    let mut w = [0u32; 16];
    w[..6].copy_from_slice(&[address as u32, (address >> 32) as u32, nodes, !0, 0, slot]);
    w
}

/// The dwords of a count ([`census_job`]).
pub const CENSUS_DWORDS: usize = 12;
/// [`BOTTOM_PASS`]'s counts from its `countsAt` on, in bytes: structures
/// taken (0) and turned down (4) since the start; the keys of 16 structures
/// last turned down (8); and, from [`MOVING_FLAGS`], a dword a row of the
/// dispatch, [`MOVING_ROWS`] rows at most: the stamp of the pass in which a
/// triangle of that row did not fit.
pub const MOVING_FLAGS: u32 = 128;
pub const MOVING_ROWS: u32 = 1024;

/// The layer's pass for bottom levels the title builds at run time
/// (characters, vehicles: built or refitted by the title's own builder on
/// the card every frame they change). One row of threads a job
/// ([`moving_job`], [`census_job`]), one thread a triangle node. Dispatched
/// three times, `phase` 0, 1 and 2, with a barrier between:
///
/// - **0, clear**: every pointer of the structure's list in the guide table
///   becomes [`NO_POINTER`]. A primitive the title's structure does not hold
///   in this frame (its builder leaves out what has no area) must not hand
///   the title's loop the node that held it a frame ago.
/// - **1, fill**: it reads the title's structure as it is after the title's
///   own build and writes, per triangle, its three vertices into the vertex
///   buffer at the host primitive the title's own primitive index names (so
///   the host's build needs nothing of the title's vertex and index buffers,
///   formats or skinning: the title's builder has digested them), and its
///   node's pointer into the structure's list, at the same primitive: what
///   the guided node test hands the title's loop. A triangle that has no
///   primitive of the host's (a geometry past the eighth, an index past its
///   geometry's count) or a vertex that is not finite leaves the pass's
///   stamp in the row's flag: the hardware would never offer it.
/// - **2, commit** (the row's first thread alone): the list's key (the root
///   node) and the structure's place in the host table: its key and host
///   structure when the title's structure is a bottom level whose triangle
///   nodes are no more than the job's triangles and every triangle found its
///   primitive in this pass, else "gone": its instances are stand-in boxes
///   in this list, the title walks them itself, and the structure's key is
///   left for the CPU among the last turned down.
///
/// So a host structure is offered to the top level's pass in a list only
/// when this pass has written it in that same list from the title's
/// structure as it then is.
///
/// Bindings: `b0` the number of jobs, where the counts go, the phase and
/// the pass's stamp (not 0); `t0` the directory; `t1` the jobs; `u0` the
/// vertex buffer; `u1` the host table; `u2` the guide table; `u3` the notes
/// of pages without a copy; `u4` the counts ([`MOVING_FLAGS`],
/// [`census_job`]); `u0, space1` the copies' 4,096 views.
///
/// It takes the triangle nodes to be one run of 64 bytes each from the
/// header's elements offset, as many as the header's elements (as
/// [`TOP_PASS`] takes the instance nodes; not yet checked on a dump).
pub const BOTTOM_PASS: &str = r"
cbuffer Jobs : register(b0) { uint jobs; uint countsAt; uint phase; uint stamp; };
ByteAddressBuffer directory : register(t0);
ByteAddressBuffer table : register(t1);
RWByteAddressBuffer vertices : register(u0);
RWByteAddressBuffer hosts : register(u1);
RWByteAddressBuffer guide : register(u2);
RWByteAddressBuffer faults : register(u3);
RWByteAddressBuffer counts : register(u4);
RWByteAddressBuffer copies[4096] : register(u0, space1);

static const uint PAGES = 1u << 21;

bool mem(uint low, uint high, uint offset, out uint value)
{
    uint a = low + offset;
    uint h = high + (a < low ? 1u : 0u);
    uint page = (h << 16) | (a >> 16);
    value = 0;
    if (h >= 0x20u || page >= PAGES) return false;
    uint2 e = directory.Load2(page * 8u);
    uint number = e.x & 0x7fffffffu;
    if (number == 0u)
    {
        if (page >= 0x10000u) faults.Store(4u * (page % 4095u + 1u), page);
        return false;
    }
    value = copies[NonUniformResourceIndex(number)].Load(e.y + (a & 0xffffu));
    return true;
}

bool finite(uint bits) { return (bits & 0x7f800000u) != 0x7f800000u; }

[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint k = id.x;
    uint row = id.y;
    if (row >= jobs || (phase == 2u && k != 0u)) return;
    uint4 j0 = table.Load4(64u * row);
    uint4 j1 = table.Load4(64u * row + 16u);
    uint low = j0.x;
    uint high = j0.y;
    uint triangles = j0.z;
    bool census = j0.w == 0xffffffffu;
    if (phase != 2u && k >= triangles) return;
    if (phase == 0u)
    {
        if (!census) guide.Store(4u * (j1.x + 1u + k), 0xffffffffu);
        return;
    }
    uint flagAt = countsAt + 128u + 4u * (row & 1023u);
    uint head, elements, at, root, own;
    bool ok = mem(low, high, 0u, head);
    ok = mem(low, high, 12u, elements) && ok;
    ok = mem(low, high, 48u, root) && ok;
    ok = mem(low, high, 52u, at) && ok;
    ok = mem(low, high, 60u, own) && ok;
    // Version 6, a bottom level, not packed, and its vertices in object
    // space (one scaled, moved or turned stays a box: the pass writes the
    // vertices as the nodes hold them).
    ok = ok && (head & 0xffffu) == 6u && (head & 0x81000000u) == 0x80000000u && (own & 0xff000000u) == 0u;
    uint n;
    if (phase == 2u)
    {
        if (census)
        {
            counts.Store(j1.y + 4u, ok ? elements : 0xffffffffu);
            counts.Store(j1.y, stamp);
            return;
        }
        // No more triangle nodes than the structure has triangles, and no
        // triangle of this pass without its primitive.
        bool fit = ok && triangles != 0u && elements <= triangles && counts.Load(flagAt) != stamp;
        uint key = (high << 24) | (low >> 8);
        if (triangles != 0u)
        {
            // (The root's key: its address in 64-byte units.)
            guide.Store(4u * j1.x, ((high << 26) | (low >> 6)) + (root >> 3));
            counts.InterlockedAdd(countsAt + (fit ? 0u : 4u), 1u);
            if (!fit) counts.Store(countsAt + 8u + 4u * (row & 15u), key);
        }
        hosts.Store4(j1.y, fit ? uint4(key, j1.x, j1.z, j1.w) : uint4(0xffffffffu, 0u, 0u, 0u));
        return;
    }
    if (!ok || k >= elements) return;
    uint node = at + 64u * k;
    uint w[16];
    bool there = true;
    for (n = 0u; n < 16u; n++)
    {
        uint v;
        there = mem(low, high, node + 4u * n, v) && there;
        w[n] = v;
    }
    uint geometry = w[12] >> 8;
    // The node's first triangle, and its second where it has one.
    uint many = (w[12] & 0x80u) != 0u ? 2u : 1u;
    uint slot;
    if (census)
    {
        if (!there) { counts.Store(j1.y + 40u, 1u); return; }
        if (geometry >= 8u) { counts.Store(j1.y + 44u, 1u); return; }
        for (slot = 0u; slot < many; slot++) counts.InterlockedMax(j1.y + 8u + 4u * geometry, w[13u + slot] + 1u);
        return;
    }
    // (A node whose page has no copy: the title's own test of it reads
    // zeros and finds nothing, with the guide and without.)
    if (!there) return;
    if (geometry >= 8u) { counts.Store(flagAt, stamp); return; }
    uint first = table.Load(64u * row + 32u + 4u * geometry);
    // The geometry's primitives end where the next geometry's begin.
    uint limit = triangles;
    for (n = 0u; n < 8u; n++)
    {
        uint other = table.Load(64u * row + 32u + 4u * n);
        if (other != 0xffffffffu && other > first && other < limit) limit = other;
    }
    for (slot = 0u; slot < many; slot++)
    {
        uint primitive = w[13u + slot];
        if (first >= limit || primitive >= limit - first) { counts.Store(flagAt, stamp); continue; }
        uint s = first + primitive;
        // Its vertices in the order the node test takes them: the one
        // neither weight belongs to, then I's, then J's (the console's
        // vertex order bytes; triangle 0 is the node's vertices 0, 1, 2 and
        // triangle 1 its vertices 1, 3, 2).
        uint order = w[15] >> (8u * slot);
        uint i = order & 3u;
        uint jj = (order >> 2) & 3u;
        uint others = (3u - i - jj) & 3u;
        uint map = slot == 0u ? 0x210u : 0x231u;
        uint picks[3] = { others, i, jj };
        uint3 p[3];
        bool sane = true;
        for (n = 0u; n < 3u; n++)
        {
            uint vertex = (map >> (4u * picks[n])) & 0xfu;
            p[n] = uint3(w[3u * vertex], w[3u * vertex + 1u], w[3u * vertex + 2u]);
            sane = sane && finite(p[n].x) && finite(p[n].y) && finite(p[n].z);
        }
        // (The driver's build and refit are given no number that is not
        // finite: such a triangle is the title's own to find.)
        if (!sane) { counts.Store(flagAt, stamp); continue; }
        for (n = 0u; n < 3u; n++) vertices.Store3(j0.w + 36u * s + 12u * n, p[n]);
        guide.Store(4u * (j1.x + 1u + s), ((node >> 6) << 3) | slot);
    }
}
";

/// A candidate the host's hardware found for a ray.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Candidate {
    /// The host instance's id and contribution ([`host_instance`]).
    pub id: u32,
    pub contribution: u32,
    /// The primitive's index in the host's bottom level.
    pub primitive: u32,
    pub t: f32,
    /// Not a triangle: the box that stands in for a bottom level the host
    /// has no structure of ([`Hosted::Bounds`]). The title walks that
    /// bottom level itself, by its own boxes.
    pub boxed: bool,
}

/// The host's inline ray query, as the guided node test uses it: every
/// triangle the ray meets before its extent comes as a candidate, in an
/// order of the hardware's choosing, until one is committed; then only
/// nearer ones.
pub trait Query {
    /// A new ray against the host top level at heap slot `slot`.
    fn trace(&mut self, slot: u32, ray: &Ray);
    /// The next candidate.
    fn proceed(&mut self) -> Option<Candidate>;
    /// The candidate [`Query::proceed`] gave last is a hit: nothing farther
    /// comes any more.
    fn commit(&mut self);
    /// Inside the box [`Query::proceed`] gave last the title found a hit at
    /// `t`: nothing farther comes any more.
    fn commit_box(&mut self, t: f32);
    /// The distance of what was committed last (the ray's extent: nothing).
    fn committed(&self) -> f32;
}

/// The guided node test ([`crate::emit::ray_guide`]) on the CPU: the code
/// `emit/intersect.rs` emits for a box node, over a [`Query`].
///
/// - The root of a guided top level: a new ray is traced; then **step**.
/// - The same node under the other box kind (the "continue mark", which no
///   node has): the title's extent at or under the candidate's distance
///   means the title accepted it, and it is committed; then **step**.
/// - Another box node while a candidate waits, and the node is the root of
///   the candidate's bottom level: the candidate's triangle node, alone.
/// - Anything else: the console's test.
/// - **step**: the next candidate's instance node and the continue mark,
///   or nothing.
pub struct Guided<'a, M: Memory, Q: Query> {
    pub mem: &'a M,
    /// The guide table ([`guide_table`]).
    pub table: &'a [u32],
    pub query: Q,
    /// The candidate that waits for its bottom level's root.
    feed: Option<Candidate>,
    last: Option<Candidate>,
    /// Candidates the query gave, and those committed; of the candidates,
    /// the boxes.
    pub candidates: u32,
    pub committed: u32,
    pub boxes: u32,
}

impl<'a, M: Memory, Q: Query> Guided<'a, M, Q> {
    pub fn new(mem: &'a M, table: &'a [u32], query: Q) -> Self {
        Self { mem, table, query, feed: None, last: None, candidates: 0, committed: 0, boxes: 0 }
    }

    fn step(&mut self, root: u32) -> [u32; 4] {
        self.last = self.query.proceed();
        // (A box has no triangle node to give: the title's own walk goes on
        // from the bottom level's root.)
        self.feed = self.last.filter(|c| !c.boxed);
        self.boxes += u32::from(self.last.is_some_and(|c| c.boxed));
        match self.last {
            Some(c) => {
                self.candidates += 1;
                [c.contribution << 3 | 6, root ^ 1, NONE, NONE]
            }
            None => [NONE; 4],
        }
    }
}

impl<M: Memory, Q: Query> NodeTest for Guided<'_, M, Q> {
    fn test(&mut self, base: u64, pointer: u32, ray: &Ray) -> Result<[u32; 4], Error> {
        if pointer & 6 == 4 {
            let address = base + u64::from(pointer >> 3) * 64;
            let mut tops = self.table[4..GUIDE_HEADER as usize].chunks_exact(4);
            let found = tops.find(|t| (u64::from(t[1]) << 32 | u64::from(t[0])) == address).map(|t| (t[2], t[3]));
            if let Some((root, slot)) = found {
                if pointer & 7 == root & 7 {
                    self.query.trace(slot, ray);
                } else if let Some(c) = self.last {
                    if !c.boxed && ray.extent <= c.t {
                        self.query.commit();
                        self.committed += 1;
                    } else if c.boxed && ray.extent < self.query.committed() {
                        self.query.commit_box(ray.extent);
                        self.committed += 1;
                    }
                }
                return Ok(self.step(root));
            }
            if let Some(c) = self.feed
                && self.table[c.id as usize] == guide_key(address)
            {
                self.feed = None;
                return Ok([self.table[(c.id + 1 + c.primitive) as usize], NONE, NONE, NONE]);
            }
        }
        Software(self.mem).test(base, pointer, ray)
    }
}

/// A stand-in for the host's hardware on the CPU: the decoded triangles of
/// a top level, every one tested for every ray. It gives its candidates
/// nearest first or in the order the structures hold them; hardware gives
/// them in the order its own walk meets them, which lies between.
pub struct Brute<'a> {
    pub top: &'a Top,
    /// The bottom levels by address, each with its list's start in the
    /// guide table.
    pub bottoms: &'a HashMap<u64, (Bottom, u32)>,
    pub nearest_first: bool,
    /// Bottom levels the host has no structure of, with their bounds: each
    /// instance of one is a box ([`Hosted::Bounds`]).
    boxes: Option<&'a HashMap<u64, [f32; 6]>>,
    found: Vec<Candidate>,
    at: usize,
    limit: f32,
}

impl<'a> Brute<'a> {
    pub fn new(top: &'a Top, bottoms: &'a HashMap<u64, (Bottom, u32)>, nearest_first: bool) -> Self {
        Self { top, bottoms, nearest_first, boxes: None, found: Vec::new(), at: 0, limit: 0.0 }
    }

    /// With box stand-ins for these bottom levels.
    pub fn with_boxes(mut self, boxes: &'a HashMap<u64, [f32; 6]>) -> Self {
        self.boxes = Some(boxes);
        self
    }
}

/// Where a ray enters the box `bounds` (0: it starts inside), if it crosses
/// it before its extent.
pub fn crosses(bounds: &[f32; 6], ray: &Ray) -> Option<f32> {
    let (mut enter, mut leave) = (0f32, ray.extent);
    for a in 0..3 {
        let (t0, t1) = ((bounds[a] - ray.origin[a]) * ray.inverse[a], (bounds[3 + a] - ray.origin[a]) * ray.inverse[a]);
        // (A ray along a face: not a number, which neither bound takes.)
        enter = enter.max(t0.min(t1));
        leave = leave.min(t0.max(t1));
    }
    (enter <= leave).then_some(enter)
}

impl Query for Brute<'_> {
    fn trace(&mut self, _slot: u32, ray: &Ray) {
        self.found.clear();
        (self.at, self.limit) = (0, ray.extent);
        for i in &self.top.instances {
            if i.mask == 0 || i.bottom == 0 {
                continue;
            }
            // The ray in the instance's space, by the title's own matrix.
            let (shift, rows) = i.to_object;
            let turn = |v: [f32; 3]| match rows {
                Some(m) => std::array::from_fn(|a| m[3 * a] * v[0] + m[3 * a + 1] * v[1] + m[3 * a + 2] * v[2]),
                None => v,
            };
            let origin = turn(ray.origin);
            let local = Ray::new(std::array::from_fn(|a| origin[a] + shift[a]), turn(ray.direction), ray.extent);
            let Some((b, list)) = self.bottoms.get(&i.bottom) else {
                // No structure of it: its box, if the host was given one.
                if let Some(bounds) = self.boxes.and_then(|boxes| boxes.get(&i.bottom))
                    && let Some(grown) = stand_in_bounds(bounds)
                    && let Some(t) = crosses(&grown, &local)
                {
                    self.found.push(Candidate { id: 0, contribution: i.pointer >> 3, primitive: 0, t, boxed: true });
                }
                continue;
            };
            let Some((id, contribution)) = host_instance(i, *list) else { continue };
            // (Hardware leaves an instance out by its bounds too; grown, so
            // that no triangle on a face is lost to rounding here.)
            if stand_in_bounds(&b.header.bounds).is_some_and(|grown| crosses(&grown, &local).is_none()) {
                continue;
            }
            for (primitive, t) in b.triangles.iter().enumerate() {
                let [numerator, denominator, ..] = bvh::triangle(&t.vertices, &local);
                let distance = numerator / denominator;
                if numerator.is_finite() && distance > 0.0 && distance < ray.extent {
                    self.found.push(Candidate { id, contribution, primitive: primitive as u32, t: distance, boxed: false });
                }
            }
        }
        if self.nearest_first {
            self.found.sort_by(|a, b| a.t.total_cmp(&b.t));
        }
    }

    fn proceed(&mut self) -> Option<Candidate> {
        while let Some(&c) = self.found.get(self.at) {
            self.at += 1;
            if c.t < self.limit {
                return Some(c);
            }
        }
        None
    }

    fn commit(&mut self) {
        if let Some(c) = self.at.checked_sub(1).and_then(|n| self.found.get(n)) {
            self.limit = c.t;
        }
    }

    fn commit_box(&mut self, t: f32) {
        self.limit = t;
    }

    fn committed(&self) -> f32 {
        self.limit
    }
}

/// Structures in the title's format made by hand, for tests.
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::bvh::tests as nodes;

    /// The 16-bit float nearest to `x` that is not below it (`up`) or not
    /// above it.
    pub(crate) fn half(x: f32, up: bool) -> u16 {
        if x.is_nan() {
            return 0x7e00;
        }
        // The halves of one sign in order of size: 0 .. 0x7c00 (infinity).
        let negative = x.is_sign_negative();
        let size = x.abs();
        let (mut low, mut high) = (0u16, 0x7c00u16);
        // The largest half not above `size`.
        while low < high {
            let mid = (low + high).div_ceil(2);
            if bvh::half_to_f32(mid) <= size {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        // Away from zero when the direction asks for a larger size.
        let exact = bvh::half_to_f32(low) == size;
        let bits = if !exact && up != negative { low + 1 } else { low };
        bits | u16::from(negative) << 15
    }

    #[test]
    fn halves_round_outward() {
        for x in [0.1f32, -0.1, 1.0, -2.5, 1000.3, -1000.3, 1e-6, 70000.0, 0.0] {
            let (up, down) = (bvh::half_to_f32(half(x, true)), bvh::half_to_f32(half(x, false)));
            assert!(down <= x && x <= up, "{x}: {down} {up}");
            assert_eq!(half(x, true), nodes::f32_to_half(x, true), "{x} up");
            assert_eq!(half(x, false), nodes::f32_to_half(x, false), "{x} down");
        }
    }

    /// A box with nothing in it, as the title's builder leaves an unused
    /// child's.
    const EMPTY: [f32; 6] =
        [f32::INFINITY, f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY];

    fn around(boxes: &[[f32; 6]]) -> [f32; 6] {
        let mut all = EMPTY;
        for b in boxes {
            for a in 0..3 {
                all[a] = all[a].min(b[a]);
                all[3 + a] = all[3 + a].max(b[3 + a]);
            }
        }
        all
    }

    /// A 16-bit box node holding `boxes` rounded outward
    /// ([`nodes::box16`], without its search through every half).
    fn box16(children: [u32; 4], boxes: [[f32; 6]; 4]) -> [u32; 28] {
        let mut w = [0u32; 28];
        w[..4].copy_from_slice(&children);
        for (b, bx) in boxes.iter().enumerate() {
            for (k, x) in bx.iter().enumerate() {
                let n = 6 * b + k;
                w[4 + n / 2] |= u32::from(half(*x, k >= 3)) << (16 * (n % 2));
            }
        }
        w
    }

    /// A tree of box nodes over `leaves` (a pointer and its bounds each):
    /// the nodes' dwords from node `first` on and the root's pointer. Box
    /// nodes take 16-bit bounds where `narrow` says so of their number.
    fn tree(leaves: &[(u32, [f32; 6])], first: u32, narrow: &dyn Fn(u32) -> bool) -> (Vec<u32>, u32) {
        fn build(
            items: &mut [(u32, [f32; 6])],
            words: &mut Vec<u32>,
            first: u32,
            narrow: &dyn Fn(u32) -> bool,
        ) -> (u32, [f32; 6]) {
            let number = first + (words.len() / 16) as u32;
            let half = narrow(number);
            let at = words.len();
            words.resize(at + if half { 16 } else { 32 }, 0);
            let mut children = [NONE; 4];
            let mut boxes = [EMPTY; 4];
            if items.len() <= 4 {
                for (n, item) in items.iter().enumerate() {
                    (children[n], boxes[n]) = *item;
                }
            } else {
                // Four runs along the longest axis.
                let all = around(&items.iter().map(|i| i.1).collect::<Vec<_>>());
                let axis = (0..3).max_by(|&a, &b| (all[3 + a] - all[a]).total_cmp(&(all[3 + b] - all[b]))).unwrap();
                items.sort_by(|a, b| (a.1[axis] + a.1[3 + axis]).total_cmp(&(b.1[axis] + b.1[3 + axis])));
                let run = items.len().div_ceil(4);
                for (n, part) in items.chunks_mut(run).enumerate() {
                    (children[n], boxes[n]) = match part {
                        [one] => *one,
                        _ => build(part, words, first, narrow),
                    };
                }
            }
            let node = if half { box16(children, boxes) } else { nodes::box32(children, boxes) };
            let dwords = if half { 16 } else { 28 };
            words[at..at + dwords].copy_from_slice(&node[..dwords]);
            // What the node holds, as the node test will read it.
            let held = if half { bvh::boxes16(&node) } else { boxes };
            let live: Vec<[f32; 6]> = (0..4).filter(|&n| children[n] != NONE).map(|n| held[n]).collect();
            (number << 3 | if half { 4 } else { 5 }, around(&live))
        }
        let mut words = Vec::new();
        let mut items = leaves.to_vec();
        let (root, _) = build(&mut items, &mut words, first, narrow);
        (words, root)
    }

    fn header_words(marks: u32, nodes: u32, elements: u32, bounds: [f32; 6], root: u32, elements_at: u32) -> [u32; 16] {
        let mut w = [0u32; 16];
        w[0] = VERSION | marks << 16;
        w[1] = nodes;
        w[3] = elements;
        for k in 0..6 {
            w[4 + k] = bounds[k].to_bits();
        }
        (w[10], w[12], w[13]) = (64, root, elements_at);
        w
    }

    /// What a made-up bottom level is made of: a triangle, or two that
    /// share an edge (a, b, c) and (b, d, c), with the geometry's index and
    /// opaque flag.
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct Piece {
        pub a: [f32; 3],
        pub b: [f32; 3],
        pub c: [f32; 3],
        pub d: Option<[f32; 3]>,
        pub geometry: u32,
        pub opaque: bool,
    }

    /// A bottom level of `pieces` in the title's format: the header, the
    /// box nodes, the triangle nodes in one run. `turn`: the vertices are
    /// stored in another order by the piece's number, and the vertex order
    /// bytes say so, as the title's builder does.
    pub(crate) fn bottom_words(pieces: &[Piece], turn: bool) -> Vec<u32> {
        // The triangle nodes come after the boxes: build the tree over
        // placeholders first to know how many nodes the boxes take.
        let bounds: Vec<[f32; 6]> = pieces
            .iter()
            .map(|p| {
                let mut b = EMPTY;
                for v in [Some(p.a), Some(p.b), Some(p.c), p.d].into_iter().flatten() {
                    for a in 0..3 {
                        b[a] = b[a].min(v[a]);
                        b[3 + a] = b[3 + a].max(v[a]);
                    }
                }
                b
            })
            .collect();
        let narrow = |n: u32| n % 3 != 0;
        let count = |first_leaf: u32| {
            let leaves: Vec<_> = bounds.iter().enumerate().map(|(n, b)| ((first_leaf + n as u32) << 3, *b)).collect();
            tree(&leaves, 1, &narrow)
        };
        let boxes = (count(0).0.len() / 16) as u32;
        let first_leaf = 1 + boxes;
        let (box_words, root) = count(first_leaf);
        assert_eq!(box_words.len() / 16, boxes as usize);
        let total = first_leaf + pieces.len() as u32;
        let mut words = header_words(MARK_BOTTOM | 2, total, pieces.len() as u32, around(&bounds), root, first_leaf * 64).to_vec();
        words.extend(box_words);
        let mut primitive = 0;
        for (n, p) in pieces.iter().enumerate() {
            // Triangle 0 is the node's vertices 0, 1, 2 and triangle 1 its
            // vertices 1, 3, 2. Which of a triangle's three the weights I
            // and J belong to is the order byte's to say: the input
            // triangle's second and third vertex, wherever they are stored.
            let orders = [(1u32, 2u32), (2, 0), (0, 1), (2, 1), (0, 2), (1, 0)];
            // (A pair shares the node's vertices 1 and 2: b and c, either
            // way round.)
            let (i, j) = match (turn, p.d) {
                (false, _) => orders[0],
                (true, None) => orders[n % 6],
                (true, Some(_)) => orders[3 * (n % 2)],
            };
            let mut stored = [[0f32; 3]; 4];
            stored[(3 - i - j) as usize] = p.a;
            stored[i as usize] = p.b;
            stored[j as usize] = p.c;
            let mut id = i | j << 2;
            if let Some(d) = p.d {
                stored[3] = d;
                // Triangle 1 = stored (1, 3, 2), to be (b, d, c): its own
                // vertices 0 and 2 are b and c, or c and b.
                id |= if i == 1 { 1 | 2 << 2 } else { 1 } << 8;
            }
            let mut w = nodes::triangles(&stored, id);
            w[12] = u32::from(p.opaque) | if p.d.is_some() { 0x80 } else { 0 } | p.geometry << 8;
            w[13] = primitive;
            w[14] = primitive + 1;
            primitive += 1 + u32::from(p.d.is_some());
            words.extend(&w[..16]);
        }
        words
    }

    /// An instance of a made-up top level.
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct Placed {
        /// Object to world: three rows of four.
        pub matrix: [f32; 12],
        pub id: u32,
        pub mask: u8,
        pub contribution: u32,
        pub flags: u8,
        /// The bottom level's address, its root and its bounds; none: an
        /// instance of nothing.
        pub bottom: Option<(u64, u32, [f32; 6])>,
    }

    /// The inverse of a 3x4 matrix: the move and the rows.
    pub(crate) fn inverse(m: &[f32; 12]) -> ([f32; 3], [f32; 9]) {
        let a = |r: usize, c: usize| f64::from(m[4 * r + c]);
        let cofactor = |r: usize, c: usize| {
            let (r1, r2, c1, c2) = ((r + 1) % 3, (r + 2) % 3, (c + 1) % 3, (c + 2) % 3);
            a(r1, c1) * a(r2, c2) - a(r1, c2) * a(r2, c1)
        };
        let det = a(0, 0) * cofactor(0, 0) + a(0, 1) * cofactor(0, 1) + a(0, 2) * cofactor(0, 2);
        let mut rows = [0f32; 9];
        let mut shift = [0f32; 3];
        for r in 0..3 {
            let mut t = 0.0;
            for c in 0..3 {
                let x = cofactor(c, r) / det;
                rows[3 * r + c] = x as f32;
                t -= x * a(c, 3);
            }
            shift[r] = t as f32;
        }
        (shift, rows)
    }

    /// A top level of `placed` in the title's format: the header, the box
    /// nodes, the instance nodes (two nodes each) in one run.
    pub(crate) fn top_words(placed: &[Placed]) -> Vec<u32> {
        let bounds: Vec<[f32; 6]> = placed
            .iter()
            .map(|p| {
                let Some((_, _, b)) = p.bottom else { return [0.0; 6] };
                let mut world = EMPTY;
                for corner in 0..8 {
                    let v: [f32; 3] = std::array::from_fn(|a| b[if corner >> a & 1 != 0 { 3 + a } else { a }]);
                    for r in 0..3 {
                        let x = p.matrix[4 * r] * v[0] + p.matrix[4 * r + 1] * v[1] + p.matrix[4 * r + 2] * v[2] + p.matrix[4 * r + 3];
                        world[r] = world[r].min(x);
                        world[3 + r] = world[3 + r].max(x);
                    }
                }
                // Room for the rounding of the walk's own matrix.
                std::array::from_fn(|k| world[k] + if k < 3 { -1.0 } else { 1.0 } * (1e-3 + 1e-4 * world[k].abs()))
            })
            .collect();
        let narrow = |n: u32| n % 2 == 0;
        let count = |first_leaf: u32| {
            let leaves: Vec<_> =
                bounds.iter().enumerate().map(|(n, b)| ((first_leaf + 2 * n as u32) << 3 | 6, *b)).collect();
            tree(&leaves, 1, &narrow)
        };
        let boxes = (count(0).0.len() / 16) as u32;
        let first_leaf = 1 + boxes;
        let (box_words, root) = count(first_leaf);
        let total = first_leaf + 2 * placed.len() as u32;
        let mut words = header_words(2, total, placed.len() as u32, around(&bounds), root, first_leaf * 64).to_vec();
        words.extend(box_words);
        for p in placed {
            let mut w = [0u32; 32];
            for k in 0..12 {
                w[k] = p.matrix[k].to_bits();
            }
            w[12] = p.id | u32::from(p.mask) << 24;
            w[13] = p.contribution | u32::from(p.flags) << 24;
            (w[14], w[15]) = match p.bottom {
                Some((address, root, _)) => (root, (address >> 8) as u32),
                None => (NO_ROOT, 0),
            };
            let (shift, rows) = inverse(&p.matrix);
            for a in 0..3 {
                w[16 + a] = shift[a].to_bits();
            }
            for k in 0..9 {
                w[19 + k] = rows[k].to_bits();
            }
            w[28] = 0x1234;
            words.extend(w);
        }
        words
    }

    /// A small generator of repeatable numbers.
    pub(crate) struct Numbers(pub u64);

    impl Numbers {
        pub(crate) fn next(&mut self) -> u32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }

        /// A float in -1..1.
        pub(crate) fn unit(&mut self) -> f32 {
            (self.next() % 16385) as f32 / 8192.0 - 1.0
        }

        pub(crate) fn vector(&mut self, scale: f32) -> [f32; 3] {
            [self.unit() * scale, self.unit() * scale, self.unit() * scale]
        }
    }

    /// A made-up scene in title memory: three bottom levels (a ground of
    /// paired triangles with bumps, a closed ball of single triangles, a box
    /// of pairs) and a top level of `instances` instances of them, turned,
    /// scaled and moved; some with a mask no walk enters, some with their
    /// culling off or their front faces turned, one of nothing.
    pub(crate) struct Scene {
        pub memory: Pages,
        pub top: u64,
        pub bottoms: Vec<u64>,
        /// The world's bounds.
        pub bounds: [f32; 6],
    }

    pub(crate) fn scene(seed: u64, instances: usize, at: u64) -> Scene {
        let mut g = Numbers(seed);
        let mut memory = Pages::default();
        // A ground: 12 x 12 cells of two triangles each.
        let mut ground = Vec::new();
        let height: Vec<f32> = (0..13 * 13).map(|_| g.unit() * 0.4).collect();
        let corner = |x: usize, z: usize| [x as f32 - 6.0, height[13 * z + x], z as f32 - 6.0];
        for z in 0..12 {
            for x in 0..12 {
                ground.push(Piece {
                    a: corner(x, z),
                    b: corner(x, z + 1),
                    c: corner(x + 1, z),
                    d: Some(corner(x + 1, z + 1)),
                    geometry: 0,
                    opaque: (x + z) % 5 != 0,
                });
            }
        }
        // A ball: 8 x 12 faces, each split in two single triangles.
        let mut ball = Vec::new();
        let on = |u: usize, v: usize| {
            let (a, b) = (v as f32 / 8.0 * std::f32::consts::PI, u as f32 / 12.0 * std::f32::consts::TAU);
            [a.sin() * b.cos() * 2.0, a.cos() * 2.0, a.sin() * b.sin() * 2.0]
        };
        for v in 0..8 {
            for u in 0..12 {
                let (p, q, r, s) = (on(u, v), on(u + 1, v), on(u, v + 1), on(u + 1, v + 1));
                if v > 0 {
                    ball.push(Piece { a: p, b: q, c: r, d: None, geometry: 1, opaque: true });
                }
                if v < 7 {
                    ball.push(Piece { a: q, b: s, c: r, d: None, geometry: 2, opaque: u % 2 == 0 });
                }
            }
        }
        // A box: six faces of a pair each.
        let mut cube = Vec::new();
        for axis in 0..3 {
            for side in [-1.0f32, 1.0] {
                let at = |u: f32, v: f32| {
                    let mut p = [0f32; 3];
                    p[axis] = side;
                    p[(axis + 1) % 3] = u * side;
                    p[(axis + 2) % 3] = v;
                    p
                };
                cube.push(Piece {
                    a: at(-1.0, -1.0),
                    b: at(-1.0, 1.0),
                    c: at(1.0, -1.0),
                    d: Some(at(1.0, 1.0)),
                    geometry: 3,
                    opaque: true,
                });
            }
        }
        let mut bottoms = Vec::new();
        let mut made = Vec::new();
        for (n, pieces) in [ground, ball, cube].iter().enumerate() {
            let address = at + 0x10_0000 * (n as u64 + 1);
            let words = bottom_words(pieces, true);
            let h = header(&Span { base: address, words: &words }, address).expect("a header");
            memory.write(address, &words);
            bottoms.push(address);
            made.push((address, h.root, h.bounds));
        }
        let mut placed = Vec::new();
        for n in 0..instances {
            let which = if n == 0 { 0 } else { g.next() as usize % 3 };
            // A turn about a random axis, a scale, a move.
            let axis = {
                let v = g.vector(1.0);
                let size = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-3);
                v.map(|x| x / size)
            };
            let angle = if n == 0 { 0.0 } else { g.unit() * 3.0 };
            let scale = if n == 0 { 4.0 } else { 0.5 + (g.unit() + 1.0) };
            let shift = if n == 0 { [0.0, -3.0, 0.0] } else { g.vector(20.0) };
            let (s, c) = angle.sin_cos();
            let mut matrix = [0f32; 12];
            for r in 0..3 {
                for col in 0..3 {
                    let cross = [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]][r][col];
                    let turn = c * f32::from(u8::from(r == col)) + s * cross + (1.0 - c) * axis[r] * axis[col];
                    matrix[4 * r + col] = turn * scale;
                }
                matrix[4 * r + 3] = shift[r];
            }
            placed.push(Placed {
                matrix,
                id: n as u32,
                mask: match n % 7 {
                    3 => 0x01,
                    5 => 0x80,
                    _ => 0xff,
                },
                contribution: 3 * n as u32,
                flags: match n % 5 {
                    1 => FLAG_CULL_DISABLE,
                    2 => FLAG_FRONT_CCW,
                    _ => 0,
                },
                bottom: (n % 11 != 10).then(|| made[which]),
            });
        }
        let words = top_words(&placed);
        memory.write(at, &words);
        let h = header(&memory, at).expect("a header");
        Scene { memory, top: at, bottoms, bounds: h.bounds }
    }

    /// A ray for the scene: from a random place towards a random point of
    /// the world's bounds.
    pub(crate) fn ray(g: &mut Numbers, bounds: &[f32; 6], extent: f32) -> Ray {
        let inside = |g: &mut Numbers, grow: f32| -> [f32; 3] {
            std::array::from_fn(|a| {
                let (mid, half) = ((bounds[a] + bounds[3 + a]) / 2.0, (bounds[3 + a] - bounds[a]) / 2.0);
                mid + g.unit() * half * grow
            })
        };
        let (from, to) = (inside(g, 1.2), inside(g, 0.9));
        let d: [f32; 3] = std::array::from_fn(|a| to[a] - from[a]);
        let size = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-3);
        Ray::new(from, d.map(|x| x / size), extent)
    }

    const SHADOW: Rules = Rules { mask: 0x40, cull: 0, first_hit: true };
    const NEAREST: Rules = Rules { mask: 0xda, cull: 1, first_hit: false };

    #[test]
    fn a_bottom_level_decodes_to_the_triangles_it_was_made_of() {
        let a = [0.0, 0.0, 0.0];
        let pieces = [
            Piece { a, b: [0.0, 1.0, 0.0], c: [1.0, 0.0, 0.0], d: Some([1.0, 1.0, 0.0]), geometry: 5, opaque: true },
            Piece { a: [2.0, 0.0, 1.0], b: [2.0, 1.0, 1.0], c: [3.0, 0.0, 1.0], d: None, geometry: 6, opaque: false },
            Piece { a: [4.0, 0.0, 2.0], b: [4.0, 1.0, 2.0], c: [5.0, 0.0, 2.0], d: None, geometry: 7, opaque: true },
        ];
        for turn in [false, true] {
            let words = bottom_words(&pieces, turn);
            let base = 0xD_0001_0000u64;
            let mem = Span { base, words: &words };
            let b = bottom(&mem, base).expect("decodes");
            assert_eq!(b.triangles.len(), 4);
            assert!(b.one_run);
            assert_eq!(b.header.elements, 3);
            // In the order the input gave them, whatever order the node
            // stores them in: first vertex, then I's, then J's.
            let mut found: Vec<&Triangle> = b.triangles.iter().collect();
            found.sort_by_key(|t| t.primitive);
            assert_eq!(found[0].vertices, [pieces[0].a, pieces[0].b, pieces[0].c]);
            assert_eq!(found[1].vertices, [pieces[0].b, [1.0, 1.0, 0.0], pieces[0].c]);
            assert_eq!(found[2].vertices, [pieces[1].a, pieces[1].b, pieces[1].c]);
            assert_eq!(found[3].vertices, [pieces[2].a, pieces[2].b, pieces[2].c]);
            assert_eq!(found.iter().map(|t| t.geometry).collect::<Vec<_>>(), [5, 5, 6, 7]);
            assert_eq!(found.iter().map(|t| t.opaque).collect::<Vec<_>>(), [true, true, false, true]);
            assert_eq!(found.iter().map(|t| t.primitive).collect::<Vec<_>>(), [0, 1, 2, 3]);
            // The pair shares its node; the pointer's low bits are the slot.
            assert_eq!(found[0].node, found[1].node);
            assert_eq!((found[0].pointer & 7, found[1].pointer & 7), (0, 1));
            assert_eq!(found[2].node, base + u64::from(found[2].pointer >> 3) * 64);
            assert!(u64::from(b.header.elements_at) <= found[0].node - base);
        }
    }

    #[test]
    fn a_top_level_decodes_to_its_instances() {
        let s = scene(7, 40, 0x1_2340_0000);
        let t = top(&s.memory, s.top).expect("decodes");
        assert_eq!(t.instances.len(), 40);
        assert_eq!(t.others, 0);
        let mut found = t.instances.clone();
        found.sort_by_key(|i| i.id);
        for (n, i) in found.iter().enumerate() {
            assert_eq!(i.id, n as u32);
            assert_eq!(i.contribution, 3 * n as u32);
            assert_eq!(i.pointer & 7, 6);
            // The instances are one run of two nodes each, from the
            // header's elements offset.
            assert_eq!(i.node, s.top + u64::from(t.header.elements_at) + 128 * n as u64);
            assert_eq!(i.bottom == 0, n % 11 == 10);
            if i.bottom != 0 {
                assert!(s.bottoms.contains(&i.bottom));
                assert_eq!(header(&s.memory, i.bottom).unwrap().root, i.root);
            }
            // The two matrices are each other's inverse.
            let (shift, rows) = (i.to_object.0, i.to_object.1.expect("rows"));
            for r in 0..3 {
                for c in 0..3 {
                    let x: f32 = (0..3).map(|k| rows[3 * r + k] * i.matrix[4 * k + c]).sum();
                    assert!((x - f32::from(u8::from(r == c))).abs() < 1e-4, "{n}: {x}");
                }
                let moved: f32 = (0..3).map(|k| rows[3 * r + k] * i.matrix[4 * k + 3]).sum::<f32>() + shift[r];
                assert!(moved.abs() < 1e-3, "{n}: {moved}");
            }
        }
        // Each bottom level decodes, and a top level is none.
        for &b in &s.bottoms {
            assert!(!bottom(&s.memory, b).unwrap().triangles.is_empty());
        }
        assert_eq!(bottom(&s.memory, s.top), Err(Error::WrongLevel(s.top)));
        assert_eq!(top(&s.memory, s.bottoms[0]).map(|_| ()), Err(Error::WrongLevel(s.bottoms[0])));
    }

    #[test]
    fn what_is_no_structure_is_refused_by_name() {
        let mut words = bottom_words(
            &[Piece { a: [0.0; 3], b: [0.0, 1.0, 0.0], c: [1.0, 0.0, 0.0], d: None, geometry: 0, opaque: true }],
            false,
        );
        let base = 0x4000u64;
        assert_eq!(header(&Span { base, words: &words }, base + 0x1_0000), Err(Error::Unmapped(base + 0x1_0000)));
        words[0] |= MARK_PACKED << 16;
        assert_eq!(bottom(&Span { base, words: &words }, base).map(|_| ()), Err(Error::Packed(base)));
        words[0] = 5;
        assert_eq!(bottom(&Span { base, words: &words }, base).map(|_| ()), Err(Error::NoHeader(base)));
    }

    /// The nearest triangle of all the decoded ones, each tested by the
    /// console's own triangle test in its instance's space: what a walk
    /// must find whatever boxes lie over the triangles.
    pub(crate) fn nearest(t: &Top, bottoms: &HashMap<u64, (Bottom, u32)>, ray: &Ray, rules: Rules) -> Option<(f32, u32, u32)> {
        let mut best: Option<(f32, u32, u32)> = None;
        for i in &t.instances {
            if i.mask & rules.mask == 0 || i.bottom == 0 {
                continue;
            }
            let (shift, rows) = (i.to_object.0, i.to_object.1.unwrap());
            let turn = |v: [f32; 3]| -> [f32; 3] {
                std::array::from_fn(|a| rows[3 * a] * v[0] + rows[3 * a + 1] * v[1] + rows[3 * a + 2] * v[2])
            };
            let origin = turn(ray.origin);
            let local = Ray::new(std::array::from_fn(|a| origin[a] + shift[a]), turn(ray.direction), ray.extent);
            let cull = rules.cull
                * match (i.flags & FLAG_CULL_DISABLE != 0, i.flags & FLAG_FRONT_CCW != 0) {
                    (true, _) => 0,
                    (false, false) => 1,
                    (false, true) => -1,
                };
            for tri in &bottoms[&i.bottom].0.triangles {
                let [numerator, denominator, ..] = bvh::triangle(&tri.vertices, &local);
                let d = (1.0 / denominator) * numerator;
                let face = i32::from((denominator.to_bits() >> 16) as i16).signum();
                if d < 0.0 || face * cull < 0 || d >= best.map_or(ray.extent, |b| b.0) {
                    continue;
                }
                best = Some((d, i.pointer, tri.pointer));
            }
        }
        best
    }

    fn decoded(s: &Scene) -> (Top, HashMap<u64, (Bottom, u32)>, Vec<u32>) {
        let t = top(&s.memory, s.top).unwrap();
        let list: Vec<Bottom> = s.bottoms.iter().map(|&b| bottom(&s.memory, b).unwrap()).collect();
        let (table, starts) = guide_table(
            &[GuidedTop { address: s.top, root: t.header.root, slot: 9 }],
            &list.iter().collect::<Vec<_>>(),
        );
        let bottoms = list.into_iter().zip(starts).map(|(b, start)| (b.header.address, (b, start))).collect();
        (t, bottoms, table)
    }

    #[test]
    fn the_walk_finds_the_nearest_of_the_decoded_triangles() {
        let s = scene(11, 60, 0x1_2340_0000);
        let (t, bottoms, _) = decoded(&s);
        let mut g = Numbers(3);
        let (mut hits, mut turns, mut deepest) = (0, 0u64, 0);
        for n in 0..3000 {
            let r = ray(&mut g, &s.bounds, if n % 4 == 0 { 9.0 } else { 1000.0 });
            let (found, counted) = walk(&s.memory, s.top, &r, NEAREST).expect("walks");
            let want = nearest(&t, &bottoms, &r, NEAREST);
            // The same distance; the same triangle unless two lie at one
            // distance (a ray through an edge two triangles share).
            match (found, want) {
                (None, None) => {}
                (Some(f), Some(w)) => {
                    assert_eq!(f.t, w.0, "ray {n}");
                    hits += 1;
                }
                other => panic!("ray {n}: {other:?}"),
            }
            turns += u64::from(counted.total());
            deepest = deepest.max(counted.deepest);
        }
        assert!(hits > 500 && hits < 2900, "{hits} hits of 3000");
        assert!(deepest < 16, "the title's stack would overflow: {deepest}");
        assert!(turns > 3000 * 5, "{turns}");
    }

    /// The guided node test over a stand-in for the hardware gives the walk
    /// the same end as the console's test, in fewer turns.
    #[test]
    fn the_guided_walk_ends_as_the_software_walk_does() {
        let s = scene(11, 60, 0x1_2340_0000);
        let (t, bottoms, table) = decoded(&s);
        for rules in [SHADOW, NEAREST, Rules { mask: 0xff, cull: -1, first_hit: false }] {
            for nearest_first in [true, false] {
                let mut g = Numbers(5);
                let (mut plain_turns, mut guided_turns, mut candidates, mut hits) = (0u64, 0u64, 0u64, 0);
                for n in 0..2000 {
                    let r = ray(&mut g, &s.bounds, if n % 4 == 0 { 9.0 } else { 1000.0 });
                    let (plain, a) = walk(&s.memory, s.top, &r, rules).expect("walks");
                    let mut guided = Guided::new(&s.memory, &table, Brute::new(&t, &bottoms, nearest_first));
                    let (led, b) = walk_with(&s.memory, &mut guided, s.top, &r, rules).expect("walks");
                    match (plain, led) {
                        (None, None) => {}
                        // A shadow ray ends at the first triangle accepted,
                        // which need not be the same one.
                        (Some(_), Some(_)) if rules.first_hit => hits += 1,
                        (Some(p), Some(l)) => {
                            assert_eq!(p.t, l.t, "ray {n}");
                            assert_eq!((p.front, p.i, p.j), (l.front, l.i, l.j), "ray {n}");
                            hits += 1;
                        }
                        other => panic!("ray {n} {rules:?}: {other:?}"),
                    }
                    assert!(b.deepest <= 1, "only the continue mark waits: {}", b.deepest);
                    // The root, then per candidate its bottom level's root
                    // (when its instance is entered) and the mark.
                    assert!(b.boxes16 + b.boxes32 <= 1 + 2 * guided.candidates, "{b:?} {}", guided.candidates);
                    plain_turns += u64::from(a.total());
                    guided_turns += u64::from(b.total());
                    candidates += u64::from(guided.candidates);
                }
                assert!(hits > 200, "{hits}");
                assert!(guided_turns < plain_turns, "{guided_turns} {plain_turns}");
                eprintln!(
                    "{rules:?} nearest first {nearest_first}: turns a ray {:.1} -> {:.1}, candidates a ray {:.2}",
                    plain_turns as f64 / 2000.0,
                    guided_turns as f64 / 2000.0,
                    candidates as f64 / 2000.0
                );
            }
        }
    }

    /// The host table finds what was put in it and nothing else.
    #[test]
    fn the_host_table_finds_its_bottom_levels() {
        let mut g = Numbers(9);
        let built: Vec<(u64, u32, u64)> =
            (0..900u32).map(|n| (0xD_0000_0000 + (u64::from(g.next()) << 8), 20 + 7 * n, 0x7_0000_0000 + 65536 * u64::from(n))).collect();
        assert!(hosts_table(&built, 64).is_none(), "900 keys have no room in 64 places");
        let table = hosts_table(&built, 4096).expect("room");
        assert_eq!(table.len(), (HOSTS_HEADER + 4 * 4096) as usize);
        for &(bottom, list, structure) in &built {
            assert_eq!(hosts_find(&table, bottom), Some((list, structure)));
        }
        for n in 0..2000u64 {
            let other = 0xE_0000_0000 + (n << 8);
            assert_eq!(hosts_find(&table, other), None);
        }
    }

    /// A stand-in instance's box, moved by its matrix, is the bottom
    /// level's bounds (grown) moved by the title's matrix.
    #[test]
    fn a_stand_in_covers_its_bottom_level() {
        let s = scene(7, 40, 0x1_2340_0000);
        let t = top(&s.memory, s.top).unwrap();
        for i in t.instances.iter().filter(|i| i.bottom != 0) {
            let bounds = header(&s.memory, i.bottom).unwrap().bounds;
            let grown = stand_in_bounds(&bounds).unwrap();
            let m = stand_in_matrix(&i.matrix, &bounds).unwrap();
            for corner in 0..8 {
                let unit: [f32; 3] = std::array::from_fn(|a| f32::from(u8::from(corner >> a & 1 != 0)));
                let local: [f32; 3] = std::array::from_fn(|a| if corner >> a & 1 != 0 { grown[3 + a] } else { grown[a] });
                for r in 0..3 {
                    let by_box = m[4 * r] * unit[0] + m[4 * r + 1] * unit[1] + m[4 * r + 2] * unit[2] + m[4 * r + 3];
                    let by_title = i.matrix[4 * r] * local[0] + i.matrix[4 * r + 1] * local[1] + i.matrix[4 * r + 2] * local[2] + i.matrix[4 * r + 3];
                    assert!((by_box - by_title).abs() <= 1e-3 * by_title.abs().max(1.0), "{by_box} {by_title}");
                }
            }
            let w = host_instance_words(i, Hosted::Bounds { bounds, structure: 0x12_3456_7800 }).unwrap();
            assert_eq!((w[12], w[13] & 0xff_ffff), (u32::from(i.mask) << 24, i.pointer >> 3));
            assert_eq!((w[14], w[15]), (0x3456_7800, 0x12));
        }
        assert_eq!(stand_in_bounds(&[0.0, 0.0, 0.0, f32::INFINITY, 1.0, 1.0]), None);
        assert_eq!(stand_in_bounds(&[1.0, 0.0, 0.0, 0.0, 1.0, 1.0]), None);
        assert!(stand_in_bounds(&[2.0, 3.0, 4.0, 2.0, 3.0, 4.0]).is_some_and(|b| b[3] > b[0]));
    }

    /// With box stand-ins for some or all bottom levels (the state of the
    /// first frames, and of what the title rebuilds every frame), the
    /// guided walk still ends as the software walk does.
    #[test]
    fn the_guided_walk_with_stand_ins_ends_as_the_software_walk_does() {
        let s = scene(11, 60, 0x1_2340_0000);
        let (t, all, _) = decoded(&s);
        // Which bottom levels the host has structures of: all, the first
        // alone, none.
        for built in [3usize, 1, 0] {
            let list: Vec<&Bottom> = s.bottoms.iter().take(built).map(|b| &all[b].0).collect();
            let (table, starts) = guide_table(&[GuidedTop { address: s.top, root: t.header.root, slot: 9 }], &list);
            let bottoms: HashMap<u64, (Bottom, u32)> =
                list.iter().zip(starts).map(|(b, start)| (b.header.address, ((*b).clone(), start))).collect();
            let boxes: HashMap<u64, [f32; 6]> = s.bottoms.iter().skip(built).map(|b| (*b, all[b].0.header.bounds)).collect();
            for rules in [SHADOW, NEAREST] {
                let mut g = Numbers(5);
                let (mut plain_turns, mut guided_turns, mut candidates, mut box_candidates, mut hits) = (0u64, 0u64, 0u64, 0u64, 0);
                for n in 0..2000 {
                    let r = ray(&mut g, &s.bounds, if n % 4 == 0 { 9.0 } else { 1000.0 });
                    let (plain, a) = walk(&s.memory, s.top, &r, rules).expect("walks");
                    let mut guided = Guided::new(&s.memory, &table, Brute::new(&t, &bottoms, false).with_boxes(&boxes));
                    let (led, b) = walk_with(&s.memory, &mut guided, s.top, &r, rules).expect("walks");
                    match (plain, led) {
                        (None, None) => {}
                        (Some(_), Some(_)) if rules.first_hit => hits += 1,
                        (Some(p), Some(l)) => {
                            assert_eq!((p.t, p.front, p.i, p.j), (l.t, l.front, l.i, l.j), "ray {n}, {built} built");
                            hits += 1;
                        }
                        other => panic!("ray {n} {rules:?}, {built} built: {other:?}"),
                    }
                    plain_turns += u64::from(a.total());
                    guided_turns += u64::from(b.total());
                    candidates += u64::from(guided.candidates);
                    box_candidates += u64::from(guided.boxes);
                }
                assert!(hits > 200, "{hits}");
                eprintln!(
                    "{built} of 3 bottom levels with a host structure, {rules:?}: turns a ray {:.1} -> {:.1}, candidates a ray {:.2} ({:.2} of them boxes)",
                    plain_turns as f64 / 2000.0,
                    guided_turns as f64 / 2000.0,
                    candidates as f64 / 2000.0,
                    box_candidates as f64 / 2000.0
                );
            }
        }
    }

    /// A dump as a diagnostics build writes it reads back: the arenas'
    /// pages, a top level that lay in a title buffer (`title.bin`), and a
    /// page of an arena that came after the copies were taken (no file).
    #[test]
    fn a_dump_reads_back() {
        let s = scene(7, 40, 0x1_2340_0000);
        // (`VICTORY_RAY_DUMP_MAKE=<folder>`: written there and kept.)
        let kept = std::env::var_os("VICTORY_RAY_DUMP_MAKE").map(std::path::PathBuf::from);
        let folder = kept.clone().unwrap_or_else(|| std::env::temp_dir().join(format!("victory-rays-dump-{}", std::process::id())));
        std::fs::create_dir_all(&folder).unwrap();
        let bytes = |page: u64| -> Vec<u8> { s.memory.page(page).unwrap().iter().flat_map(|w| w.to_le_bytes()).collect() };
        let (mut text, mut arena, mut title) = (format!("walked {:#x}\npacked {:#x} 4096 7\nwhere {:#x} copy 3 {:#x} 65536\n", s.top, s.bottoms[0], s.top, s.top), Vec::new(), Vec::new());
        for page in s.memory.numbers() {
            if page == s.top / PAGE {
                text += &format!("title {:#x} {:#x}\n", page * PAGE, title.len());
                title.extend(bytes(page));
            } else {
                text += &format!("{:#x} 0 {:#x}\n", page * PAGE, arena.len());
                arena.extend(bytes(page));
            }
        }
        text += "frame 99\n0xe00000000 5 0x0\n";
        std::fs::write(folder.join("pages.txt"), text).unwrap();
        std::fs::write(folder.join("arena-0.bin"), arena).unwrap();
        std::fs::write(folder.join("title.bin"), title).unwrap();
        let dump = Dump::open(&folder).expect("a dump");
        if kept.is_none() {
            let _ = std::fs::remove_dir_all(&folder);
        }
        assert_eq!((dump.walked.as_slice(), dump.packed.as_slice(), dump.absent), (&[s.top][..], &[s.bottoms[0]][..], 1));
        assert_eq!(dump.pages.numbers(), s.memory.numbers());
        assert_eq!(top(&dump.pages, s.top), top(&s.memory, s.top));
        for &b in &s.bottoms {
            assert_eq!(bottom(&dump.pages, b), bottom(&s.memory, b));
        }
    }

    /// With `VICTORY_RAY_DUMP=<folder>` (a dump of a run, `VICTORY_MEMORY_DUMP`
    /// in a diagnostics build; run it with `--release`): every structure the
    /// walking programs were given decodes; what the layer's side takes for
    /// granted of the title's layout is counted; and the turns of today's
    /// walk and of the guided one are said, with every bottom level on the
    /// hardware and with the static ones alone (the others as boxes), for
    /// rays of this test's making (through the whole of a top level's
    /// bounds: not the game's rays, whose turns the turn counter of the
    /// same run gives).
    #[test]
    fn a_dump_of_a_run_decodes_and_its_turns_are_counted() {
        let Some(folder) = std::env::var_os("VICTORY_RAY_DUMP") else {
            eprintln!("no VICTORY_RAY_DUMP; skipped");
            return;
        };
        let dump = Dump::open(std::path::Path::new(&folder)).expect("a dump");
        eprintln!(
            "{} pages ({} listed without a file), {} top levels, {} packed",
            dump.pages.numbers().len(),
            dump.absent,
            dump.walked.len(),
            dump.packed.len()
        );
        let packed: HashSet<u64> = dump.packed.iter().copied().collect();
        for &address in &dump.walked {
            let t = match top(&dump.pages, address) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("{address:#x}: {e:?}");
                    continue;
                }
            };
            let mut list = Vec::new();
            let mut missing = 0;
            for i in &t.instances {
                if i.bottom == 0 || list.iter().any(|b: &Bottom| b.header.address == i.bottom) {
                    continue;
                }
                match bottom(&dump.pages, i.bottom) {
                    Ok(b) => list.push(b),
                    Err(_) => missing += 1,
                }
            }
            let count = |of: &dyn Fn(&Bottom) -> bool| -> (usize, usize) {
                let chosen: Vec<&Bottom> = list.iter().filter(|b| of(b)).collect();
                (chosen.len(), chosen.iter().map(|b| b.triangles.len()).sum())
            };
            let (all, fixed) = (count(&|_| true), count(&|b| packed.contains(&b.header.address)));
            // What the layer's side takes for granted.
            let in_run = t
                .instances
                .iter()
                .filter(|i| {
                    (i.node - address)
                        .checked_sub(u64::from(t.header.elements_at))
                        .is_some_and(|k| k % 128 == 0 && k / 128 < u64::from(t.header.elements))
                })
                .count();
            let outside = list
                .iter()
                .filter(|b| {
                    stand_in_bounds(&b.header.bounds).is_none_or(|g| {
                        b.triangles.iter().flat_map(|t| t.vertices).any(|v| (0..3).any(|a| v[a] < g[a] || v[a] > g[3 + a]))
                    })
                })
                .count();
            let of_static = t.instances.iter().filter(|i| packed.contains(&i.bottom)).count();
            eprintln!(
                "{address:#x}: {} instances (the header says {}; {in_run} in the run of instance nodes; {of_static} of static bottom levels, \
                 {} of none); bottom levels decoded {} with {} triangles ({} of them one run of triangle nodes, {} with the header's elements for triangle nodes; {outside} with a triangle outside \
                 their bounds), not decoded {missing}; of the decoded, static {} with {} triangles; nodes {} + {} + {} (64-byte boxes, \
                 128-byte boxes, others) in the top level",
                t.instances.len(),
                t.header.elements,
                t.instances.iter().filter(|i| i.bottom == 0).count(),
                all.0,
                all.1,
                list.iter().filter(|b| b.one_run).count(),
                list.iter().filter(|b| b.one_run && b.leaves == (b.header.elements_at / 64, b.header.elements)).count(),
                fixed.0,
                fixed.1,
                t.boxes[0],
                t.boxes[1],
                t.others,
            );
            for (name, static_only) in [("every bottom level on the hardware", false), ("the static ones on the hardware, the others boxes", true)] {
                let chosen: Vec<&Bottom> = list.iter().filter(|b| !static_only || packed.contains(&b.header.address)).collect();
                let (table, starts) = guide_table(&[GuidedTop { address, root: t.header.root, slot: 0 }], &chosen);
                let bottoms: HashMap<u64, (Bottom, u32)> =
                    chosen.iter().zip(starts).map(|(b, start)| (b.header.address, ((*b).clone(), start))).collect();
                let boxes: HashMap<u64, [f32; 6]> =
                    list.iter().filter(|b| !bottoms.contains_key(&b.header.address)).map(|b| (b.header.address, b.header.bounds)).collect();
                for rules in [SHADOW, NEAREST] {
                    for nearest_first in [false, true] {
                        let mut g = Numbers(address);
                        let (mut plain, mut guided_turns, mut candidates, mut box_candidates, mut rays, mut differ) = (0u64, 0u64, 0u64, 0u64, 0u64, 0);
                        let mut deepest = 0;
                        for _ in 0..500 {
                            let r = ray(&mut g, &t.header.bounds, 10_000.0);
                            let Ok((a, turns)) = walk(&dump.pages, address, &r, rules) else { continue };
                            let mut guided = Guided::new(&dump.pages, &table, Brute::new(&t, &bottoms, nearest_first).with_boxes(&boxes));
                            let Ok((b, led)) = walk_with(&dump.pages, &mut guided, address, &r, rules) else { continue };
                            rays += 1;
                            differ += u32::from(a.map(|h| h.t) != b.map(|h| h.t) && !rules.first_hit || a.is_some() != b.is_some());
                            plain += u64::from(turns.total());
                            guided_turns += u64::from(led.total());
                            candidates += u64::from(guided.candidates);
                            box_candidates += u64::from(guided.boxes);
                            deepest = deepest.max(turns.deepest);
                        }
                        let per = |x: u64| x as f64 / rays.max(1) as f64;
                        eprintln!(
                            "  {name}, {rules:?}, candidates {}: {rays} rays, turns a ray {:.1} today (the stack at most {deepest} deep), {:.1} guided, \
                             {:.2} candidates a ray ({:.2} of them boxes), {differ} rays end differently",
                            if nearest_first { "nearest first" } else { "in the structures' order" },
                            per(plain),
                            per(guided_turns),
                            per(candidates),
                            per(box_candidates)
                        );
                    }
                }
            }
        }
    }

    /// A bottom level whose builder scaled and moved it
    /// (the console's "minimise memory" builds): the nodes hold the
    /// vertices in that space, the decoder gives them back in the object's.
    #[test]
    fn a_bottom_level_in_a_space_of_its_own_decodes_to_object_space() {
        let local = [
            Piece { a: [0.0, 0.0, 0.0], b: [0.0, 1.0, 0.0], c: [1.0, 0.0, 0.0], d: Some([1.0, 1.0, 0.0]), geometry: 5, opaque: true },
            Piece { a: [2.0, 0.0, 1.0], b: [2.0, 1.0, 1.0], c: [3.0, 0.0, 1.0], d: None, geometry: 6, opaque: false },
        ];
        let mut words = bottom_words(&local, false);
        let base = 0xD_0001_0000u64;
        let plain = bottom(&Span { base, words: &words }, base).expect("decodes");
        assert!(!plain.header.own_space());
        assert!(plain.triangles.iter().any(|t| t.vertices == [local[0].a, local[0].b, local[0].c]));
        // Object bounds 100..132, -40..-8, -1..3: the centre is (116, -24, 0)
        // (an axis whose bounds hold 0 is not moved). Scales 2^-3, 2^-4, 2^1:
        // the exponent bytes 124, 123, 128; the inverse scales 8, 16, 0.5.
        let bounds = [100.0f32, -40.0, -1.0, 132.0, -8.0, 3.0];
        for k in 0..6 {
            words[4 + k] = bounds[k].to_bits();
        }
        words[15] = SPACE_SCALED | SPACE_MOVED | 124 | 123 << 8 | 128 << 16;
        let b = bottom(&Span { base, words: &words }, base).expect("decodes");
        assert!(b.header.own_space());
        let object = |v: [f32; 3]| [v[0] * 8.0 + 116.0, v[1] * 16.0 - 24.0, v[2] * 0.5];
        let mut found: Vec<&Triangle> = b.triangles.iter().collect();
        found.sort_by_key(|t| t.primitive);
        assert_eq!(found[0].vertices, [local[0].a, local[0].b, local[0].c].map(object));
        assert_eq!(found[1].vertices, [local[0].b, [1.0, 1.0, 0.0], local[0].c].map(object));
        assert_eq!(found[2].vertices, [local[1].a, local[1].b, local[1].c].map(object));
        // Unscaled exponents (what every plain structure has there) are no space of its own.
        words[15] = 0x007F_7F7F;
        assert!(!header(&Span { base, words: &words }, base).unwrap().own_space());
        // Turn 0x20: 45 degrees about z (the steps are 22.5 degrees); undone, (x, y) is ((x - y), (x + y)) / root 2.
        words[15] = 0x2000_0000;
        let h = header(&Span { base, words: &words }, base).unwrap();
        let v = h.to_object([1.0, 2.0, 3.0]);
        assert!((v[0] + 0.707_106_8).abs() < 1e-6 && (v[1] - 2.121_320_3).abs() < 1e-6 && v[2] == 3.0, "{v:?}");
    }

    /// The console's own instance flags do not reach the host's build.
    #[test]
    fn a_host_instance_carries_only_the_flags_the_host_knows() {
        let i = Instance {
            matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            id: 7,
            mask: 0xDA,
            contribution: 3,
            flags: 0xC5,
            bottom: 0xD_0001_0000,
            root: 0x25,
            node: 0xE_0000_0100,
            pointer: 0x86,
            to_object: ([0.0; 3], None),
        };
        let w = host_instance_words(&i, Hosted::Triangles { list: 9, structure: 0x1234_5600 }).expect("a host instance");
        assert_eq!(w[13] >> 24, 0x05);
        assert_eq!(w[12] >> 24, 0xDA);
        assert_eq!(HOST_FLAGS, 0x0F);
    }

}
