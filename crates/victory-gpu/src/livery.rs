//! The livery probe (`VICTORY_LIVERY_PROBE=0`: off): about every five seconds,
//! for the vehicles one frame draws with the title's paint shaders that have
//! decals (`PS_VehicleLiveryOn*`), what decides their body colour, read from
//! title memory as the shader reads it, and what the host shows in its place.
//! An instrument: nothing a draw reads is changed.
//!
//! Why: vehicles drawn by those shaders come out flat lime green on this
//! layer (runs 416 to 430, and upstream's issue 218). The green is no
//! arithmetic of the shader: it is the colour of a paint (three bytes of a
//! paint's record, or a row of the paint ramp picture when those are zero),
//! picked by a number a mask texture gives. Which input is wrong was never
//! measured: the frame captures hold no sampled texture and no record. The
//! lines below name it during a normal drive.
//!
//! What the shader reads (from the native code of
//! `PS_VehicleLiveryOnDetailMapsOffDeferred_NoDeform` 4b3d2c8f7a15e7a1; the
//! 24 programs with decals read the same records, their per-draw buffers at
//! other places of table 1, found here by their element sizes):
//! - word 5 of the draw's own 68-byte record names its entity; the high half
//!   of the word at 36 of an entity's 48 bytes its vehicle; of a vehicle's
//!   1,580 bytes the word at 936 is its first decal (0: none), the float at
//!   212 the level bias of the mask samples, the 8 words at 876 zones off;
//! - a vertex's zone names up to two decals ([`zone_decals`]);
//! - a decal is 64 bytes: byte 0 and 1 of word 0 pick the texture
//!   coordinates, words 1 to 3 hold a transform in halves, the words at 16,
//!   20, 24, 28 are sets of the 16 paints, the word at 32 the first paint,
//!   36 the heap slot of a colour picture, 40 and 44 of two masks, 48 to 56
//!   their scales;
//! - a mask's red is a distance, its blue times 16 a paint of the 16
//!   ([`mask_reading`]); a paint is three 76-byte records (colour bytes at
//!   16, ramp row and slice at 20) and one 100-byte record.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;

/// Whether the probe runs (`VICTORY_LIVERY_PROBE=0`: no).
fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LIVERY_PROBE").is_none_or(|v| v != "0"))
}

/// Milliseconds between two frames looked at.
const EVERY_MS: u64 = 5000;
/// Vehicles looked at in one frame (as many again of those told lately),
/// decals of one vehicle.
const VEHICLES: usize = 6;
const DECALS: usize = 4;
/// A vehicle whose lines are the same as its last ones is told again after
/// this long.
const AGAIN_MS: u64 = 60_000;

/// The frame being looked at and when it began; the entities seen in it
/// and how many of them were looked at, those told lately and the others
/// (so that a street of vehicles is gone through, six new ones a frame);
/// per entity the hash of its last lines and when they were printed.
#[derive(Default)]
struct Window {
    frame: u64,
    at: u64,
    seen: Vec<u32>,
    looked: [usize; 2],
    told: std::collections::HashMap<u32, (u64, u64)>,
    draws: u32,
    reports: u32,
}

static WINDOW: Mutex<Option<Window>> = Mutex::new(None);
/// The frame the window last refused or took (a draw of another frame asks
/// the clock again), and whether it took it.
static ASKED: AtomicU64 = AtomicU64::new(u64::MAX);
static TAKEN: AtomicU64 = AtomicU64::new(u64::MAX);

/// Whether draws of `frame` may be looked at: one frame in 64 asks the
/// clock, and one every [`EVERY_MS`] is taken. (An integer test for the
/// other draws.)
pub(crate) fn due(frame: u64) -> bool {
    if frame & 63 != 0 || !on() {
        return false;
    }
    if TAKEN.load(Ordering::Relaxed) == frame {
        return true;
    }
    if ASKED.swap(frame, Ordering::Relaxed) == frame {
        return false;
    }
    let now = crate::host::now_ms();
    let mut window = WINDOW.lock().unwrap_or_else(|e| e.into_inner());
    let w = window.get_or_insert_with(Window::default);
    if w.at != 0 && now.saturating_sub(w.at) < EVERY_MS {
        return false;
    }
    if w.draws > 0 {
        tracing::info!(
            frame = w.frame,
            draws = w.draws,
            vehicles = w.seen.len(),
            reports = w.reports,
            "livery probe: the frame looked at last (vehicles not reported had the same lines as before)"
        );
    }
    w.frame = frame;
    w.at = now.max(1);
    w.seen.clear();
    w.looked = [0; 2];
    w.draws = 0;
    w.reports = 0;
    w.told
        .retain(|_, (_, at)| now.saturating_sub(*at) < 10 * AGAIN_MS);
    TAKEN.store(frame, Ordering::Relaxed);
    true
}

/// Where a decal shader's buffers are in its tables: (root parameter, dword).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Places {
    pub(crate) paints: (u8, u32),
    pub(crate) paints_more: Option<(u8, u32)>,
    pub(crate) decals: (u8, u32),
    pub(crate) vehicles: (u8, u32),
    pub(crate) entities: (u8, u32),
    pub(crate) global: Option<(u8, u32)>,
    pub(crate) own: (u8, u32),
    pub(crate) zones: Option<(u8, u32)>,
    pub(crate) first_paint: Option<(u8, u32)>,
}

/// The places of a pixel shader's decal buffers, by their element sizes (the
/// title's layout: the world's buffers in table 0, the draw's in table 1);
/// `None` for a shader without decals. (Measured on the title's 55
/// `PS_VehicleLiveryOn*` programs and one `LiveryOff`: exactly the 24 that
/// sample the masks have all of these.)
pub(crate) fn places(used: &[UsedResource]) -> Option<Places> {
    let find = |slot: u8, stride: u32| {
        used.iter()
            .find(|u| {
                u.root == RootType::Table
                    && u.kind == ResourceType::Srv
                    && u.slot == slot
                    && u.stride == stride
            })
            .map(|u| (u.slot, u.table_offset_dwords))
    };
    // The decal code alone reads a gradient's origin (12 bytes) from the
    // draw's table: the paint shaders without it read the other buffers too
    // (the 64-byte records hold more than the decals).
    find(1, 12)?;
    Some(Places {
        decals: find(0, 64)?,
        paints: find(0, 76)?,
        vehicles: find(0, 1580)?,
        entities: find(0, 48)?,
        own: find(1, 68)?,
        paints_more: find(0, 100),
        global: find(0, 1),
        zones: find(1, 16),
        first_paint: find(1, 4),
    })
}

/// Whether the pixel shader whose resources are `used` has decals; `known`
/// keeps the answer (0 not asked, 1 no, 2 yes).
pub(crate) fn reads_decals(used: &[UsedResource], known: &AtomicU8) -> bool {
    match known.load(Ordering::Relaxed) {
        0 => {
            let yes = places(used).is_some();
            known.store(if yes { 2 } else { 1 }, Ordering::Relaxed);
            yes
        }
        n => n == 2,
    }
}

/// The decals the zones of a vehicle name: `zones` is its table of 16 bytes
/// a zone, whose word at 12 holds two decal numbers, each plus one, in its
/// two low bytes (0: none). Each number once, in the order met, `most` at
/// most.
pub(crate) fn zone_decals(zones: &[u8], most: usize) -> Vec<u32> {
    let mut numbers = Vec::new();
    for zone in zones.chunks_exact(16) {
        let word = u32::from_le_bytes([zone[12], zone[13], zone[14], zone[15]]);
        for byte in [word & 0xFF, word >> 8 & 0xFF] {
            if byte != 0 && !numbers.contains(&(byte - 1)) && numbers.len() < most {
                numbers.push(byte - 1);
            }
        }
    }
    numbers
}

/// What the decal code makes of a mask texel (`red`, `blue`) for a decal
/// whose paint sets are the words at 16, 20 and 28 (`sets`), with or
/// without a colour picture: the paint of the 16 it picks, how much of the
/// surface that paint covers, and how much the picture does.
///
/// The shader (lines 0x2684 to 0x2740 of the program named in the module's
/// text, the decal a vertex names second, with its mask at 44): paint =
/// truncate(16 x blue); with the paint's bit in the set at 20 the cover is
/// saturate(2.5 x (red - 0.1)); else with bit 16 + paint in the set at 28
/// it is saturate(2.084 x (red - 0.52)); else the picture's, which is that
/// second value where the set at 16 holds the paint and a picture is there.
/// The decal named first (mask at 40, lines 0x297c to 0x2a14) has no set at
/// 28 and takes the other decal's cut-outs off; for one texel value over the
/// whole mask, as the grey stand-in gives, both come to the same.
pub(crate) fn mask_reading(red: f32, blue: f32, sets: [u32; 3], picture: bool) -> (u32, f32, f32) {
    let [set16, set20, set28] = sets;
    let paint = (16.0 * blue).max(0.0) as u32;
    let bit = 1u32 << (paint & 31);
    let high = 1u32 << ((paint + 16) & 31);
    let soft = (2.5 * (red - 0.1)).clamp(0.0, 1.0);
    let hard = (2.084 * (red - 0.52)).clamp(0.0, 1.0);
    let by_picture = if picture && set16 & bit != 0 {
        hard
    } else {
        0.0
    };
    let cover = if set20 & bit != 0 {
        soft
    } else if set28 & high != 0 {
        hard
    } else {
        by_picture
    };
    (paint, cover, by_picture)
}

/// The texel of the layer's 1x1 stand-in texture (0x80 in every channel).
pub(crate) const GREY: f32 = 128.0 / 255.0;

/// A paint's colour word (bytes red, green, blue from the low one) in a few
/// letters: `none` for no colour (the ramp row decides), `GREEN` for the
/// green the fault shows (green above 200, red and blue under 48).
pub(crate) fn colour_word(word: u32) -> String {
    let [r, g, b, _] = word.to_le_bytes();
    let mark = if word & 0x00FF_FFFF == 0 {
        " none"
    } else if g >= 200 && r < 48 && b < 48 {
        " GREEN"
    } else {
        ""
    };
    format!("{r:02x}{g:02x}{b:02x}{mark}")
}

/// A buffer of the draw: its title memory, the view it is read through, and
/// the host copy that view is made on (the one the draw's own binding found
/// a moment ago), once asked.
struct Buffer {
    range: (u64, u64),
    view: View,
    copy: std::cell::OnceCell<Option<std::sync::Arc<crate::shadow::Shadow>>>,
}

impl Buffer {
    fn of(view: Option<View>) -> Option<Buffer> {
        let view = view?;
        let range = crate::descriptor::host::buffer_range(&view)?;
        Some(Buffer {
            range,
            view,
            copy: std::cell::OnceCell::new(),
        })
    }

    /// `len` bytes at `at` of the title's memory.
    fn bytes(&self, at: u64, len: usize) -> Option<Vec<u8>> {
        let mut bytes = vec![0u8; len];
        let there = at + len as u64 <= self.range.1
            && crate::host::read_into((self.range.0 + at) as *const u8, &mut bytes);
        there.then_some(bytes)
    }

    fn word(&self, at: u64) -> Option<u32> {
        self.bytes(at, 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// Whether the host's copy holds `title` at `at`: "same", "DIFFERS"
    /// with the copy's bytes as words, "in place" (shaders read the title's
    /// own memory), "no copy".
    fn copy(&self, at: u64, title: &[u8]) -> String {
        let copy = self.copy.get_or_init(|| {
            crate::descriptor::host::buffer_copy(&self.view).map(|(shadow, _, _)| shadow)
        });
        let Some(shadow) = copy else {
            return "no copy".into();
        };
        if shadow.in_place() {
            return "in place".into();
        }
        match shadow.held_at(self.range.0 + at, title.len()) {
            None => "not in its copy".into(),
            Some(held) if held == title => "same".into(),
            Some(held) => {
                let words: Vec<String> = held
                    .chunks_exact(4)
                    .take(16)
                    .map(|w| format!("{:08x}", u32::from_le_bytes([w[0], w[1], w[2], w[3]])))
                    .collect();
                format!("DIFFERS host=[{}]", words.join(" "))
            }
        }
    }
}

/// One texture slot a record names: what the title's heap holds there and
/// what the host slot shows.
fn slot_text(heap: &crate::descriptor::Heap, index: u32) -> (String, &'static str) {
    if index == 0 {
        return ("slot 0 (none)".into(), "none");
    }
    let handle = heap.handle(index);
    let title = handle.and_then(crate::descriptor::view);
    let title_text = match &title {
        None => "no view".to_owned(),
        Some(View::Placed(false, r, d)) if r[0] != 1 => {
            let address = u64::from(d[10]) | u64::from(d[11]) << 32;
            let first = crate::resource::first_mip(d);
            let len = crate::resource::placed_bytes_from(r, first).unwrap_or(0);
            let mapped = crate::texture::mapped_bytes(address as *const u8, len as usize);
            // (The view's channel mapping is 5768 when it is the plain one;
            // its lowest level of detail is a float.)
            format!(
                "placed {}x{} f{} levels {} view f{} mapping {} from level {} count {} min_lod {} at {address:#x} mapped {mapped} of {len}",
                u64::from(r[4]) | u64::from(r[5]) << 32,
                r[6],
                r[8],
                r[7] >> 16,
                d[0],
                d[2],
                d[4],
                d[5] as i32,
                f32::from_bits(d[7])
            )
        }
        Some(View::Shader(r, d)) => format!(
            "object {r:#x} view (format, dimension, mapping, first level, count) {:?}",
            d.map(|d| (d[0], d[1], d[2], d[4], d[5] as i32))
        ),
        Some(other) => format!(
            "NOT A TEXTURE {}",
            format!("{other:?}").chars().take(60).collect::<String>()
        ),
    };
    let (shown, kind) = match (heap.host(), handle) {
        (Some(host), Some(handle)) => host.slot_probe(index, handle),
        _ => ("no host slot".to_owned(), "none"),
    };
    let queue = heap.host().map_or_else(String::new, |h| {
        h.texture_slot_state(index, crate::frame::current())
    });
    (
        format!("slot {index} title=[{title_text}] host=[{shown}] queue=[{queue}]"),
        kind,
    )
}

/// A probe line without its `queue=[...]` parts.
fn without_queue(line: &str) -> String {
    let mut text = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(at) = rest.find("queue=[") {
        text.push_str(&rest[..at]);
        rest = rest[at..].find(']').map_or("", |end| &rest[at + end + 1..]);
    }
    text.push_str(rest);
    text
}

/// The probe of one draw of a decal shader in a frame [`due`] took: finds
/// the vehicle, and prints its lines unless they are its last ones again.
/// `view` gives the view at a place of the draw's tables, `record` the byte
/// offset its root constants name in the world's material buffer.
pub(crate) fn probe(
    frame: u64,
    pixel: &str,
    heap: &crate::descriptor::Heap,
    places: &Places,
    view: &dyn Fn((u8, u32)) -> Option<View>,
    record: Option<u32>,
) {
    let buffer = |place: (u8, u32)| Buffer::of(view(place));
    let (Some(own), Some(entities), Some(vehicles), Some(decals), Some(paints)) = (
        buffer(places.own),
        buffer(places.entities),
        buffer(places.vehicles),
        buffer(places.decals),
        buffer(places.paints),
    ) else {
        return;
    };
    let Some(entity) = own.word(20) else {
        return;
    };
    // One report a vehicle a frame; a bounded number of vehicles.
    {
        let mut window = WINDOW.lock().unwrap_or_else(|e| e.into_inner());
        let Some(w) = window.as_mut().filter(|w| w.frame == frame) else {
            return;
        };
        w.draws += 1;
        if w.seen.contains(&entity) || w.seen.len() >= 512 {
            return;
        }
        w.seen.push(entity);
        let now = crate::host::now_ms();
        let lately = w
            .told
            .get(&entity)
            .is_some_and(|&(_, at)| now.saturating_sub(at) < AGAIN_MS);
        let looked = &mut w.looked[usize::from(lately)];
        if *looked >= VEHICLES {
            return;
        }
        *looked += 1;
    }
    let mut lines: Vec<String> = Vec::new();
    let Some(vehicle) = entities.word(u64::from(entity) * 48 + 36).map(|w| w >> 16) else {
        tracing::info!(
            frame,
            pixel,
            entity,
            "livery probe: the entity is not in the entities' buffer"
        );
        return;
    };
    let at = u64::from(vehicle) * 1580;
    let first = vehicles.word(at + 936).unwrap_or(0);
    let bias = vehicles.word(at + 212).unwrap_or(0);
    let off: Vec<u32> = (0..8)
        .map(|k| vehicles.word(at + 876 + 4 * k).unwrap_or(0))
        .collect();
    // Whether the host's copies hold what the title's memory does, of the
    // records read above: the draw's own, the entity's, the vehicle's words
    // at 200 to 215 (the bias among them) and at 876 to 939.
    let same = |b: &Buffer, at: u64, len: usize| -> String {
        b.bytes(at, len)
            .map_or_else(|| "unreadable".into(), |title| b.copy(at, &title))
    };
    let own_copy = same(&own, 0, 68);
    let entity_copy = same(&entities, u64::from(entity) * 48, 48);
    let vehicle_copy = match (same(&vehicles, at + 200, 16), same(&vehicles, at + 876, 64)) {
        (a, b) if a == b => a,
        (a, b) => format!("at 200: {a}, at 876: {b}"),
    };
    let zones = places.zones.and_then(buffer);
    let numbers = match &zones {
        Some(z) => match z.bytes(0, z.range.1.min(4096) as usize & !15) {
            Some(bytes) => zone_decals(&bytes, 24),
            None => Vec::new(),
        },
        None => Vec::new(),
    };
    let zones_copy = match &zones {
        Some(z) => z.bytes(0, z.range.1.min(4096) as usize & !15).map_or_else(
            || "unreadable".into(),
            |b| z.copy(0, &b).chars().take(12).collect::<String>(),
        ),
        None => "no zones buffer".into(),
    };
    let own_paint = places
        .first_paint
        .and_then(buffer)
        .and_then(|b| b.word(1024));
    // The paint ramp picture (an array: a paint without colour bytes takes
    // its colour from a row of it): its slot is the word at 32 of the
    // material record in 15 of the 24 decal programs, the word at 24 in the
    // other 9.
    let global = places.global.and_then(buffer);
    let ramp = match (record, &global) {
        (Some(record), Some(g)) => {
            let word = |at: u64| {
                g.word(u64::from(record) + at)
                    .map_or_else(|| "unreadable".into(), |slot| slot_text(heap, slot).0)
            };
            Some(format!(
                "material word 32: {} / word 24: {}",
                word(32),
                word(24)
            ))
        }
        _ => None,
    };
    lines.push(format!(
        "livery probe vehicle: pixel={pixel} entity={entity} vehicle={vehicle} first_decal={first} zone_decals={numbers:?} zones_off={off:x?} \
         mask_bias={} ({bias:#010x}) own_first_paint={own_paint:?} copies: own={own_copy} entity={entity_copy} vehicle={vehicle_copy} zones={zones_copy} | ramp picture, {}",
        f32::from_bits(bias),
        ramp.unwrap_or_else(|| "not found".into()),
    ));
    let paints_more = places.paints_more.and_then(buffer);
    // A paint in a few words: colour, ramp row and slice, the flags of its
    // 100-byte record; and whether the host's copy of its colour word is the
    // title's.
    let paint = |number: u32| -> String {
        let at = u64::from(number) * 3 * 76;
        let Some(bytes) = paints.bytes(at + 16, 8) else {
            return "unreadable".into();
        };
        let colour = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let ramp = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let flags = paints_more
            .as_ref()
            .and_then(|m| m.word(u64::from(number) * 100 + 48));
        let copy = paints.copy(at + 16, &bytes);
        let copy = if copy == "same" || copy == "in place" {
            String::new()
        } else {
            format!(" COPY {copy}")
        };
        format!(
            "{} ramp row {} slice {} flags {}{copy}",
            colour_word(colour),
            ramp & 0xFF,
            ramp >> 8 & 0xFF,
            flags.map_or_else(|| "?".into(), |f| format!("{f:08x}"))
        )
    };
    if first != 0 {
        for &number in numbers.iter().take(DECALS) {
            let at = (u64::from(first) + u64::from(number)) * 64;
            let Some(bytes) = decals.bytes(at, 64) else {
                lines.push(format!("livery probe decal: entity={entity} n={number} record unreadable or past the decals' buffer"));
                continue;
            };
            let w: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
                .collect();
            let copy = decals.copy(at, &bytes);
            let sets = [w[4], w[5], w[7]];
            let (picture, picture_kind) = slot_text(heap, w[9]);
            let (mask_a, a_kind) = slot_text(heap, w[10]);
            let (mask_b, b_kind) = slot_text(heap, w[11]);
            // The paints its sets hold, and paint 8, which a grey mask picks.
            let used = w[4] | w[5] | w[6] | w[6] >> 16 | w[7] >> 16;
            let mut paints_text = String::new();
            for k in (0..16u32).filter(|k| used >> k & 1 != 0 || *k == 8) {
                paints_text.push_str(&format!(" [{k}: {}]", paint(w[8].wrapping_add(k))));
            }
            // What a mask that is not its own texture makes of this decal.
            let mut reads = String::new();
            for (name, kind) in [("mask_a", a_kind), ("mask_b", b_kind)] {
                match kind {
                    "stand-in" => {
                        let (k, cover, _) = mask_reading(GREY, GREY, sets, w[9] != 0);
                        reads.push_str(&format!(
                            " {name} ON THE GREY STAND-IN: the shader takes paint {k} with cover {cover:.2} everywhere = {};",
                            paint(w[8].wrapping_add(k))
                        ));
                    }
                    "null" => reads.push_str(&format!(" {name} null: reads zeros, no cover;")),
                    "other" => {
                        reads.push_str(&format!(" {name} SHOWS ANOTHER VIEW than the title's;"))
                    }
                    _ => {}
                }
            }
            if picture_kind == "stand-in" || picture_kind == "other" {
                reads.push_str(" picture not its own;");
            }
            lines.push(format!(
                "livery probe decal: entity={entity} n={number} record={:08x?} copy={copy} uv_sets=({},{}) scales=({},{},{}) \
                 sets16/20/24/28=({:08x},{:08x},{:08x},{:08x}) first_paint={} | picture {picture} | mask_a {mask_a} | mask_b {mask_b} | \
                 paints{paints_text} | reads:{}",
                &w[..4],
                w[0] & 0xFF,
                w[0] >> 8 & 0xFF,
                f32::from_bits(w[12]),
                f32::from_bits(w[13]),
                f32::from_bits(w[14]),
                w[4],
                w[5],
                w[6],
                w[7],
                w[8],
                if reads.is_empty() { " every mask is its own texture" } else { reads.as_str() },
            ));
        }
    }
    // The same lines as this vehicle's last ones: not again for a while.
    // (But for what the texture queue says: its frame numbers move while a
    // slot waits for its looks again, and nothing else does.)
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    for line in &lines {
        std::hash::Hash::hash(&without_queue(line), &mut hash);
    }
    let hash = std::hash::Hasher::finish(&hash);
    let now = crate::host::now_ms();
    {
        let mut window = WINDOW.lock().unwrap_or_else(|e| e.into_inner());
        let Some(w) = window.as_mut() else {
            return;
        };
        match w.told.get(&entity) {
            Some(&(was, at)) if was == hash && now.saturating_sub(at) < AGAIN_MS => return,
            _ => {}
        }
        w.told.insert(entity, (hash, now));
        w.reports += 1;
    }
    for line in lines {
        tracing::info!(frame, "{line}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decals a vehicle's zones name: the two low bytes of the word at
    /// 12 of each 16-byte zone, each minus one, 0 for none.
    #[test]
    fn zones_name_their_decals_once() {
        let zone = |word: u32| {
            let mut z = vec![0xEEu8; 16];
            z[12..].copy_from_slice(&word.to_le_bytes());
            z
        };
        let table: Vec<u8> = [0, 0x0000_0003, 0x0000_0102, 0x0000_0300, 0xFFFF_0005]
            .into_iter()
            .flat_map(zone)
            .collect();
        assert_eq!(zone_decals(&table, 24), vec![2, 1, 0, 4]);
        assert_eq!(zone_decals(&table, 2), vec![2, 1]);
        assert_eq!(zone_decals(&table[..31], 24), Vec::<u32>::new());
    }

    /// What the decal code makes of a mask texel. The layer's grey stand-in
    /// (0x80) is "inside the decal, paint 8" for a decal whose set at 20
    /// holds paint 8, and nothing for one that holds it only in the sets
    /// with the 0.52 threshold; a null view (zeros) is no cover at all; a
    /// real texel inside and outside.
    #[test]
    fn a_grey_mask_is_paint_eight_everywhere() {
        // Grey, paint 8 in the set at 20: full cover.
        let (paint, cover, by_picture) = mask_reading(GREY, GREY, [0, 1 << 8, 0], false);
        assert_eq!(paint, 8);
        assert_eq!(cover, 1.0);
        assert_eq!(by_picture, 0.0);
        // Grey, paint 8 only in the set at 28 (bit 24) or at 16 with a picture: 0.502 is under 0.52.
        assert_eq!(mask_reading(GREY, GREY, [0, 0, 1 << 24], false).1, 0.0);
        assert_eq!(
            mask_reading(GREY, GREY, [1 << 8, 0, 0], true),
            (8, 0.0, 0.0)
        );
        // Grey, paint 8 in no set.
        assert_eq!(
            mask_reading(GREY, GREY, [0xFF, 0xFF, 0x00FF_0000], true).1,
            0.0
        );
        // A null view: zeros, paint 0, no cover even with every set full.
        assert_eq!(mask_reading(0.0, 0.0, [u32::MAX; 3], true), (0, 0.0, 0.0));
        // A real texel well inside, paint 3: full cover by either threshold.
        assert_eq!(
            mask_reading(1.0, 3.2 / 16.0, [0, 1 << 3, 0], false),
            (3, 1.0, 0.0)
        );
        assert_eq!(
            mask_reading(1.0, 3.2 / 16.0, [1 << 3, 0, 0], true),
            (3, 1.0, 1.0)
        );
        // The edge of the soft threshold: red 0.3 is half covered.
        let (_, cover, _) = mask_reading(0.3, 0.0, [0, 1, 0], false);
        assert!((cover - 0.5).abs() < 1e-6);
        // Blue 1.0 is paint 16: bit 16 of the sets (as the shader's shift does).
        assert_eq!(
            mask_reading(1.0, 1.0, [0, 1 << 16, 0], false),
            (16, 1.0, 0.0)
        );
    }

    /// What a host copy holds is read at the title's address, inside the copy
    /// only: the probe compares it with the title's own bytes.
    #[test]
    fn a_copys_bytes_are_read_at_the_titles_address() {
        let mut copy: Vec<u8> = (0..=255).collect();
        let shadow = crate::shadow::Shadow::unbacked(0x10_0000, copy.as_mut_ptr(), 256, false);
        assert_eq!(
            shadow.held_at(0x10_0000 + 16, 4),
            Some(vec![16, 17, 18, 19])
        );
        assert_eq!(
            shadow.held_at(0x10_0000 + 252, 4),
            Some(vec![252, 253, 254, 255])
        );
        assert_eq!(shadow.held_at(0x10_0000 + 254, 4), None);
        assert_eq!(shadow.held_at(0x0F_FFFF, 4), None);
        assert!(!shadow.in_place());
    }

    #[test]
    fn a_line_is_compared_without_the_queues_words() {
        let a =
            "x slot 5 title=[t] host=[h] queue=[queued due=@12 written@3] | y queue=[not queued] z";
        let b = "x slot 5 title=[t] host=[h] queue=[queued due=@99 written@3] | y queue=[] z";
        assert_eq!(without_queue(a), "x slot 5 title=[t] host=[h]  | y  z");
        assert_eq!(without_queue(a), without_queue(b));
        assert_eq!(without_queue("no such part"), "no such part");
        assert_eq!(without_queue("cut queue=[open"), "cut ");
    }

    #[test]
    fn a_paints_colour_in_words() {
        assert_eq!(colour_word(0x0000_FF00), "00ff00 GREEN");
        assert_eq!(colour_word(0xFF0C_FB0D), "0dfb0c GREEN");
        assert_eq!(colour_word(0x0000_0000), "000000 none");
        assert_eq!(colour_word(0xFF00_0000), "000000 none");
        assert_eq!(colour_word(0x0020_40C0), "c04020");
    }

    /// A decal shader is known by the element sizes of its buffers, wherever
    /// its per-draw buffers lie in table 1; a paint shader without decals is
    /// none.
    #[test]
    fn a_decal_shader_is_known_by_its_buffers() {
        let srv = |slot: u8, dwords: u32, stride: u32| UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: dwords,
            lower_bound: 0,
            stride,
            dwords: 4,
            user_data: 8 + slot,
            slot,
        };
        // 4b3d2c8f7a15e7a1 (run 421's lime boat).
        let a = [
            srv(0, 336, 76),
            srv(0, 368, 64),
            srv(0, 640, 1580),
            srv(0, 680, 100),
            srv(0, 688, 1),
            srv(0, 712, 48),
            srv(1, 16, 68),
            srv(1, 24, 16),
            srv(1, 32, 12),
            srv(1, 64, 4),
        ];
        let p = places(&a).unwrap();
        assert_eq!(
            (p.decals, p.paints, p.own, p.zones, p.first_paint),
            ((0, 368), (0, 336), (1, 16), Some((1, 24)), Some((1, 64)))
        );
        // c421da89dcdc9cf2: its own record at dword 8, the zones at 16.
        let b = [
            srv(0, 336, 76),
            srv(0, 368, 64),
            srv(0, 640, 1580),
            srv(0, 680, 100),
            srv(0, 688, 1),
            srv(0, 712, 48),
            srv(1, 8, 68),
            srv(1, 16, 16),
            srv(1, 24, 12),
            srv(1, 56, 4),
        ];
        let p = places(&b).unwrap();
        assert_eq!(
            (p.own, p.zones, p.first_paint),
            ((1, 8), Some((1, 16)), Some((1, 56)))
        );
        // A paint shader without decals (19acd330dacc8f32): it reads the
        // 64-byte records and the zones too, not the gradient's origin.
        let c = [
            srv(0, 336, 76),
            srv(0, 368, 64),
            srv(0, 640, 1580),
            srv(0, 680, 100),
            srv(0, 688, 1),
            srv(0, 712, 48),
            srv(1, 16, 68),
            srv(1, 24, 16),
            srv(1, 48, 288),
            srv(1, 64, 4),
        ];
        assert_eq!(places(&c), None);
        let known = AtomicU8::new(0);
        assert!(reads_decals(&a, &known));
        assert_eq!(known.load(Ordering::Relaxed), 2);
        assert!(reads_decals(&c, &known), "the answer is kept");
        let known = AtomicU8::new(0);
        assert!(!reads_decals(&c, &known));
    }
}
