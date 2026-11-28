//! The host's ray-tracing hardware against the title's structures, on the
//! software adapter and nowhere else:
//!
//! - host structures built from structures decoded out of title memory
//!   ([`crate::rays`]) give an inline ray query the hits the title's own
//!   walk finds ([`crate::rays::walk`]), ray for ray;
//! - a walking program of the title, translated with and without the guided
//!   node test (`VICTORY_RAY_GUIDE`), writes the same output over the same
//!   structures and rays, in fewer turns of its loop; also where the host
//!   has structures of some bottom levels only, or of none (box stand-ins),
//!   and where a page of title memory has no copy yet;
//! - the layer's pass that writes a host top level's instances on the card
//!   ([`crate::rays::TOP_PASS`]) writes what the CPU would
//!   ([`crate::rays::top_pass_words`]), and the host builds its top level
//!   from that buffer where it lies.
//!
//! The structures are made up here in the title's format
//! (`rays::tests::scene`). The first test needs a PC `dxc.exe` for its own
//! ray-query program (`VICTORY_PC_DXC`, else the newest Windows SDK's, as
//! `diff_tests.rs`). The second needs a container of the title, which is
//! not ours to keep in the tree: `VICTORY_RAY_WALKER=<file>` names the
//! container of f607780aa8e502d0 (`CS_IrradianceCache_ShadowTrace_WithFar`,
//! saved from the title's shader archive by `translate` with
//! `VICTORY_DUMP_MATCH`). Both are skipped without what they need.
//!
//! - the layer's own code for the host's structures (`victory-gpu`
//!   `src/guide/core.rs`, taken in here by path: it names nothing of the
//!   layer) copies the title's static bottom levels back, builds them into
//!   its pool, makes a top level before a walk and keeps the two tables; the
//!   title's walker, guided by that alone, writes what it writes without
//!   the guide.
//!
//! Every device here is the software adapter: [`software`] refuses any
//! other, so a test cannot fall back to the card.

use std::collections::HashMap;
use std::path::PathBuf;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

use super::host::Host;
use super::lower::intersect::{TEST_GUIDE, TEST_TURNS};
use super::shader::{LANES_ARE_THREADS, MEMORY_FAULTS, MEMORY_PAGES, USER_MEMORY_BASE, USER_VIEW_ADDRESS};
use super::{HostShader, SLOT_FAULTS, SLOT_GUIDE, SLOT_MEMORY, UserData, validator};
use crate::bvh::Ray;
use crate::native::{HwStage, RootType};
use crate::rays::tests::{Numbers, Scene, scene};
use crate::rays::{self, Bottom, GuidedTop, Hosted, Instance, Pages, Rules, Top};

/// The layer's side of the guide: the same file `victory-gpu` builds.
#[path = "../../../victory-gpu/src/guide/core.rs"]
mod guide_core;

fn err(e: windows::core::Error) -> String {
    e.message().to_string()
}

/// The software adapter with inline ray queries.
struct Card {
    host: Host,
    device: ID3D12Device5,
    queue: ID3D12CommandQueue,
}

/// The software adapter, or none when it has no ray queries. Never the
/// card: a device on anything else fails the test.
fn software() -> Option<Card> {
    let host = match Host::new(true) {
        Ok(host) => host,
        Err(e) => {
            eprintln!("no SM 6.6 software adapter ({e}); skipped");
            return None;
        }
    };
    // SAFETY: plain D3D12/DXGI calls with valid out pointers.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).expect("a factory");
        let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(host.device.GetAdapterLuid()).expect("the device's adapter");
        let desc = adapter.GetDesc1().expect("its description");
        assert!(
            desc.VendorId == 0x1414 && desc.DeviceId == 0x8c && desc.Flags & 2 != 0,
            "these tests run on the software adapter alone: vendor {:#x} device {:#x}",
            desc.VendorId,
            desc.DeviceId
        );
        let mut o5 = D3D12_FEATURE_DATA_D3D12_OPTIONS5::default();
        let asked = host.device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, (&raw mut o5).cast(), size_of_val(&o5) as u32);
        if asked.is_err() || o5.RaytracingTier.0 < D3D12_RAYTRACING_TIER_1_1.0 {
            eprintln!("the software adapter has no inline ray queries (tier {}); skipped", o5.RaytracingTier.0);
            return None;
        }
        let device: ID3D12Device5 = host.device.cast().expect("a device of ray tracing");
        let queue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })
            .expect("a queue");
        Some(Card { host, device, queue })
    }
}

/// A view in the heap for [`Card::run`].
enum Bound<'a> {
    /// A null raw view (the memory copies' first slot).
    Null,
    /// A raw buffer, read only.
    Raw(&'a [u32]),
    /// A raw buffer with these contents, written and read back.
    RawRw(&'a [u32]),
    /// A typed buffer of these dwords: the format and its bytes.
    Typed(&'a [u32], DXGI_FORMAT, u32),
    /// A typed buffer of so many elements of zeros, written and read back.
    TypedRw(u32, DXGI_FORMAT, u32),
    /// A constant buffer.
    Cbv(&'a [u32]),
    /// A host top-level structure at this address.
    Structure(u64),
}

/// A root view for [`Card::run`].
enum RootView<'a> {
    Srv(&'a [u32]),
    /// A buffer that lies at this address on the adapter.
    SrvAt(u64),
    /// So many dwords of zeros, read back.
    Uav(u32),
}

/// A dispatch: the user data, root views by root parameter, the heap by
/// slot, and the groups along X.
struct Call<'a> {
    constants: Vec<u32>,
    roots: Vec<(u32, RootView<'a>)>,
    heap: Vec<(u32, Bound<'a>)>,
    /// Descriptor tables: the root parameter, the heap slot it starts at,
    /// and how many slots it may reach.
    tables: Vec<(u32, u32, u32)>,
    groups: u32,
}

/// What a dispatch wrote: by heap slot, and by root parameter; and the
/// buffers of the root views it wrote, as they lie on the adapter (left as
/// copy sources).
struct Ran {
    heap: HashMap<u32, Vec<u32>>,
    roots: HashMap<u32, Vec<u32>>,
    kept: HashMap<u32, ID3D12Resource>,
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

impl Card {
    fn buffer(&self, bytes: u64, heap: D3D12_HEAP_TYPE, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: bytes.max(4).next_multiple_of(256),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut r: Option<ID3D12Resource> = None;
        // SAFETY: the descriptions outlive the call.
        unsafe {
            self.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: heap, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    state,
                    None,
                    &mut r,
                )
                .unwrap_or_else(|e| panic!("a buffer of {bytes} bytes: {}", err(e)));
        }
        r.expect("a buffer")
    }

    /// An upload buffer holding `words`.
    fn upload(&self, words: &[u32]) -> ID3D12Resource {
        let b = self.buffer(4 * words.len() as u64, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        // SAFETY: a fresh mapping of at least that many bytes.
        unsafe {
            let mut p = std::ptr::null_mut();
            b.Map(0, None, Some(&mut p)).expect("mapped");
            std::ptr::copy_nonoverlapping(words.as_ptr(), p.cast::<u32>(), words.len());
            b.Unmap(0, None);
        }
        b
    }

    /// Records a command list with `record`, runs it and waits for it.
    fn submit(&self, pso: Option<&ID3D12PipelineState>, record: impl FnOnce(&ID3D12GraphicsCommandList4)) -> Result<(), String> {
        // SAFETY: D3D12 calls; what the list names is kept by the caller
        // until this returns, after the queue is done.
        unsafe {
            let alloc: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).map_err(err)?;
            let list: ID3D12GraphicsCommandList4 =
                self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, pso).map_err(err)?;
            record(&list);
            list.Close().map_err(err)?;
            self.queue.ExecuteCommandLists(&[Some(list.cast().map_err(err)?)]);
            let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).map_err(err)?;
            self.queue.Signal(&fence, 1).map_err(err)?;
            while fence.GetCompletedValue() < 1 {
                if let Err(e) = self.device.GetDeviceRemovedReason() {
                    return Err(format!("device removed: {}", err(e)));
                }
                std::thread::yield_now();
            }
        }
        Ok(())
    }

    /// A host structure of `inputs`, built.
    fn structure(&self, inputs: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS) -> ID3D12Resource {
        let mut info = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO::default();
        // SAFETY: `inputs` points at descriptions the caller keeps.
        unsafe { self.device.GetRaytracingAccelerationStructurePrebuildInfo(&inputs, &mut info) };
        assert!(info.ResultDataMaxSizeInBytes > 0, "the adapter builds no such structure");
        let result = self.buffer(
            info.ResultDataMaxSizeInBytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        );
        let scratch = self.buffer(
            info.ScratchDataSizeInBytes,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COMMON,
        );
        self.submit(None, |list| {
            // SAFETY: live buffers of the sizes the adapter asked for.
            unsafe {
                list.ResourceBarrier(&[transition(&scratch, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
                list.BuildRaytracingAccelerationStructure(
                    &D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC {
                        DestAccelerationStructureData: result.GetGPUVirtualAddress(),
                        Inputs: inputs,
                        SourceAccelerationStructureData: 0,
                        ScratchAccelerationStructureData: scratch.GetGPUVirtualAddress(),
                    },
                    None,
                );
            }
        })
        .expect("the build ran");
        result
    }

    /// The host's bottom level of a decoded one: its triangles in the
    /// decoded order (a host primitive's index is its place in
    /// [`Bottom::triangles`]), one geometry.
    fn bottom(&self, b: &Bottom) -> ID3D12Resource {
        let vertices: Vec<u32> = b.triangles.iter().flat_map(|t| t.vertices).flatten().map(f32::to_bits).collect();
        let buffer = self.upload(&vertices);
        let geometry = D3D12_RAYTRACING_GEOMETRY_DESC {
            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES,
            Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_NONE,
            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                Triangles: D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC {
                    Transform3x4: 0,
                    IndexFormat: DXGI_FORMAT_UNKNOWN,
                    VertexFormat: DXGI_FORMAT_R32G32B32_FLOAT,
                    IndexCount: 0,
                    VertexCount: 3 * b.triangles.len() as u32,
                    IndexBuffer: 0,
                    VertexBuffer: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                        // SAFETY: a live buffer.
                        StartAddress: unsafe { buffer.GetGPUVirtualAddress() },
                        StrideInBytes: 12,
                    },
                },
            },
        };
        self.structure(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: 1,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { pGeometryDescs: &geometry },
        })
    }

    /// The host's top level of these instances: the title's matrix, mask
    /// and flags as they are, the two numbers the guided node test reads
    /// ([`rays::host_instance`]), and the host's bottom level or the
    /// stand-in box ([`rays::host_instance_words`]).
    fn top(&self, instances: &[(Instance, Hosted)]) -> ID3D12Resource {
        let mut words = Vec::with_capacity(16 * instances.len());
        for (i, hosted) in instances {
            words.extend(rays::host_instance_words(i, *hosted).expect("an instance the guide takes"));
        }
        let buffer = self.upload(&words);
        // SAFETY: a live buffer.
        self.top_at(unsafe { buffer.GetGPUVirtualAddress() }, instances.len() as u32)
    }

    /// The host's top level of the `count` instance descriptions at
    /// `descriptions` on the adapter.
    fn top_at(&self, descriptions: u64, count: u32) -> ID3D12Resource {
        self.structure(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: count,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { InstanceDescs: descriptions },
        })
    }

    /// The host's stand-in for a bottom level it has no structure of: one
    /// box primitive, the unit cube ([`rays::STAND_IN`]).
    fn stand_in(&self) -> ID3D12Resource {
        let buffer = self.upload(&rays::STAND_IN.map(f32::to_bits));
        let geometry = D3D12_RAYTRACING_GEOMETRY_DESC {
            Type: D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS,
            Flags: D3D12_RAYTRACING_GEOMETRY_FLAG_NONE,
            Anonymous: D3D12_RAYTRACING_GEOMETRY_DESC_0 {
                AABBs: D3D12_RAYTRACING_GEOMETRY_AABBS_DESC {
                    AABBCount: 1,
                    AABBs: D3D12_GPU_VIRTUAL_ADDRESS_AND_STRIDE {
                        // SAFETY: a live buffer.
                        StartAddress: unsafe { buffer.GetGPUVirtualAddress() },
                        StrideInBytes: 24,
                    },
                },
            },
        };
        self.structure(D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS {
            Type: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL,
            Flags: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE,
            NumDescs: 1,
            DescsLayout: D3D12_ELEMENTS_LAYOUT_ARRAY,
            Anonymous: D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_0 { pGeometryDescs: &geometry },
        })
    }

    /// Runs a compute pipeline over `call` and reads back what it wrote.
    fn run(&self, root: &ID3D12RootSignature, pso: &ID3D12PipelineState, call: &Call) -> Result<Ran, String> {
        let dev = &self.device;
        let default = |bytes: u64, state| self.buffer(bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, state);
        let readback =
            |bytes: u64| self.buffer(bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
        // SAFETY: D3D12 calls with descriptions that outlive them; every
        // resource is kept until the queue is done.
        unsafe {
            let slots = call.heap.iter().map(|h| h.0 + 1).chain(call.tables.iter().map(|t| t.1 + t.2)).max().unwrap_or(1);
            let views: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: slots,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
                .map_err(err)?;
            let step = dev.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            let mut keep: Vec<ID3D12Resource> = Vec::new();
            // What is copied in before the dispatch (to, from, bytes), and
            // read back after it (the key, the buffer, its dwords).
            let mut fills: Vec<(ID3D12Resource, ID3D12Resource, u64)> = Vec::new();
            let mut backs: Vec<((bool, u32), ID3D12Resource, ID3D12Resource, u32)> = Vec::new();
            let srv = |format, elements: u32, flags| D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_SRV { FirstElement: 0, NumElements: elements, StructureByteStride: 0, Flags: flags },
                },
            };
            let uav = |format, elements: u32, flags| D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                    Buffer: D3D12_BUFFER_UAV { NumElements: elements, Flags: flags, ..Default::default() },
                },
            };
            for (slot, bound) in &call.heap {
                let mut at = views.GetCPUDescriptorHandleForHeapStart();
                at.ptr += *slot as usize * step;
                match *bound {
                    Bound::Null => {
                        let desc = uav(DXGI_FORMAT_R32_TYPELESS, 1, D3D12_BUFFER_UAV_FLAG_RAW);
                        dev.CreateUnorderedAccessView(None, None, Some(&desc), at);
                    }
                    Bound::Raw(words) => {
                        let b = self.upload(words);
                        let desc = srv(DXGI_FORMAT_R32_TYPELESS, words.len() as u32, D3D12_BUFFER_SRV_FLAG_RAW);
                        dev.CreateShaderResourceView(&b, Some(&desc), at);
                        keep.push(b);
                    }
                    Bound::Typed(words, format, bytes) => {
                        let b = self.upload(words);
                        let desc = srv(format, 4 * words.len() as u32 / bytes, D3D12_BUFFER_SRV_FLAG_NONE);
                        dev.CreateShaderResourceView(&b, Some(&desc), at);
                        keep.push(b);
                    }
                    Bound::Cbv(words) => {
                        let b = self.upload(words);
                        dev.CreateConstantBufferView(
                            Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC {
                                BufferLocation: b.GetGPUVirtualAddress(),
                                SizeInBytes: (4 * words.len() as u32).next_multiple_of(256),
                            }),
                            at,
                        );
                        keep.push(b);
                    }
                    Bound::RawRw(words) => {
                        let bytes = 4 * words.len() as u64;
                        let b = default(bytes, D3D12_RESOURCE_STATE_COPY_DEST);
                        let desc = uav(DXGI_FORMAT_R32_TYPELESS, words.len() as u32, D3D12_BUFFER_UAV_FLAG_RAW);
                        dev.CreateUnorderedAccessView(&b, None, Some(&desc), at);
                        fills.push((b.clone(), self.upload(words), bytes));
                        backs.push(((false, *slot), b, readback(bytes), words.len() as u32));
                    }
                    Bound::TypedRw(elements, format, bytes) => {
                        let size = u64::from(elements) * u64::from(bytes);
                        let b = default(size, D3D12_RESOURCE_STATE_COPY_DEST);
                        let desc = uav(format, elements, D3D12_BUFFER_UAV_FLAG_NONE);
                        dev.CreateUnorderedAccessView(&b, None, Some(&desc), at);
                        fills.push((b.clone(), self.upload(&vec![0; (size / 4) as usize]), size));
                        backs.push(((false, *slot), b, readback(size), (size / 4) as u32));
                    }
                    Bound::Structure(location) => {
                        let desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                            Format: DXGI_FORMAT_UNKNOWN,
                            ViewDimension: D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE,
                            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                                RaytracingAccelerationStructure: D3D12_RAYTRACING_ACCELERATION_STRUCTURE_SRV { Location: location },
                            },
                        };
                        dev.CreateShaderResourceView(None, Some(&desc), at);
                    }
                }
            }
            let mut root_views = Vec::new();
            for (param, view) in &call.roots {
                match *view {
                    RootView::Srv(words) => {
                        let b = self.upload(words);
                        root_views.push((*param, true, b.GetGPUVirtualAddress()));
                        keep.push(b);
                    }
                    RootView::SrvAt(location) => root_views.push((*param, true, location)),
                    RootView::Uav(dwords) => {
                        let bytes = 4 * u64::from(dwords);
                        let b = default(bytes, D3D12_RESOURCE_STATE_COPY_DEST);
                        root_views.push((*param, false, b.GetGPUVirtualAddress()));
                        fills.push((b.clone(), self.upload(&vec![0; dwords as usize]), bytes));
                        backs.push(((true, *param), b, readback(bytes), dwords));
                    }
                }
            }
            self.submit(Some(pso), |list| {
                for (to, from, bytes) in &fills {
                    list.CopyBufferRegion(to, 0, from, 0, *bytes);
                    list.ResourceBarrier(&[transition(to, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
                }
                list.SetDescriptorHeaps(&[Some(views.clone())]);
                list.SetComputeRootSignature(root);
                list.SetComputeRoot32BitConstants(0, call.constants.len() as u32, call.constants.as_ptr().cast(), 0);
                for &(param, slot, _) in &call.tables {
                    let mut at = views.GetGPUDescriptorHandleForHeapStart();
                    at.ptr += u64::from(slot) * step as u64;
                    list.SetComputeRootDescriptorTable(param, at);
                }
                for &(param, srv, location) in &root_views {
                    if srv {
                        list.SetComputeRootShaderResourceView(param, location);
                    } else {
                        list.SetComputeRootUnorderedAccessView(param, location);
                    }
                }
                list.Dispatch(call.groups, 1, 1);
                for (_, b, back, dwords) in &backs {
                    list.ResourceBarrier(&[transition(b, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                    list.CopyBufferRegion(back, 0, b, 0, 4 * u64::from(*dwords));
                }
            })?;
            let mut ran = Ran { heap: HashMap::new(), roots: HashMap::new(), kept: HashMap::new() };
            for ((is_root, key), b, back, dwords) in &backs {
                let mut p = std::ptr::null_mut();
                back.Map(0, None, Some(&mut p)).map_err(err)?;
                let words = std::slice::from_raw_parts(p.cast::<u32>(), *dwords as usize).to_vec();
                back.Unmap(0, None);
                if *is_root {
                    ran.kept.insert(*key, b.clone());
                }
                if *is_root { &mut ran.roots } else { &mut ran.heap }.insert(*key, words);
            }
            drop((keep, fills));
            Ok(ran)
        }
    }
}

/// The root parameter of each root view of `s`, as
/// [`Host::root_signature_for`] numbers them: by slot, then kind (0 a
/// constant buffer, 1 a read-only view, 2 an unordered one).
fn root_parameter(s: &HostShader, slot: u8, kind: u8) -> Option<u32> {
    let mut views: Vec<(u8, u8)> = s
        .root_cbvs
        .iter()
        .map(|&r| (r, 0))
        .chain(s.root_srvs.iter().map(|&r| (r, 1)))
        .chain(s.root_uavs.iter().map(|&r| (r, 2)))
        .collect();
    views.sort_unstable();
    views.dedup();
    views.iter().position(|&v| v == (slot, kind)).map(|n| n as u32 + 1)
}

/// Title memory with its structures decoded, and the host's structures
/// built from them.
struct World {
    tops: Vec<Top>,
    /// The bottom levels the host has a structure of, each with its list's
    /// start in the guide table.
    bottoms: HashMap<u64, (Bottom, u32)>,
    /// Those structures: a bottom level's address, its list's start, the
    /// host structure's address ([`rays::hosts_table`]'s input).
    built: Vec<(u64, u32, u64)>,
    /// The stand-in box's address.
    stand_in: u64,
    /// The guide table for every top level, at heap slots from
    /// [`TOP_SLOT`].
    table: Vec<u32>,
    /// Per top level: the host instances in their order (a host instance's
    /// index names the title's instance) and the host structure's address.
    hosts: Vec<(Vec<Instance>, u64)>,
    _keep: Vec<ID3D12Resource>,
}

/// The heap slot of the host's first top level.
const TOP_SLOT: u32 = 80;
/// The views of the copies of title memory in a view heap's run
/// (`victory-gpu` `memory.rs`): the null view and 4,095 copies.
const COPIES: u32 = 4096;

fn world(card: &Card, memory: &Pages, tops: &[u64]) -> World {
    world_with(card, memory, tops, &|_| true)
}

/// The world where the host has structures of the bottom levels `has` says
/// so of; an instance of another is a stand-in box.
fn world_with(card: &Card, memory: &Pages, tops: &[u64], has: &dyn Fn(u64) -> bool) -> World {
    let tops: Vec<Top> = tops.iter().map(|&t| rays::top(memory, t).expect("a top level")).collect();
    let mut list: Vec<Bottom> = Vec::new();
    for i in tops.iter().flat_map(|t| &t.instances) {
        if i.bottom != 0 && has(i.bottom) && !list.iter().any(|b| b.header.address == i.bottom) {
            list.push(rays::bottom(memory, i.bottom).expect("a bottom level"));
        }
    }
    let guided: Vec<GuidedTop> = tops
        .iter()
        .enumerate()
        .map(|(n, t)| GuidedTop { address: t.header.address, root: t.header.root, slot: TOP_SLOT + n as u32 })
        .collect();
    let (table, starts) = rays::guide_table(&guided, &list.iter().collect::<Vec<_>>());
    let mut keep = Vec::new();
    let mut built: HashMap<u64, u64> = HashMap::new();
    for b in &list {
        let s = card.bottom(b);
        // SAFETY: a live buffer.
        built.insert(b.header.address, unsafe { s.GetGPUVirtualAddress() });
        keep.push(s);
    }
    let bottoms: HashMap<u64, (Bottom, u32)> = list.into_iter().zip(starts).map(|(b, s)| (b.header.address, (b, s))).collect();
    let stand_in = {
        let s = card.stand_in();
        // SAFETY: a live buffer.
        let at = unsafe { s.GetGPUVirtualAddress() };
        keep.push(s);
        at
    };
    let mut hosts = Vec::new();
    for t in &tops {
        // An instance of nothing has no host instance; nor has one whose
        // bottom level is not there to read (a dump of a run has holes).
        let placed: Vec<Instance> = t
            .instances
            .iter()
            .copied()
            .filter(|i| i.bottom != 0 && (bottoms.contains_key(&i.bottom) || rays::header(memory, i.bottom).is_ok()))
            .collect();
        let described: Vec<(Instance, Hosted)> = placed
            .iter()
            .map(|i| {
                let hosted = match bottoms.get(&i.bottom) {
                    Some((_, list)) => Hosted::Triangles { list: *list, structure: built[&i.bottom] },
                    None => Hosted::Bounds { bounds: rays::header(memory, i.bottom).expect("a header").bounds, structure: stand_in },
                };
                (*i, hosted)
            })
            .collect();
        let s = card.top(&described);
        // SAFETY: a live buffer.
        hosts.push((placed, unsafe { s.GetGPUVirtualAddress() }));
        keep.push(s);
    }
    let built = bottoms.iter().map(|(&address, (_, list))| (address, *list, built[&address])).collect();
    World { tops, bottoms, built, stand_in, table, hosts, _keep: keep }
}

fn pc_dxc() -> Option<PathBuf> {
    let dxc = match std::env::var_os("VICTORY_PC_DXC") {
        Some(p) => PathBuf::from(p),
        None => validator::Validator::default_path()?.with_file_name("dxc.exe"),
    };
    dxc.exists().then_some(dxc)
}

/// The host's own nearest hit of a ray (mode 0: every triangle opaque, the
/// instance mask and the culling the hardware's), and the candidates it
/// gives a program that judges each itself as the title's code does (mode
/// 1). Eight dwords a ray: whether a triangle was committed, its distance,
/// its two weights, the instance's index, the primitive's index, its face,
/// the number of candidates (mode 1: triangles in the low half, boxes in
/// the high half).
const ORACLE: &str = r"
cbuffer C : register(b0) { uint raysSlot; uint outSlot; uint topSlot; uint infoSlot; uint count; uint mode; uint mask; int cull; uint first; }
[numthreads(32, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= count) return;
    ByteAddressBuffer rays = ResourceDescriptorHeap[raysSlot];
    ByteAddressBuffer infos = ResourceDescriptorHeap[infoSlot];
    RWByteAddressBuffer o = ResourceDescriptorHeap[outSlot];
    RaytracingAccelerationStructure top = ResourceDescriptorHeap[topSlot];
    RayDesc ray;
    ray.Origin = asfloat(rays.Load3(id.x * 32));
    ray.Direction = asfloat(rays.Load3(id.x * 32 + 12));
    ray.TMin = 0;
    ray.TMax = asfloat(rays.Load(id.x * 32 + 24));
    RayQuery<RAY_FLAG_NONE> q;
    uint candidates = 0;
    if (mode == 0)
    {
        uint flags = RAY_FLAG_FORCE_OPAQUE;
        if (cull > 0) flags |= RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
        if (cull < 0) flags |= RAY_FLAG_CULL_FRONT_FACING_TRIANGLES;
        if (first != 0) flags |= RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
        q.TraceRayInline(top, flags, mask, ray);
        while (q.Proceed()) candidates++;
    }
    else
    {
        float extent = ray.TMax;
        q.TraceRayInline(top, RAY_FLAG_FORCE_NON_OPAQUE, 0xff, ray);
        while (q.Proceed())
        {
            // (A box: an instance the host has no triangles of. The title
            // walks it itself; here it is only counted, in the high half.)
            if (q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) { candidates += 0x10000; continue; }
            candidates++;
            uint info = infos.Load(q.CandidateInstanceIndex() * 4);
            bool front = q.CandidateTriangleFrontFace();
            int c = (info & 0x100) != 0 ? 0 : cull;
            bool culled = (c > 0 && !front) || (c < 0 && front);
            float t = q.CandidateTriangleRayT();
            if ((info & mask & 0xff) != 0 && !culled && t < extent)
            {
                extent = t;
                q.CommitNonOpaqueTriangleHit();
                if (first != 0) q.Abort();
            }
        }
    }
    uint4 a = 0, b = 0;
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        float2 w = q.CommittedTriangleBarycentrics();
        a = uint4(1, asuint(q.CommittedRayT()), asuint(w.x), asuint(w.y));
        b = uint4(q.CommittedInstanceIndex(), q.CommittedPrimitiveIndex(), q.CommittedTriangleFrontFace() ? 1 : 0, 0);
    }
    b.w = candidates;
    o.Store4(id.x * 32, a);
    o.Store4(id.x * 32 + 16, b);
}
";

fn compile(dxc: &std::path::Path, source: &str) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("victory-rays-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (input, output) = (dir.join("oracle.hlsl"), dir.join("oracle.bin"));
    std::fs::write(&input, source).unwrap();
    let out = std::process::Command::new(dxc).args(["-T", "cs_6_6", "-E", "main", "-Fo"]).arg(&output).arg(&input).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let bytes = std::fs::read(&output).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

/// Rays for a world: random ones through the bounds, and ones aimed at a
/// vertex, at a point of an edge, and inside a triangle of a random
/// instance (kind 1, 2, 3; 0 the random ones).
fn rays_for(w: &World, top: usize, count: usize, seed: u64) -> Vec<(Ray, u8)> {
    let mut g = Numbers(seed);
    let t = &w.tops[top];
    let placed: Vec<&Instance> = t.instances.iter().filter(|i| i.bottom != 0).collect();
    (0..count)
        .map(|n| {
            let extent = if n % 5 == 0 { 12.0 } else { 1000.0 };
            let kind = (n % 8).min(4) as u8 % 4;
            let random = rays::tests::ray(&mut g, &t.header.bounds, extent);
            if kind == 0 {
                return (random, 0);
            }
            let i = placed[g.next() as usize % placed.len()];
            let triangles = &w.bottoms[&i.bottom].0.triangles;
            let v = triangles[g.next() as usize % triangles.len()].vertices;
            let weights = match kind {
                1 => [1.0, 0.0, 0.0],
                2 => {
                    let s = (g.next() % 1025) as f32 / 1024.0;
                    [s, 1.0 - s, 0.0]
                }
                _ => [0.25, 0.5, 0.25],
            };
            let local: [f32; 3] = std::array::from_fn(|a| weights[0] * v[0][a] + weights[1] * v[1][a] + weights[2] * v[2][a]);
            let target: [f32; 3] = std::array::from_fn(|r| {
                i.matrix[4 * r] * local[0] + i.matrix[4 * r + 1] * local[1] + i.matrix[4 * r + 2] * local[2] + i.matrix[4 * r + 3]
            });
            let d: [f32; 3] = std::array::from_fn(|a| target[a] - random.origin[a]);
            let size = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-3);
            (Ray::new(random.origin, d.map(|x| x / size), extent), kind)
        })
        .collect()
}

/// How far inside the triangle `t` of instance `i` the ray passes: the
/// smallest of the three weights of the point where it meets the
/// triangle's plane, in the instance's space, in doubles. Negative:
/// outside, by that much of the triangle.
fn margin(i: &Instance, t: &rays::Triangle, ray: &Ray) -> f64 {
    let (shift, rows) = i.to_object;
    let turn = |v: [f32; 3]| -> [f64; 3] {
        match rows {
            Some(m) => std::array::from_fn(|a| (0..3).map(|k| f64::from(m[3 * a + k]) * f64::from(v[k])).sum()),
            None => v.map(f64::from),
        }
    };
    let (o, d) = (turn(ray.origin), turn(ray.direction));
    let o: [f64; 3] = std::array::from_fn(|a| o[a] + f64::from(shift[a]));
    let v = t.vertices.map(|p| p.map(f64::from));
    let sub = |a: [f64; 3], b: [f64; 3]| -> [f64; 3] { std::array::from_fn(|k| a[k] - b[k]) };
    let cross = |a: [f64; 3], b: [f64; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let (e1, e2) = (sub(v[1], v[0]), sub(v[2], v[0]));
    let p = cross(d, e2);
    let det = dot(e1, p);
    let s = sub(o, v[0]);
    let u = dot(s, p) / det;
    let w = dot(d, cross(s, e1)) / det;
    u.min(w).min(1.0 - u - w)
}

/// Stage 1: the host's ray query over host structures built from the
/// decoded title structures, against the title's own walk.
///
/// The rays that end differently are rays aimed at a vertex or at a point
/// of an edge: there the hardware's triangle test and the console's place
/// the ray a hair to either side of the edge. Both tests are watertight in
/// their own way: along an edge two triangles share, the console's takes
/// both and the hardware's takes one; which one the walk meets first is
/// its own order. Where the edge is the mesh's rim, or the ray sees the two
/// triangles from their sides, one test's hit is the other's miss, and the
/// ray goes on to whatever lies behind. The test says how near the edge
/// every such ray passes.
#[test]
fn the_hosts_ray_query_finds_what_the_titles_walk_finds() {
    let Some(dxc) = pc_dxc() else {
        eprintln!("no PC dxc.exe; skipped");
        return;
    };
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(card) = software() else { return };
    let oracle = HostShader {
        stage: HwStage::Cs,
        dxil: compile(&dxc, ORACLE),
        user_data: vec![UserData { index: 11, slot: 0, root: RootType::Constants, dword: 0 }],
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        root_indexed_views: Vec::new(),
    };
    let (root, pso) = card.host.compute_pipeline(&oracle).expect("the ray-query program is taken");
    let s: Scene = scene(11, 60, 0x1_2340_0000);
    let w = world(&card, &s.memory, &[s.top]);
    let triangles: usize = w.bottoms.values().map(|b| b.0.triangles.len()).sum();
    let (placed, top) = &w.hosts[0];
    eprintln!("the scene: {} instances ({} with a host instance), {} bottom levels, {triangles} triangles", w.tops[0].instances.len(), placed.len(), w.bottoms.len());
    const RAYS: usize = 4096;
    let rays = rays_for(&w, 0, RAYS, 3);
    let mut packed = vec![0u32; 8 * RAYS];
    for (n, (r, _)) in rays.iter().enumerate() {
        for a in 0..3 {
            packed[8 * n + a] = r.origin[a].to_bits();
            packed[8 * n + 3 + a] = r.direction[a].to_bits();
        }
        packed[8 * n + 6] = r.extent.to_bits();
    }
    let infos: Vec<u32> = placed.iter().map(|i| u32::from(i.mask) | u32::from(i.flags & rays::FLAG_CULL_DISABLE) << 8).collect();
    let zeros = vec![0u32; 8 * RAYS];
    let ask = |mode: u32, rules: Rules| -> Vec<u32> {
        let call = Call {
            constants: vec![0, 1, TOP_SLOT, 2, RAYS as u32, mode, u32::from(rules.mask), rules.cull as u32, u32::from(rules.first_hit), 0, 0, 0],
            roots: Vec::new(),
            heap: vec![(0, Bound::Raw(&packed)), (1, Bound::RawRw(&zeros)), (2, Bound::Raw(&infos)), (TOP_SLOT, Bound::Structure(*top))],
            tables: Vec::new(),
            groups: (RAYS / 32) as u32,
        };
        card.run(&root, &pso, &call).expect("the dispatch ran").heap.remove(&1).expect("the output")
    };
    // Distances agree within this much of the distance (of 1 below 1).
    const NEAR: f32 = 1e-4;
    let mut worst = (0f32, 0f32);
    let mut far_apart = 0;
    for rules in [
        Rules { mask: 0xda, cull: 1, first_hit: false },
        Rules { mask: 0xff, cull: 0, first_hit: false },
        Rules { mask: 0x40, cull: -1, first_hit: false },
        Rules { mask: 0x40, cull: 0, first_hit: true },
    ] {
        let got = ask(0, rules);
        let led = ask(1, rules);
        // By the kind of the ray: rays; both miss; the same triangle at the
        // same distance; another triangle at the same distance; another
        // distance; a hit of the host alone; a hit of the walk alone.
        let mut by_kind = [[0u32; 7]; 4];
        let (mut candidates, mut turns, mut led_differs, mut faces) = (0u64, 0u64, 0u32, 0u32);
        // Of the rays that end differently: those where the walk's end is
        // not that of the console's triangle test over every triangle (a
        // box of the title's hid one), and the farthest any of them passes
        // from the nearest edge of the triangles in question.
        let (mut boxes, mut widest) = (0u32, 0f64);
        let mut told = 0;
        for (n, (r, kind)) in rays.iter().enumerate() {
            let (want, counted) = rays::walk(&s.memory, s.top, r, rules).expect("the walk");
            turns += u64::from(counted.total());
            candidates += u64::from(led[8 * n + 7]);
            let o = &got[8 * n..8 * n + 8];
            let host = (o[0] == 1).then(|| {
                let i = &placed[o[4] as usize];
                let t = &w.bottoms[&i.bottom].0.triangles[o[5] as usize];
                // The host's front is the instance's: its flags may turn
                // the faces round. The walk's is the triangle's own.
                let turned = i.flags & rays::FLAG_FRONT_CCW != 0;
                (f32::from_bits(o[1]), f32::from_bits(o[2]), f32::from_bits(o[3]), (o[6] == 1) != turned, i.pointer, t.pointer)
            });
            // The program that judges candidates itself must end as the
            // host's own search does: the same distance.
            led_differs += u32::from((led[8 * n] == 1) != (o[0] == 1) || (!rules.first_hit && o[0] == 1 && led[8 * n + 1] != o[1]));
            let count = &mut by_kind[*kind as usize];
            count[0] += 1;
            let class = match (host, want) {
                (None, None) => 1,
                (Some(_), Some(_)) if rules.first_hit => 2,
                (Some(h), Some(want)) => {
                    let off = (h.0 - want.t).abs() / want.t.max(1.0);
                    let close = off <= NEAR;
                    if h.4 == want.instance && h.5 == want.triangle {
                        // One triangle: the two distances differ by the
                        // two ways the ray is brought into the instance.
                        worst.0 = worst.0.max(off);
                        worst.1 = worst.1.max((h.1 - want.i).abs()).max((h.2 - want.j).abs());
                        far_apart += u32::from(!close);
                        faces += u32::from(h.3 != want.front);
                        2
                    } else if close {
                        3
                    } else {
                        4
                    }
                }
                (Some(_), None) => 5,
                (None, Some(_)) => 6,
            };
            count[class] += 1;
            if class >= 3 {
                let all = rays::tests::nearest(&w.tops[0], &w.bottoms, r, rules);
                boxes += u32::from(all.map(|a| a.0) != want.map(|x| x.t) && !rules.first_hit);
                let found = |instance: u32, triangle: u32| {
                    let i = placed.iter().find(|i| i.pointer == instance).expect("the instance");
                    let t = w.bottoms[&i.bottom].0.triangles.iter().find(|t| t.pointer == triangle).expect("the triangle");
                    margin(i, t, r).abs()
                };
                // The edge the two tests disagree about is one of these
                // triangles' nearest.
                let near_edge = host
                    .map(|h| found(h.4, h.5))
                    .into_iter()
                    .chain(want.map(|x| found(x.instance, x.triangle)))
                    .fold(f64::INFINITY, f64::min);
                widest = widest.max(near_edge);
                if class >= 4 && *kind == 3 {
                    eprintln!("  ray {n} aimed inside a triangle ends differently: {near_edge:e} from an edge
    host {host:?}
    walk {want:?}");
                }
            }
            if class >= 4 && told < 2 {
                told += 1;
                eprintln!("  ray {n} (kind {kind}) {r:?}:\n    host {host:?}\n    walk {want:?}");
            }
        }
        let all: [u32; 7] = std::array::from_fn(|k| by_kind.iter().map(|c| c[k]).sum());
        eprintln!(
            "{rules:?}: {} rays: both miss {}, same triangle {}, another triangle at the same distance {}, another distance {}, \
             host alone {}, walk alone {}; by kind of ray (random, at a vertex, at an edge, inside) {by_kind:?}; \
             turns of the title's loop a ray {:.1}; candidates a ray {:.2}; judged by the program itself: {led_differs} rays end differently;              the same triangle seen from the other side: {faces}; of the rays that end differently, a box of the title's hid a triangle              for {boxes}, and each passes within {widest:e} of an edge of a triangle it ends at (in weights of the triangle)",
            all[0],
            all[1],
            all[2],
            all[3],
            all[4],
            all[5],
            all[6],
            turns as f64 / RAYS as f64,
            candidates as f64 / RAYS as f64,
        );
        // Rays that are not aimed at an edge or a vertex agree but for a
        // handful; aimed ones may land on either side.
        let plain = by_kind[0][4] + by_kind[0][5] + by_kind[0][6] + by_kind[3][4] + by_kind[3][5] + by_kind[3][6];
        assert!(plain * 200 <= by_kind[0][0] + by_kind[3][0], "{plain} plain rays disagree");
        assert!(all[2] > 500 || rules.first_hit && all[2] > 300, "hits: {}", all[2]);
    }
    eprintln!(
        "the same triangle: distances within {:e} of the distance (of 1 below 1), {far_apart} of them more than {NEAR:e} apart; weights within {:e}",
        worst.0, worst.1
    );
}

/// Sets the tests' switches for this thread until dropped.
struct Switches;

impl Switches {
    fn set(guide: bool, turns: Option<u32>, threads: bool) -> Self {
        TEST_GUIDE.set(Some(guide));
        TEST_TURNS.set(turns);
        LANES_ARE_THREADS.set(threads);
        Self
    }
}

impl Drop for Switches {
    fn drop(&mut self) {
        TEST_GUIDE.set(None);
        TEST_TURNS.set(None);
        LANES_ARE_THREADS.set(false);
    }
}

/// The guided node test lowers and validates in the tests' own program of
/// one node test, and asks the host for the guide table; without the
/// switch the program is byte for byte what it was.
#[test]
fn the_guided_node_test_lowers_and_validates() {
    use super::intersect_tests::one_node_test;
    let plain = one_node_test().expect("lowers");
    let (off, on) = {
        let _s = Switches::set(false, None, false);
        let off = one_node_test().expect("lowers");
        let _s = Switches::set(true, None, false);
        (off, one_node_test().expect("lowers with the guide"))
    };
    assert_eq!(plain.dxil, off.dxil);
    assert!(!off.root_srvs.contains(&SLOT_GUIDE));
    assert!(on.root_srvs.contains(&SLOT_GUIDE) && on.root_srvs.contains(&SLOT_MEMORY));
    assert!(on.dxil.len() > off.dxil.len());
    // The 32-bit form is not guided.
    let _s = Switches::set(true, None, false);
    let narrow = super::intersect_tests::one_narrow_node_test().expect("lowers");
    assert!(!narrow.root_srvs.contains(&SLOT_GUIDE));
}

/// Where the test's views are in the heap: the title's two tables, the
/// memory copies, the node tests' counts.
const TABLES: [u32; 2] = [0, 32];
const MEMORY_SLOT: u32 = 64;
const TURNS_SLOT: u32 = 90;

/// The probes along one axis of the walker's grid, and their number.
const GRID: u32 = 16;
const PROBES: u32 = GRID * GRID * GRID;

/// Stage 2: a walking program of the title, translated without and with
/// the guided node test, over the same structures and rays.
///
/// f607780aa8e502d0 traces one shadow ray a probe of a grid towards the
/// light: through the near top level and, when nothing is hit there,
/// through the far one with front faces culled. It writes 0 for a probe in
/// shadow and 0.5 for a lit one. All it reads are buffers and constants,
/// which this test makes up; its structures are the made-up scenes.
#[test]
fn a_walker_of_the_title_writes_the_same_with_the_guide() {
    let Some(path) = std::env::var_os("VICTORY_RAY_WALKER") else {
        eprintln!("no VICTORY_RAY_WALKER; skipped");
        return;
    };
    let container = std::fs::read(&path).expect("the container");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // The program as the layer runs it today declares waves of 32 lanes.
    let today = super::translate(&container).expect("translates");
    let translate = |guide: bool| {
        let _s = Switches::set(guide, Some(TURNS_SLOT), true);
        let shader = super::translate(&container).expect("translates");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        } else {
            eprintln!("no dxil.dll: not validated");
        }
        shader
    };
    let (plain, guided) = (translate(false), translate(true));
    assert!(!plain.root_srvs.contains(&SLOT_GUIDE) && guided.root_srvs.contains(&SLOT_GUIDE));
    eprintln!(
        "translated: today {} bytes; for the software adapter {} bytes without the guide, {} with it",
        today.dxil.len(),
        plain.dxil.len(),
        guided.dxil.len()
    );
    let Some(card) = software() else { return };
    if let Err(e) = card.host.compute_pipeline(&today) {
        eprintln!("as the layer runs it today the software adapter refuses the program: {e}");
    } else {
        eprintln!("as the layer runs it today the software adapter takes the program");
    }

    // Title memory: the near and the far scene.
    let near = scene(11, 60, 0x1_2340_0000);
    let far = scene(23, 30, 0x2_4680_0000);
    let mut memory = near.memory.clone();
    for n in far.memory.numbers() {
        memory.write(n * rays::PAGE, far.memory.page(n).unwrap());
    }
    let w = world(&card, &memory, &[near.top, far.top]);
    // One copy holds every page; the directory says where.
    let numbers = memory.numbers();
    let arena: Vec<u32> = numbers.iter().flat_map(|&n| memory.page(n).unwrap().iter().copied()).collect();
    let mut directory = vec![0u32; 2 * MEMORY_PAGES as usize];
    for (k, &n) in numbers.iter().enumerate() {
        directory[2 * n as usize] = 1;
        directory[2 * n as usize + 1] = (k as u64 * rays::PAGE) as u32;
    }
    eprintln!("title memory: {} pages; the guide table: {} dwords", numbers.len(), w.table.len());

    // What the program reads beside the structures.
    let list: Vec<u32> = (0..PROBES).map(|i| 1 << 29 | i).collect();
    let counts = vec![1f32.to_bits(); 64];
    let dummy = vec![0u32; 64];
    let mut grid = vec![0u32; 512];
    (grid[355], grid[356]) = (GRID, GRID);
    // The grid's middle and half sizes: the near scene's world, a little
    // above the ground.
    let (middle, half) = ([0.0f32, 3.0, 0.0], [24.0f32, 24.0, 24.0]);
    for a in 0..3 {
        grid[361 + a] = middle[a].to_bits();
    }
    (grid[364], grid[384]) = (half[0].to_bits(), half[2].to_bits());
    const EXTENT: f32 = 60.0;
    let mut constants = vec![0u32; 64];
    (constants[5], constants[17]) = (PROBES, EXTENT.to_bits());
    let address = |slot: u8, dword: u32| match (slot, dword) {
        (1, 32) => near.top,
        (1, 24) => far.top,
        _ => panic!("the address of the view at dword {dword} of table {slot}"),
    };
    let user = |s: &HostShader| -> Vec<u32> {
        let mut data = vec![0u32; (s.user_data_dwords() as usize).max(1)];
        for u in &s.user_data {
            data[u.index as usize] = match u.root {
                // (A table's address in the heap: 32 bytes a view.)
                RootType::Table => 32 * TABLES[u.slot as usize],
                // (The program's only constant picks its list of rays.)
                RootType::Constants => 0,
                RootType::Other(USER_MEMORY_BASE) => MEMORY_SLOT,
                RootType::Other(USER_VIEW_ADDRESS) => address(u.slot, u.dword) as u32,
                RootType::Other(x) if x == USER_VIEW_ADDRESS + 1 => (address(u.slot, u.dword) >> 32) as u32,
                RootType::ResourceHeap | RootType::SamplerHeap => 0,
                other => panic!("user data the test does not know: {other:?}"),
            };
        }
        data
    };
    let turns = vec![0u32; PROBES as usize];
    // The light's directions (the rays run against them).
    let lights: [[f32; 3]; 6] = [
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.6, 0.64, 0.48],
        [-0.48, 0.6, -0.64],
        [1.0, 0.0, 0.0],
        [-0.28, -0.0, 0.96],
    ];
    // How a translation runs over the host's top levels `tops` and the
    // directory `directory`: its output (a dword a probe), its counts of
    // node tests (a dword a probe), and how many pages it found no copy of.
    let run_over = |shader: &HostShader, table: &[u32], tops: [u64; 2], directory: &[u32], light: [f32; 3]| -> (Vec<u32>, Vec<u32>, usize) {
        let (root, pso) = card.host.compute_pipeline(shader).unwrap_or_else(|e| panic!("the pipeline: {e}"));
        let mut lighting = vec![0u32; 128];
        for a in 0..3 {
            lighting[68 + a] = light[a].to_bits();
        }
        let mut roots = vec![
            (root_parameter(shader, SLOT_MEMORY, 1).expect("the directory"), RootView::Srv(directory)),
            (root_parameter(shader, SLOT_FAULTS, 2).expect("the notes"), RootView::Uav(2 * MEMORY_FAULTS)),
        ];
        if let Some(param) = root_parameter(shader, SLOT_GUIDE, 1) {
            roots.push((param, RootView::Srv(table)));
        }
        let notes = roots[1].0;
        let call = Call {
            constants: user(shader),
            roots,
            heap: vec![
                (TABLES[0] + 20, Bound::Raw(&grid)),
                (TABLES[1], Bound::Typed(&list, DXGI_FORMAT_R32_FLOAT, 4)),
                (TABLES[1] + 1, Bound::Raw(&lighting)),
                (TABLES[1] + 2, Bound::Typed(&counts, DXGI_FORMAT_R32_FLOAT, 4)),
                (TABLES[1] + 5, Bound::Typed(&dummy, DXGI_FORMAT_R32_FLOAT, 4)),
                (TABLES[1] + 6, Bound::TypedRw(2 * PROBES, DXGI_FORMAT_R32G32B32A32_FLOAT, 16)),
                (TABLES[1] + 7, Bound::Cbv(&constants)),
                (MEMORY_SLOT, Bound::Null),
                (MEMORY_SLOT + 1, Bound::RawRw(&arena)),
                (TOP_SLOT, Bound::Structure(tops[0])),
                (TOP_SLOT + 1, Bound::Structure(tops[1])),
                (TURNS_SLOT, Bound::RawRw(&turns)),
            ],
            tables: Vec::new(),
            groups: PROBES / 32,
        };
        let mut ran = card.run(&root, &pso, &call).unwrap_or_else(|e| panic!("the dispatch: {e}"));
        let missing = ran.roots[&notes].iter().filter(|&&x| x != 0).count();
        assert_eq!(ran.heap[&(MEMORY_SLOT + 1)], arena, "the program wrote title memory");
        // The output's second half: the probes of list 1, the first of four
        // floats each.
        let out = ran.heap.remove(&(TABLES[1] + 6)).unwrap();
        let out = out[4 * PROBES as usize..].chunks_exact(4).map(|c| c[0]).collect();
        (out, ran.heap.remove(&TURNS_SLOT).unwrap(), missing)
    };
    let run = |shader: &HostShader, table: &[u32], light: [f32; 3]| -> (Vec<u32>, Vec<u32>) {
        let (out, turns, missing) = run_over(shader, table, [w.hosts[0].1, w.hosts[1].1], &directory, light);
        assert_eq!(missing, 0, "pages the program found no copy of");
        (out, turns)
    };
    // The guide table with the far top level left out: the far walk is
    // the console's own, in a guided program.
    let mut near_only = w.table.clone();
    near_only[8..12].copy_from_slice(&rays::GUIDE_NO_TOP);

    let shadow = Rules { mask: 0x40, cull: 0, first_hit: true };
    let (mut rays_run, mut shadowed, mut model_differs) = (0u32, 0u32, 0u32);
    let (mut differs, mut differs_half) = (0u32, 0u32);
    let (mut turns_plain, mut turns_guided, mut turns_half, mut turns_model, mut turns_differ) = (0u64, 0u64, 0u64, 0u64, 0u32);
    let (mut most_plain, mut most_guided) = (0u32, 0u32);
    // A group of 32 threads is one console wave, and the card runs a wave
    // until its last lane is done: the longest lane of each group, summed.
    let longest = |counts: &[u32]| -> u64 { counts.chunks(32).map(|group| u64::from(group.iter().copied().max().unwrap_or(0))).sum() };
    let (mut waves, mut wave_plain, mut wave_guided) = (0u64, 0u64, 0u64);
    let mut plain_outputs = Vec::new();
    for light in lights {
        let (out_plain, count_plain) = run(&plain, &w.table, light);
        plain_outputs.push(out_plain.clone());
        let (out_guided, count_guided) = run(&guided, &w.table, light);
        let (out_half, count_half) = run(&guided, &near_only, light);
        waves += u64::from(PROBES / 32);
        wave_plain += longest(&count_plain);
        wave_guided += longest(&count_guided);
        for i in 0..PROBES {
            // The probe's place, as the program computes it.
            let (x, y, z) = (i / (GRID * GRID), i / GRID % GRID, i % GRID);
            let at = |k: u32| (k as f32 + 0.5) * (1.0 / GRID as f32) * 2.0 - 1.0;
            let origin = [half[0] * at(x) + middle[0], half[0] * at(y) + middle[1], half[2] * at(z) + middle[2]];
            let ray = Ray::new(origin, light.map(|c| 0.0 - c), EXTENT);
            let (hit, mut model) = rays::walk(&memory, near.top, &ray, shadow).expect("the walk");
            let mut dark = hit.is_some();
            if !dark {
                let (hit, more) = rays::walk(&memory, far.top, &ray, Rules { cull: -1, ..shadow }).expect("the walk");
                dark = hit.is_some();
                model.boxes16 += more.total();
            }
            let n = i as usize;
            let said = |x: u32| match f32::from_bits(x) {
                0.0 => true,
                0.5 => false,
                other => panic!("probe {i}: the program wrote {other}"),
            };
            rays_run += 1;
            shadowed += u32::from(said(out_plain[n]));
            model_differs += u32::from(said(out_plain[n]) != dark);
            turns_plain += u64::from(count_plain[n]);
            turns_guided += u64::from(count_guided[n]);
            turns_half += u64::from(count_half[n]);
            turns_model += u64::from(model.total());
            turns_differ += u32::from(count_plain[n] != model.total());
            most_plain = most_plain.max(count_plain[n]);
            most_guided = most_guided.max(count_guided[n]);
            for (other, counter) in [(&out_guided, &mut differs), (&out_half, &mut differs_half)] {
                if other[n] != out_plain[n] {
                    *counter += 1;
                    if *counter <= 8 {
                        eprintln!(
                            "  light {light:?} probe {i} at {origin:?}: without the guide {}, with it {} (the walk on the CPU: {dark}); turns {} and {}",
                            f32::from_bits(out_plain[n]),
                            f32::from_bits(other[n]),
                            count_plain[n],
                            count_guided[n],
                        );
                    }
                }
            }
        }
    }
    let per = |x: u64| x as f64 / f64::from(rays_run);
    eprintln!(
        "{rays_run} rays, {shadowed} in shadow without the guide; with the guide {differs} rays differ; with the guide for the near top level \
         alone {differs_half} differ; the walk on the CPU says otherwise than the program for {model_differs}"
    );
    eprintln!(
        "node tests a ray: {:.2} without the guide (most {most_plain}), {:.2} with it (most {most_guided}), {:.2} with the near top level alone; \
         the walk on the CPU counts {:.2} ({turns_differ} rays with another count)",
        per(turns_plain),
        per(turns_guided),
        per(turns_half),
        per(turns_model),
    );
    eprintln!(
        "the longest lane of a group of 32 (what a wave runs for): {:.1} node tests without the guide, {:.1} with it",
        wave_plain as f64 / waves as f64,
        wave_guided as f64 / waves as f64
    );
    assert!(shadowed > rays_run / 20 && shadowed < rays_run * 19 / 20, "{shadowed} of {rays_run} in shadow");
    assert!(model_differs * 100 <= rays_run, "the program and the walk on the CPU differ for {model_differs} rays");
    // A ray may end differently where it passes an edge the hardware and
    // the title's test place a hair apart: a handful, said above.
    assert!(differs * 500 <= rays_run, "{differs} of {rays_run} rays differ with the guide");
    assert!(differs_half * 500 <= rays_run, "{differs_half} of {rays_run} rays differ with the near guide");
    assert!(turns_guided * 2 < turns_plain, "the guide saves no turns: {turns_guided} against {turns_plain}");

    // Box stand-ins. The host has structures of each scene's first bottom
    // level alone (the ground), then of none: every other instance comes
    // from the hardware as a box, and the title walks the bottom level
    // itself. That is the state of the first frames in the world, and for
    // good that of what the title rebuilds every frame.
    for built in [1usize, 0] {
        let has = |b: u64| built == 1 && (b == near.bottoms[0] || b == far.bottoms[0]);
        let part = world_with(&card, &memory, &[near.top, far.top], &has);
        let (mut differs, mut turns_part, mut wave_part) = (0u32, 0u64, 0u64);
        for (light, out_plain) in lights.iter().zip(&plain_outputs) {
            let (out, count, missing) = run_over(&guided, &part.table, [part.hosts[0].1, part.hosts[1].1], &directory, *light);
            assert_eq!(missing, 0, "pages the program found no copy of");
            differs += out.iter().zip(out_plain).filter(|(a, b)| a != b).count() as u32;
            turns_part += count.iter().map(|&c| u64::from(c)).sum::<u64>();
            wave_part += longest(&count);
        }
        eprintln!(
            "host structures of {} of 6 bottom levels, the others stand-in boxes: {differs} of {rays_run} rays differ; node tests a ray {:.2} \
             (without the guide {:.2}, with all structures {:.2}); the longest lane of a group of 32: {:.1}",
            2 * built,
            per(turns_part),
            per(turns_plain),
            per(turns_guided),
            wave_part as f64 / waves as f64,
        );
        assert!(differs * 500 <= rays_run, "{differs} of {rays_run} rays differ with stand-ins ({built} built)");
        assert!(turns_part < turns_plain, "stand-ins cost turns: {turns_part} against {turns_plain}");
    }

    // A late page: title memory of the near scene's second bottom level
    // (the ball) has no copy on the card, as when the title's pages lag on
    // the move. The host's structures were built while it was there. Both
    // programs then read zeros there; what each writes is counted, and the
    // pages each asks for.
    let late = near.bottoms[1] / rays::PAGE;
    let mut holed = directory.clone();
    (holed[2 * late as usize], holed[2 * late as usize + 1]) = (0, 0);
    let (mut differs, mut dark_plain, mut dark_guided, mut asked) = (0u32, 0u32, 0u32, [0usize; 2]);
    for light in lights {
        let tops = [w.hosts[0].1, w.hosts[1].1];
        let (out_plain, _, missing_plain) = run_over(&plain, &w.table, tops, &holed, light);
        let (out_guided, _, missing_guided) = run_over(&guided, &w.table, tops, &holed, light);
        differs += out_plain.iter().zip(&out_guided).filter(|(a, b)| a != b).count() as u32;
        dark_plain += out_plain.iter().filter(|&&x| x == 0).count() as u32;
        dark_guided += out_guided.iter().filter(|&&x| x == 0).count() as u32;
        asked = [asked[0].max(missing_plain), asked[1].max(missing_guided)];
    }
    eprintln!(
        "a page without a copy (one bottom level's): {differs} of {rays_run} rays differ with the guide; in shadow {dark_plain} without it, \
         {dark_guided} with it ({shadowed} with the page there); notes of pages asked for in a dispatch: {} without the guide, {} with it",
        asked[0], asked[1]
    );
    assert!(asked[0] > 0 && asked[1] > 0, "the page is asked for either way: {asked:?}");
}

/// With `VICTORY_RAY_DUMP=<folder>` (a dump of a run, `VICTORY_MEMORY_DUMP`
/// in a diagnostics build; run it with `--release`): the host's structures
/// built from the dump's on the software adapter, and for rays of this
/// test's making how many candidates an inline ray query hands a program
/// that judges each itself, as the guided node test does: with every
/// decoded bottom level on the hardware, and with the static ones alone
/// (the others as boxes). The software adapter's order of candidates is its
/// own, not the card's; the counts say what order of size to expect.
#[test]
fn a_dump_of_a_run_on_the_hosts_structures() {
    let Some(folder) = std::env::var_os("VICTORY_RAY_DUMP") else {
        eprintln!("no VICTORY_RAY_DUMP; skipped");
        return;
    };
    let Some(dxc) = pc_dxc() else {
        eprintln!("no PC dxc.exe; skipped");
        return;
    };
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(card) = software() else { return };
    let dump = rays::Dump::open(std::path::Path::new(&folder)).expect("a dump");
    let packed: std::collections::HashSet<u64> = dump.packed.iter().copied().collect();
    let oracle = HostShader {
        stage: HwStage::Cs,
        dxil: compile(&dxc, ORACLE),
        user_data: vec![UserData { index: 11, slot: 0, root: RootType::Constants, dword: 0 }],
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        root_indexed_views: Vec::new(),
    };
    let (root, pso) = card.host.compute_pipeline(&oracle).expect("the ray-query program is taken");
    const RAYS: usize = 4096;
    for &address in &dump.walked {
        let Ok(t) = rays::top(&dump.pages, address) else {
            eprintln!("{address:#x}: no top level to decode");
            continue;
        };
        for (name, static_only) in [("every decoded bottom level on the hardware", false), ("the static ones on the hardware, the others boxes", true)] {
            let has = |b: u64| (!static_only || packed.contains(&b)) && rays::bottom(&dump.pages, b).is_ok();
            let started = std::time::Instant::now();
            let w = world_with(&card, &dump.pages, &[address], &has);
            let (placed, top) = &w.hosts[0];
            if placed.is_empty() {
                eprintln!("{address:#x}, {name}: no instance to place");
                continue;
            }
            let triangles: usize = w.bottoms.values().map(|b| b.0.triangles.len()).sum();
            eprintln!(
                "{address:#x}, {name}: {} host instances of {} title instances, {} host bottom levels of {triangles} triangles, built in {:.1} s",
                placed.len(),
                t.instances.len(),
                w.bottoms.len(),
                started.elapsed().as_secs_f64()
            );
            let mut g = Numbers(address);
            let rays: Vec<Ray> = (0..RAYS).map(|_| rays::tests::ray(&mut g, &t.header.bounds, 10_000.0)).collect();
            let mut packed_rays = vec![0u32; 8 * RAYS];
            for (n, r) in rays.iter().enumerate() {
                for a in 0..3 {
                    packed_rays[8 * n + a] = r.origin[a].to_bits();
                    packed_rays[8 * n + 3 + a] = r.direction[a].to_bits();
                }
                packed_rays[8 * n + 6] = r.extent.to_bits();
            }
            let infos: Vec<u32> = placed.iter().map(|i| u32::from(i.mask) | u32::from(i.flags & rays::FLAG_CULL_DISABLE) << 8).collect();
            let zeros = vec![0u32; 8 * RAYS];
            for rules in [Rules { mask: 0x40, cull: 0, first_hit: true }, Rules { mask: 0xda, cull: 1, first_hit: false }] {
                let call = Call {
                    constants: vec![0, 1, TOP_SLOT, 2, RAYS as u32, 1, u32::from(rules.mask), rules.cull as u32, u32::from(rules.first_hit), 0, 0, 0],
                    roots: Vec::new(),
                    heap: vec![(0, Bound::Raw(&packed_rays)), (1, Bound::RawRw(&zeros)), (2, Bound::Raw(&infos)), (TOP_SLOT, Bound::Structure(*top))],
                    tables: Vec::new(),
                    groups: (RAYS / 32) as u32,
                };
                let led = card.run(&root, &pso, &call).expect("the dispatch ran").heap.remove(&1).expect("the output");
                let (mut hits, mut candidates, mut boxes, mut most) = (0u32, 0u64, 0u64, 0u32);
                let (mut walked, mut turns, mut walk_hits) = (0u32, 0u64, 0u32);
                for (n, r) in rays.iter().enumerate() {
                    let o = &led[8 * n..8 * n + 8];
                    hits += o[0];
                    candidates += u64::from(o[7] & 0xffff);
                    boxes += u64::from(o[7] >> 16);
                    most = most.max((o[7] & 0xffff) + (o[7] >> 16));
                    // (The title's own walk of the first rays, for its
                    // turns; it stops where the dump has a hole.)
                    if n < 512
                        && let Ok((hit, counted)) = rays::walk(&dump.pages, address, r, rules)
                    {
                        walked += 1;
                        turns += u64::from(counted.total());
                        walk_hits += u32::from(hit.is_some());
                    }
                }
                eprintln!(
                    "  {rules:?}: {RAYS} rays, {hits} end at a triangle of the host's; candidates a ray {:.2} triangles and {:.2} boxes (most {most}); \
                     the title's walk of {walked} of the first 512: {:.1} turns a ray, {walk_hits} hits",
                    candidates as f64 / RAYS as f64,
                    boxes as f64 / RAYS as f64,
                    turns as f64 / f64::from(walked.max(1)),
                );
            }
        }
    }
}

/// The source compiled by the SDK's `fxc.exe` (the compiler `victory-gpu`
/// compiles its own passes with) or `dxc.exe`.
fn compile_pass(tool: &std::path::Path, target: &str, source: &str) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("victory-rays-pass-{}-{target}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (input, output) = (dir.join("pass.hlsl"), dir.join("pass.bin"));
    std::fs::write(&input, source).unwrap();
    let fxc = tool.file_name().is_some_and(|n| n == "fxc.exe");
    let mut command = std::process::Command::new(tool);
    if fxc {
        command.args(["/nologo", "/T", target, "/E", "main", "/Fo"]).arg(&output).arg(&input);
    } else {
        command.args(["-T", target, "-E", "main", "-Fo"]).arg(&output).arg(&input);
    }
    let out = command.output().unwrap();
    assert!(out.status.success(), "{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
    let bytes = std::fs::read(&output).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

/// The layer's side, as far as the software adapter can say: the pass that
/// writes a host top level's instance descriptions on the card
/// ([`rays::TOP_PASS`]) from the title's instance nodes, read through the
/// directory and the copies' views as translated programs read them.
///
/// What it writes is what the CPU writes from the decoded structures
/// ([`rays::top_pass_words`]), dword for dword: instances of bottom levels
/// the host has a structure of, stand-in boxes for the others, nothing for
/// an instance of nothing and past the title's instances. The host then
/// builds its top level from that buffer where it lies, and a ray query
/// over it ends as over a top level built from the CPU's words.
#[test]
fn the_pass_on_the_card_writes_the_hosts_instances() {
    let Some(sdk) = validator::Validator::default_path() else {
        eprintln!("no Windows SDK; skipped");
        return;
    };
    let (fxc, dxc) = (sdk.with_file_name("fxc.exe"), sdk.with_file_name("dxc.exe"));
    if !dxc.exists() {
        eprintln!("no PC dxc.exe; skipped");
        return;
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Some(card) = software() else { return };
    // The pass's own root signature: its constants, the directory, the host
    // table, the descriptions, the counts, the copies' views.
    let root = {
        let range = D3D12_DESCRIPTOR_RANGE1 {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
            NumDescriptors: COPIES,
            BaseShaderRegister: 0,
            RegisterSpace: 1,
            Flags: D3D12_DESCRIPTOR_RANGE_FLAG_DESCRIPTORS_VOLATILE | D3D12_DESCRIPTOR_RANGE_FLAG_DATA_VOLATILE,
            OffsetInDescriptorsFromTableStart: 0,
        };
        let view = |kind, register| D3D12_ROOT_PARAMETER1 {
            ParameterType: kind,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 { ShaderRegister: register, RegisterSpace: 0, Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        };
        let params = [
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Constants: D3D12_ROOT_CONSTANTS { ShaderRegister: 0, RegisterSpace: 0, Num32BitValues: 8 },
                },
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
            D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).expect("a root signature");
            let blob = blob.expect("its blob");
            let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
            card.device.CreateRootSignature::<ID3D12RootSignature>(0, bytes).expect("the root signature")
        }
    };

    // Title memory: the near and the far scene; one copy holds every page.
    let near = scene(11, 60, 0x1_2340_0000);
    let far = scene(23, 30, 0x2_4680_0000);
    let mut memory = near.memory.clone();
    for n in far.memory.numbers() {
        memory.write(n * rays::PAGE, far.memory.page(n).unwrap());
    }
    let numbers = memory.numbers();
    let arena: Vec<u32> = numbers.iter().flat_map(|&n| memory.page(n).unwrap().iter().copied()).collect();
    let mut directory = vec![0u32; 2 * MEMORY_PAGES as usize];
    for (k, &n) in numbers.iter().enumerate() {
        directory[2 * n as usize] = 1;
        directory[2 * n as usize + 1] = (k as u64 * rays::PAGE) as u32;
    }
    // The far scene's cube (its third bottom level) lies in a page without
    // a copy: its instances cannot be stand-ins (no bounds to read).
    let late = far.bottoms[2] / rays::PAGE;
    let mut holed = directory.clone();
    (holed[2 * late as usize], holed[2 * late as usize + 1]) = (0, 0);

    // The host has structures of the two grounds and the near ball.
    let has = |b: u64| b == near.bottoms[0] || b == far.bottoms[0] || b == near.bottoms[1];
    let w = world_with(&card, &memory, &[near.top, far.top], &has);
    let hosts = rays::hosts_table(&w.built, 64).expect("room in the host table");
    // More than either top level has instances.
    const MOST: u32 = 128;
    let passes: Vec<(&str, Vec<u8>)> = [(&fxc, "cs_5_1"), (&dxc, "cs_6_0")]
        .into_iter()
        .filter(|(tool, _)| tool.exists())
        .map(|(tool, target)| (target, compile_pass(tool, target, rays::TOP_PASS)))
        .collect();
    eprintln!("the pass compiles: {:?}", passes.iter().map(|(t, b)| (*t, b.len())).collect::<Vec<_>>());
    let oracle = HostShader {
        stage: HwStage::Cs,
        dxil: compile(&dxc, ORACLE),
        user_data: vec![UserData { index: 11, slot: 0, root: RootType::Constants, dword: 0 }],
        samplers: Vec::new(),
        static_samplers: Vec::new(),
        root_cbvs: Vec::new(),
        root_srvs: Vec::new(),
        root_uavs: Vec::new(),
        root_indexed_views: Vec::new(),
    };
    let (oracle_root, oracle_pso) = card.host.compute_pipeline(&oracle).expect("the ray-query program is taken");
    for (target, code) in &passes {
        let pso = card.host.compute_pipeline_with(&root, code).unwrap_or_else(|e| panic!("the pass ({target}): {e}"));
        for (which, (top, directory)) in [(0usize, &directory), (1, &directory), (1, &holed)].into_iter().enumerate() {
            let t = &w.tops[top];
            let address = t.header.address;
            // (Its entry of the guide table is the second; the view of the
            // host's top level would be in slot 77; its counts from byte 32.)
            let call = Call {
                constants: vec![address as u32, (address >> 32) as u32, MOST, w.stand_in as u32, (w.stand_in >> 32) as u32, 1, 77, 32],
                roots: vec![
                    (1, RootView::Srv(directory)),
                    (2, RootView::Srv(&hosts)),
                    (3, RootView::Uav(16 * MOST)),
                    (4, RootView::Uav(16)),
                    (5, RootView::Uav(rays::GUIDE_HEADER)),
                    (6, RootView::Uav(MEMORY_FAULTS)),
                ],
                heap: vec![(MEMORY_SLOT, Bound::Null), (MEMORY_SLOT + 1, Bound::RawRw(&arena))],
                tables: vec![(7, MEMORY_SLOT, COPIES)],
                groups: MOST.div_ceil(64),
            };
            let ran = card.run(&root, &pso, &call).unwrap_or_else(|e| panic!("the pass ({target}) ran: {e}"));
            let gone = directory[2 * late as usize] == 0;
            let want = rays::top_pass_words(t, MOST, &hosts, w.stand_in, |bottom| {
                (!(gone && bottom / rays::PAGE == late)).then(|| rays::header(&memory, bottom).expect("a header").bounds)
            });
            let (got, counts) = (&ran.roots[&3], &ran.roots[&4][8..12]);
            // The guide table's entry of this top level: what the CPU's
            // table says of it, and the other entries untouched.
            let (said, _) = rays::guide_table(&[GuidedTop { address, root: t.header.root, slot: 77 }], &[]);
            assert_eq!(ran.roots[&5][8..12], said[4..8], "{target}, case {which}: the top level's entry");
            assert_eq!(ran.roots[&5][4..8], [0; 4]);
            // The page without a copy is asked for, as a walking program
            // asks: its number at its place of the notes; nothing else is.
            let notes: Vec<(usize, u32)> = ran.roots[&6].iter().copied().enumerate().filter(|n| n.1 != 0).collect();
            let asked = if gone { vec![((late % 4095 + 1) as usize, late as u32)] } else { Vec::new() };
            assert_eq!(notes, asked, "{target}, case {which}: the notes");
            let made = |words: &[u32]| words.chunks_exact(16).filter(|d| d[12] >> 24 != 0).count();
            let differ = got.chunks_exact(16).zip(want.chunks_exact(16)).filter(|(a, b)| a != b).count();
            eprintln!(
                "{target}, top level {top}{}: the title's instances {}, with a host structure {}, stand-in boxes {}, left out {}; \
                 descriptions that are not the CPU's: {differ} of {MOST}",
                if gone { " with a late page" } else { "" },
                counts[0],
                counts[1],
                counts[2],
                counts[3],
            );
            assert_eq!(counts[0] as usize, t.instances.len());
            assert_eq!(differ, 0, "{target}, case {which}");
            assert_eq!((counts[1] + counts[2]) as usize, made(&want));
            assert_eq!(counts[1] + counts[2] + counts[3], counts[0]);
            assert!(counts[1] > 0 && counts[2] > 0 && counts[3] > 0, "{counts:?}");
            assert_eq!(ran.heap[&(MEMORY_SLOT + 1)], arena, "the pass wrote title memory");

            // The host's top level, built from the buffer the pass wrote,
            // where it lies; and one built from the CPU's words.
            let on_card = &ran.kept[&3];
            card.submit(None, |list| {
                // SAFETY: a live buffer in the state the run left it in.
                unsafe {
                    list.ResourceBarrier(&[transition(on_card, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
                }
            })
            .expect("the transition ran");
            let upload = card.upload(&want);
            // SAFETY: a live buffer.
            let from_cpu = card.top_at(unsafe { upload.GetGPUVirtualAddress() }, MOST);
            // SAFETY: live buffers.
            let from_pass = card.top_at(unsafe { on_card.GetGPUVirtualAddress() }, MOST);
            const RAYS: usize = 2048;
            let mut g = Numbers(17 + which as u64);
            let mut packed = vec![0u32; 8 * RAYS];
            for n in 0..RAYS {
                let r = rays::tests::ray(&mut g, &t.header.bounds, 1000.0);
                for a in 0..3 {
                    packed[8 * n + a] = r.origin[a].to_bits();
                    packed[8 * n + 3 + a] = r.direction[a].to_bits();
                }
                packed[8 * n + 6] = r.extent.to_bits();
            }
            let (zeros, infos) = (vec![0u32; 8 * RAYS], vec![0xffu32; MOST as usize]);
            let ask = |structure: &ID3D12Resource, mode: u32| -> Vec<u32> {
                let call = Call {
                    constants: vec![0, 1, TOP_SLOT, 2, RAYS as u32, mode, 0xff, 0, 0, 0, 0, 0],
                    roots: Vec::new(),
                    // SAFETY: a live buffer.
                    heap: vec![
                        (0, Bound::Raw(&packed)),
                        (1, Bound::RawRw(&zeros)),
                        (2, Bound::Raw(&infos)),
                        (TOP_SLOT, Bound::Structure(unsafe { structure.GetGPUVirtualAddress() })),
                    ],
                    tables: Vec::new(),
                    groups: (RAYS / 32) as u32,
                };
                card.run(&oracle_root, &oracle_pso, &call).expect("the dispatch ran").heap.remove(&1).expect("the output")
            };
            // (The host's own search: its nearest triangle of the built
            // bottom levels, and how many boxes it asks the program about.)
            for mode in [0] {
                let (a, b) = (ask(&from_pass, mode), ask(&from_cpu, mode));
                let hits = a.chunks_exact(8).filter(|o| o[0] == 1).count();
                let candidates: u64 = a.chunks_exact(8).map(|o| u64::from(o[7])).sum();
                assert_eq!(a, b, "{target}, case {which}, mode {mode}: the two top levels answer differently");
                assert!(hits > RAYS / 20 && candidates > 0, "{hits} hits, {candidates} candidates");
            }
        }
    }
}

/// The title's walker f607780aa8e502d0 with what it reads beside the
/// structures (as `a_walker_of_the_title_writes_the_same_with_the_guide`
/// makes it up): two scenes in title memory, one copy of every page.
struct Walker {
    plain: HostShader,
    guided: HostShader,
    near: Scene,
    far: Scene,
    memory: Pages,
    numbers: Vec<u64>,
    arena: Vec<u32>,
    directory: Vec<u32>,
    grid: Vec<u32>,
    list: Vec<u32>,
    counts: Vec<u32>,
    constants: Vec<u32>,
}

impl Walker {
    /// `more`: structures put in title memory beside the two scenes.
    fn new(more: &[(u64, Vec<u32>)]) -> Option<Self> {
        let Some(path) = std::env::var_os("VICTORY_RAY_WALKER") else {
            eprintln!("no VICTORY_RAY_WALKER; skipped");
            return None;
        };
        let container = std::fs::read(&path).expect("the container");
        let translate = |guide: bool| {
            let _s = Switches::set(guide, Some(TURNS_SLOT), true);
            super::translate(&container).expect("translates")
        };
        let (plain, guided) = (translate(false), translate(true));
        let near = scene(11, 60, 0x1_2340_0000);
        let far = scene(23, 30, 0x2_4680_0000);
        let mut memory = near.memory.clone();
        for n in far.memory.numbers() {
            memory.write(n * rays::PAGE, far.memory.page(n).unwrap());
        }
        for (address, words) in more {
            memory.write(*address, words);
        }
        let numbers = memory.numbers();
        let arena: Vec<u32> = numbers.iter().flat_map(|&n| memory.page(n).unwrap().iter().copied()).collect();
        let mut directory = vec![0u32; 2 * MEMORY_PAGES as usize];
        for (k, &n) in numbers.iter().enumerate() {
            directory[2 * n as usize] = 1;
            directory[2 * n as usize + 1] = (k as u64 * rays::PAGE) as u32;
        }
        let mut grid = vec![0u32; 512];
        (grid[355], grid[356]) = (GRID, GRID);
        for (a, x) in [0.0f32, 3.0, 0.0].into_iter().enumerate() {
            grid[361 + a] = x.to_bits();
        }
        (grid[364], grid[384]) = (24f32.to_bits(), 24f32.to_bits());
        let mut constants = vec![0u32; 64];
        (constants[5], constants[17]) = (PROBES, 60f32.to_bits());
        Some(Walker {
            plain,
            guided,
            near,
            far,
            memory,
            numbers,
            arena,
            directory,
            grid,
            list: (0..PROBES).map(|i| 1 << 29 | i).collect(),
            counts: vec![1f32.to_bits(); 64],
            constants,
        })
    }

    /// Runs `shader` towards `light`: `table` is the guide table's address
    /// on the adapter and `tops` the host top levels (a heap slot and an
    /// address each). What it wrote a probe, and its node tests a probe.
    fn run(&self, card: &Card, shader: &HostShader, table: Option<u64>, tops: &[(u32, u64)], light: [f32; 3]) -> (Vec<u32>, Vec<u32>) {
        let (root, pso) = card.host.compute_pipeline(shader).unwrap_or_else(|e| panic!("the pipeline: {e}"));
        let mut lighting = vec![0u32; 128];
        for a in 0..3 {
            lighting[68 + a] = light[a].to_bits();
        }
        let mut roots = vec![
            (root_parameter(shader, SLOT_MEMORY, 1).expect("the directory"), RootView::Srv(&self.directory)),
            (root_parameter(shader, SLOT_FAULTS, 2).expect("the notes"), RootView::Uav(2 * MEMORY_FAULTS)),
        ];
        if let Some(param) = root_parameter(shader, SLOT_GUIDE, 1) {
            roots.push((param, RootView::SrvAt(table.expect("a guide table for a guided program"))));
        }
        let notes = roots[1].0;
        let address = |slot: u8, dword: u32| match (slot, dword) {
            (1, 32) => self.near.top,
            (1, 24) => self.far.top,
            _ => panic!("the address of the view at dword {dword} of table {slot}"),
        };
        let mut user = vec![0u32; (shader.user_data_dwords() as usize).max(1)];
        for u in &shader.user_data {
            user[u.index as usize] = match u.root {
                RootType::Table => 32 * TABLES[u.slot as usize],
                RootType::Constants => 0,
                RootType::Other(USER_MEMORY_BASE) => MEMORY_SLOT,
                RootType::Other(USER_VIEW_ADDRESS) => address(u.slot, u.dword) as u32,
                RootType::Other(x) if x == USER_VIEW_ADDRESS + 1 => (address(u.slot, u.dword) >> 32) as u32,
                RootType::ResourceHeap | RootType::SamplerHeap => 0,
                other => panic!("user data the test does not know: {other:?}"),
            };
        }
        let (dummy, turns) = (vec![0u32; 64], vec![0u32; PROBES as usize]);
        let mut heap = vec![
            (TABLES[0] + 20, Bound::Raw(&self.grid)),
            (TABLES[1], Bound::Typed(&self.list, DXGI_FORMAT_R32_FLOAT, 4)),
            (TABLES[1] + 1, Bound::Raw(&lighting)),
            (TABLES[1] + 2, Bound::Typed(&self.counts, DXGI_FORMAT_R32_FLOAT, 4)),
            (TABLES[1] + 5, Bound::Typed(&dummy, DXGI_FORMAT_R32_FLOAT, 4)),
            (TABLES[1] + 6, Bound::TypedRw(2 * PROBES, DXGI_FORMAT_R32G32B32A32_FLOAT, 16)),
            (TABLES[1] + 7, Bound::Cbv(&self.constants)),
            (MEMORY_SLOT, Bound::Null),
            (MEMORY_SLOT + 1, Bound::RawRw(&self.arena)),
            (TURNS_SLOT, Bound::RawRw(&turns)),
        ];
        heap.extend(tops.iter().map(|&(slot, location)| (slot, Bound::Structure(location))));
        let call = Call { constants: user, roots, heap, tables: Vec::new(), groups: PROBES / 32 };
        let mut ran = card.run(&root, &pso, &call).unwrap_or_else(|e| panic!("the dispatch: {e}"));
        assert_eq!(ran.roots[&notes].iter().filter(|&&x| x != 0).count(), 0, "pages the program found no copy of");
        let out = ran.heap.remove(&(TABLES[1] + 6)).unwrap();
        (out[4 * PROBES as usize..].chunks_exact(4).map(|c| c[0]).collect(), ran.heap.remove(&TURNS_SLOT).unwrap())
    }
}

/// A buffer the CPU maps and the adapter copies to and shaders write, as
/// the layer's shared buffers are.
fn shared(card: &Card, bytes: u64) -> guide_core::Mapped {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: bytes,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_CUSTOM,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
        MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
        ..Default::default()
    };
    let mut made: Option<ID3D12Resource> = None;
    let mut data = std::ptr::null_mut();
    // SAFETY: plain creation and mapping of a CPU-visible buffer.
    unsafe {
        card.device
            .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut made)
            .unwrap_or_else(|e| panic!("a shared buffer: {}", err(e)));
        let res = made.expect("a buffer");
        res.Map(0, None, Some(&mut data)).expect("mapped");
        guide_core::Mapped { res, data: data.cast(), len: bytes }
    }
}

/// The layer's side whole, as far as the software adapter can say: the
/// code of `victory-gpu` `src/guide/core.rs` itself, driven as the layer
/// drives it, frame by frame.
///
/// Four of six bottom levels are "static" (the builder is seen unpacking
/// them); the two others stay stand-in boxes. The guide copies the four
/// back, builds them, and makes a top level before each walk. The title's
/// walker, given only the guide's table and top levels, writes what it
/// writes without the guide. Then one structure goes; then the pool is too
/// small; then the card has no room: the walker's output never changes.
/// The times of the CPU's part are said, and the software adapter's time
/// for the builds.
#[test]
fn the_layers_guide_builds_what_a_walker_is_guided_by() {
    use guide_core::{Config, Guide, TopIo};
    let Some(sdk) = validator::Validator::default_path() else {
        eprintln!("no Windows SDK; skipped");
        return;
    };
    let fxc = sdk.with_file_name("fxc.exe");
    if !fxc.exists() {
        eprintln!("no fxc.exe; skipped");
        return;
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // A larger bottom level no top level names, for the builds' times: a
    // ground of 60 x 60 cells, 7,200 triangles.
    let big_at = 0x3_0000_0000u64;
    let big = {
        let corner = |x: usize, z: usize| [x as f32, ((x * 7 + z * 13) % 5) as f32 * 0.1, z as f32];
        let mut pieces = Vec::new();
        for z in 0..60 {
            for x in 0..60 {
                pieces.push(rays::tests::Piece {
                    a: corner(x, z),
                    b: corner(x, z + 1),
                    c: corner(x + 1, z),
                    d: Some(corner(x + 1, z + 1)),
                    geometry: 0,
                    opaque: true,
                });
            }
        }
        rays::tests::bottom_words(&pieces, true)
    };
    let Some(walker) = Walker::new(&[(big_at, big)]) else { return };
    let Some(card) = software() else { return };
    let pass = compile_pass(&fxc, "cs_5_1", rays::TOP_PASS);

    // Title memory on the adapter: one buffer of every page, the directory,
    // and a view heap with the copies' run (the null view, the buffer).
    let arena_bytes = 4 * walker.arena.len() as u64;
    let arena = card.buffer(arena_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
    let filled = card.upload(&walker.arena);
    card.submit(None, |list| {
        // SAFETY: live buffers of that size.
        unsafe {
            list.CopyBufferRegion(&arena, 0, &filled, 0, arena_bytes);
            list.ResourceBarrier(&[transition(&arena, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
        }
    })
    .expect("title memory is on the adapter");
    let directory = card.upload(&walker.directory);
    // SAFETY: plain D3D12 calls with descriptions that outlive them.
    let (views, copies) = unsafe {
        let views: ID3D12DescriptorHeap = card
            .device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: COPIES,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
            .expect("a view heap");
        let step = card.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        let raw = |elements: u32| D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV { NumElements: elements, Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() },
            },
        };
        let mut at = views.GetCPUDescriptorHandleForHeapStart();
        card.device.CreateUnorderedAccessView(None, None, Some(&raw(1)), at);
        at.ptr += step;
        card.device.CreateUnorderedAccessView(&arena, None, Some(&raw(walker.arena.len() as u32)), at);
        (views.clone(), views.GetGPUDescriptorHandleForHeapStart())
    };
    let page = |number: u64| walker.numbers.iter().position(|&n| n == number).map(|k| (arena.clone(), k as u64 * rays::PAGE));
    // The notes of pages without a copy (none here: every page has one),
    // and the guide table of no top level, which a walking program is bound
    // until the guide's own is live (`victory-gpu` `memory.rs` `no_guide`).
    let notes = shared(&card, 4 * u64::from(MEMORY_FAULTS));
    let no_guide = {
        let b = card.upload(&rays::guide_table(&[], &[]).0);
        // SAFETY: a live buffer.
        let at = unsafe { b.GetGPUVirtualAddress() };
        std::mem::forget(b);
        at
    };
    // The structure the builder is "given": its address, its bytes, its
    // first two dwords.
    let packed = |address: u64| {
        let h = rays::header(&walker.memory, address).expect("a header");
        let mut w = [0u32; 2];
        assert!(rays::Memory::read(&walker.memory, address, &mut w));
        (address, (u64::from(h.nodes) + 1) * 64, u64::from(w[1]) << 32 | u64::from(w[0]))
    };
    let (near, far) = (&walker.near, &walker.far);
    let statics = [near.bottoms[0], near.bottoms[1], far.bottoms[0], far.bottoms[1], big_at];
    let lights: [[f32; 3]; 3] = [[0.0, 1.0, 0.0], [0.6, 0.64, 0.48], [-0.48, 0.6, -0.64]];
    let plain: Vec<(Vec<u32>, Vec<u32>)> = lights.iter().map(|&l| walker.run(&card, &walker.plain, None, &[], l)).collect();
    let turns = |counts: &[u32]| counts.iter().map(|&c| u64::from(c)).sum::<u64>() as f64 / f64::from(PROBES);
    let plain_turns = plain.iter().map(|p| turns(&p.1)).sum::<f64>() / lights.len() as f64;

    // One frame of the layer: the copies back and the builds where a list
    // first comes to title memory by address, then a top level before each
    // walk, then the walk. What the walker wrote for each light, its node
    // tests a ray, and the microseconds of the CPU's part (copies and
    // builds; the two top levels).
    let frame_of = |guide: &mut Guide, frame: u64, room: bool, walk: bool| -> (Option<f64>, [f64; 2], f64) {
        let mut micros = [0f64; 2];
        let mut uses = Vec::new();
        let began = std::time::Instant::now();
        card.submit(None, |list| {
            let t = std::time::Instant::now();
            guide.begin(list, frame).expect("the first list");
            guide.copy_back(list, frame, &page);
            guide.work(list, frame, &mut |_| room).expect("the builds");
            micros[0] = t.elapsed().as_secs_f64() * 1e6;
            let t = std::time::Instant::now();
            // SAFETY: an open list and a live heap.
            unsafe { list.SetDescriptorHeaps(&[Some(views.clone())]) };
            for top in [near.top, far.top] {
                if let Some(used) = guide.top(top, frame).expect("a top level") {
                    // SAFETY: a live buffer.
                    let io = TopIo {
                        directory: unsafe { directory.GetGPUVirtualAddress() },
                        faults: unsafe { notes.res.GetGPUVirtualAddress() },
                        copies,
                        slot: TOP_SLOT + used.index as u32,
                    };
                    guide.record_top(list, &used, io);
                    uses.push((TOP_SLOT + used.index as u32, used.location));
                }
            }
            micros[1] = t.elapsed().as_secs_f64() * 1e6;
        })
        .expect("the frame's list ran");
        let ran = began.elapsed().as_secs_f64() * 1e3;
        if !walk {
            return (None, micros, ran);
        }
        let mut all = 0.0;
        for (light, (want, _)) in lights.iter().zip(&plain) {
            let (out, count) = walker.run(&card, &walker.guided, Some(guide.table(frame).unwrap_or(no_guide)), &uses, *light);
            let differ = out.iter().zip(want).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "frame {frame}: {differ} rays differ with the layer's guide");
            all += turns(&count);
        }
        (Some(all / lights.len() as f64), micros, ran)
    };
    let new_guide = |cfg: Config| {
        let guide = Guide::new(&card.device, cfg, &pass, shared(&card, 8 << 20), shared(&card, 4096)).expect("the guide");
        assert!(guide.buffers().len() >= 5);
        guide
    };

    // 1. Room for everything; top levels made for 32 instances at first
    // (the near one has 60).
    let mut cfg = Config::of(4);
    cfg.most = 32;
    let mut guide = new_guide(cfg);
    let mut frame = 1;
    // The first list; nothing is guided before its frames have passed.
    let (first, _, _) = frame_of(&mut guide, frame, true, true);
    assert_eq!(first, Some(plain_turns), "no top level yet: the walk is the title's own");
    for &address in &statics {
        let (address, len, header) = packed(address);
        guide.came(address, len, header, frame);
    }
    // What is no bottom level (a top level's address), and what has no page
    // on the adapter: refused, the first for good.
    let (address, len, header) = packed(near.top);
    guide.came(address, len, header, frame);
    guide.came(0x5_0000_0000, 4096, 0x3f_8000_0006, frame);
    // The builder names one of them again some frames on: it waits longer.
    frame = 6;
    let (address, len, header) = packed(near.bottoms[1]);
    guide.came(address, len, header, frame);
    let (mut work_micros, mut top_micros, mut idle) = (Vec::new(), Vec::new(), Vec::new());
    let mut build_frames = Vec::new();
    let mut said = Vec::new();
    while frame < 64 {
        frame += 1;
        let before = guide.stats;
        let walk = [cfg.quarantine + 1, cfg.quarantine + 2, 40, 63].contains(&frame);
        let (walked, micros, ran) = frame_of(&mut guide, frame, true, walk);
        let built = guide.stats.built - before.built;
        if built > 0 {
            build_frames.push((frame, built, guide.stats.triangles - before.triangles, micros[0], ran));
        } else if guide.stats.pages == before.pages && frame > 40 {
            idle.push(micros[0]);
        }
        work_micros.push(micros[0]);
        if frame > cfg.quarantine + 2 {
            top_micros.push(micros[1]);
        }
        if let Some(t) = walked {
            said.push((frame, t, guide.standing(), guide.counted(0), guide.counted(1)));
        }
    }
    for (frame, t, standing, near_counts, far_counts) in &said {
        eprintln!(
            "frame {frame}: node tests a ray {t:.2} (without the guide {plain_turns:.2}); structures built {}, waiting {}, without room {}; \
             the pass counted (instances; with a host structure, boxes, left out, since the start) near {near_counts:?} far {far_counts:?}",
            standing.0, standing.1, standing.2
        );
    }
    for (frame, built, triangles, micros, ran) in &build_frames {
        eprintln!(
            "frame {frame}: {built} bottom levels of {triangles} triangles built: the CPU's part (decode, vertices, tables, recording) {micros:.0} \
             microseconds; the list ran on the software adapter in {ran:.1} ms"
        );
    }
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let mut sorted = top_micros.clone();
    sorted.sort_by(f64::total_cmp);
    eprintln!(
        "the CPU's part a frame, microseconds: copies and builds with nothing to do {:.1} (mean of {} frames); two top levels planned and \
         recorded {:.1} (median of {}; mean {:.1})",
        mean(&idle),
        idle.len(),
        sorted[sorted.len() / 2],
        sorted.len(),
        mean(&top_micros),
    );
    let (blocks, held, other) = guide.bytes();
    eprintln!(
        "the pool: {} KB in blocks, {} KB of structures ({} triangles: {:.0} bytes a triangle ON THE SOFTWARE ADAPTER); everything else of the \
         guide's on the adapter {} KB; stats {:?}",
        blocks >> 10,
        held >> 10,
        guide.standing().3,
        held as f64 / guide.standing().3.max(1) as f64,
        other >> 10,
        guide.stats
    );
    // The near top level was made for 32 instances and has 60: not guided
    // at its first walk, made again for more, guided from then on.
    assert!(said[0].1 > said[1].1, "the second walk is guided further: {said:?}");
    assert!(said[1].1 < plain_turns && said[2].1 < 0.5 * plain_turns, "{said:?}");
    assert_eq!((guide.stats.built, guide.stats.unsound, guide.stats.no_room), (5, 1, 0), "{:?}", guide.stats);
    assert_eq!(guide.standing().0, 5);
    assert_eq!(said[3].3[0], 60, "the near top level's instances, counted on the adapter");
    // SAFETY: the notes' mapping.
    let noted = unsafe { std::slice::from_raw_parts(notes.data.cast::<u32>(), MEMORY_FAULTS as usize) };
    assert!(noted.iter().all(|&n| n == 0), "the pass asked for a page though every page has a copy");
    // With all four on the hardware the walk is what round 1's own
    // structures gave with 4 of 6 (between its "all" and its "a third").
    let all_static = said[3].1;

    // 2. The title drops one: its instances are boxes again, at once for
    // the pass, and its memory is free after the wait.
    guide.went(near.bottoms[1]);
    let held_before = guide.bytes().1;
    let mut after = Vec::new();
    while frame < 64 + 2 * cfg.quarantine + 4 {
        frame += 1;
        let walk = frame == 66 || frame == 64 + 2 * cfg.quarantine + 4;
        if let (Some(t), _, _) = frame_of(&mut guide, frame, true, walk) {
            after.push(t);
        }
    }
    eprintln!(
        "one bottom level dropped: node tests a ray {:.2} -> {:.2}; structures held {} KB -> {} KB; built now {}",
        all_static,
        after[0],
        held_before >> 10,
        guide.bytes().1 >> 10,
        guide.standing().0
    );
    assert!(after[0] > all_static && after[1] == after[0], "{after:?}");
    assert!(guide.bytes().1 < held_before && guide.standing().0 == 4 && guide.stats.went == 1);
    assert_eq!(guide.stats.not_there, 1, "what has no page on the adapter is given up after its looks: {:?}", guide.stats);

    // 3. A pool of one small block: what does not fit stays boxes. 4. A
    // card without room: nothing is built. The walker writes the same.
    for (name, cap, room) in [("a pool too small", (held_before / 3).next_multiple_of(1 << 16), true), ("no room on the card", 4 << 20, false)] {
        let mut cfg = Config::of(4);
        (cfg.cap, cfg.block) = (cap, cap);
        let mut guide = new_guide(cfg);
        frame_of(&mut guide, 1, room, false);
        for &address in &statics {
            let (address, len, header) = packed(address);
            guide.came(address, len, header, 1);
        }
        let mut last = None;
        for frame in 2..48 {
            last = frame_of(&mut guide, frame, room, frame == 47).0.or(last);
        }
        let (built, waiting, starved, _) = guide.standing();
        eprintln!(
            "{name} ({} KB): built {built}, waiting {waiting}, without room {starved}; node tests a ray {:.2}; the pool holds {} KB",
            cap >> 10,
            last.unwrap_or(0.0),
            guide.bytes().0 >> 10
        );
        assert!(starved > 0 && built + starved == 5 && guide.bytes().0 <= cap, "{:?}", guide.stats);
        assert_eq!(built == 0, !room);
        assert!(last.is_some_and(|t| t < plain_turns));
    }
}

/// Round 3, a bottom level the title builds at run time: the layer's code
/// (`guide/core.rs` `moving`, `record_moving`, the pass
/// [`rays::BOTTOM_PASS`]) builds the host's own of it from the title's
/// structure as it lies on the adapter, and refits it every frame.
///
/// The near top level here has four instances: a static ground, a ball that
/// changes every frame (twice), and a box nobody builds. Every frame the
/// "title" writes the ball's structure anew (moved, and squeezed so that its
/// tree changes too) and its top level; the guide is told what the title's
/// builder would be told (the address, the triangles of its two
/// geometries); the title's walker, guided, must write what it writes
/// without the guide, frame by frame. While the ball is a stand-in box the
/// walk takes more turns than once it is on the hardware.
#[test]
fn a_bottom_level_that_moves_is_refitted_and_guided() {
    use crate::rays::tests::{Piece, Placed, bottom_words, top_words};
    use guide_core::{Config, Guide, Moving, TopIo};
    let Some(sdk) = validator::Validator::default_path() else {
        eprintln!("no Windows SDK; skipped");
        return;
    };
    let fxc = sdk.with_file_name("fxc.exe");
    if !fxc.exists() {
        eprintln!("no fxc.exe; skipped");
        return;
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    const NEAR: u64 = 0x1_2340_0000;
    let (mover_at, big_at) = (0x3_1000_0000u64, 0x3_0000_0000u64);
    // The ball in frame `f`: single triangles, the upper half geometry 0 and
    // the lower geometry 1, each with primitive indices of its own.
    let ball = |f: u32, split: usize| -> (Vec<u32>, [u32; 2]) {
        let (lift, squeeze) = (0.12 * (f % 16) as f32, 1.0 + 0.05 * (f % 7) as f32);
        let on = |u: usize, v: usize| {
            let (a, b) = (v as f32 / 8.0 * std::f32::consts::PI, u as f32 / 12.0 * std::f32::consts::TAU);
            [a.sin() * b.cos() * 2.0 * squeeze, a.cos() * 2.0 + lift, a.sin() * b.sin() * 2.0]
        };
        let mut pieces = [Vec::new(), Vec::new()];
        for v in 0..8 {
            for u in 0..12 {
                let (p, q, r, t) = (on(u, v), on(u + 1, v), on(u, v + 1), on(u + 1, v + 1));
                let geometry = usize::from(v >= split);
                if v > 0 {
                    pieces[geometry].push(Piece { a: p, b: q, c: r, d: None, geometry: geometry as u32, opaque: true });
                }
                if v < 7 {
                    pieces[geometry].push(Piece { a: q, b: t, c: r, d: None, geometry: geometry as u32, opaque: u % 2 == 0 });
                }
            }
        }
        let counts = [pieces[0].len() as u32, pieces[1].len() as u32];
        let all: Vec<Piece> = pieces.concat();
        let mut words = bottom_words(&all, true);
        // (The encoder numbers primitives through; the title numbers each
        // geometry's from 0.)
        let first = (words[13] / 64) as usize;
        for n in counts[0] as usize..all.len() {
            words[16 * (first + n) + 13] -= counts[0];
        }
        (words, counts)
    };
    let big = {
        let corner = |x: usize, z: usize| [x as f32, ((x * 7 + z * 13) % 5) as f32 * 0.1, z as f32];
        let mut pieces = Vec::new();
        for z in 0..60 {
            for x in 0..60 {
                pieces.push(Piece { a: corner(x, z), b: corner(x, z + 1), c: corner(x + 1, z), d: Some(corner(x + 1, z + 1)), geometry: 0, opaque: true });
            }
        }
        bottom_words(&pieces, true)
    };
    // The near top level in a frame: the ground, the ball twice, a box.
    let top_of = |ground: (u64, u32, [f32; 6]), cube: (u64, u32, [f32; 6]), mover: &[u32]| -> Vec<u32> {
        let h = rays::header(&rays::Span { base: mover_at, words: mover }, mover_at).expect("the ball's header");
        // (The second ball forces its triangles opaque, the box not
        // opaque: the hardware hands the title's loop every triangle all
        // the same, because the guided node test's query is made with
        // "force not opaque", `shader.rs` `ray_guide`.)
        let placed = |matrix: [f32; 12], id: u32, bottom: (u64, u32, [f32; 6])| Placed {
            matrix,
            id,
            mask: 0xff,
            contribution: 3 * id,
            flags: [0, 0, 8, 4][id as usize],
            bottom: Some(bottom),
        };
        top_words(&[
            placed([4.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, -3.0, 0.0, 0.0, 4.0, 0.0], 0, ground),
            placed([3.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 6.0, 0.0, 0.0, 3.0, 0.0], 1, (mover_at, h.root, h.bounds)),
            placed([2.0, 0.0, 0.0, 8.0, 0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 2.0, 2.0], 2, cube),
            placed([0.0, 0.0, 2.0, -9.0, 0.0, 2.0, 0.0, 5.0, -2.0, 0.0, 0.0, 4.0], 3, (mover_at, h.root, h.bounds)),
        ])
    };
    // The scenes' own first frame, to know the static structures.
    let probe = scene(11, 60, NEAR);
    let of = |address: u64| {
        let h = rays::header(&probe.memory, address).expect("a header");
        (address, h.root, h.bounds)
    };
    let (ground, cube) = (of(probe.bottoms[0]), of(probe.bottoms[2]));
    let (first_ball, counts) = ball(0, 4);
    let first_top = top_of(ground, cube, &first_ball);
    let Some(mut walker) = Walker::new(&[(big_at, big), (mover_at, first_ball), (NEAR, first_top)]) else { return };
    let Some(card) = software() else { return };
    let (top_pass, bottom_pass) = (compile_pass(&fxc, "cs_5_1", rays::TOP_PASS), compile_pass(&fxc, "cs_5_1", rays::BOTTOM_PASS));
    eprintln!("the passes compile (cs_5_1): the top level's {} bytes, the bottom levels' {} bytes", top_pass.len(), bottom_pass.len());

    let arena_bytes = 4 * walker.arena.len() as u64;
    let arena = card.buffer(arena_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
    let filled = card.upload(&walker.arena);
    card.submit(None, |list| {
        // SAFETY: live buffers of that size.
        unsafe {
            list.CopyBufferRegion(&arena, 0, &filled, 0, arena_bytes);
            list.ResourceBarrier(&[transition(&arena, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
        }
    })
    .expect("title memory is on the adapter");
    let directory = card.upload(&walker.directory);
    // SAFETY: plain D3D12 calls with descriptions that outlive them.
    let (views, copies) = unsafe {
        let views: ID3D12DescriptorHeap = card
            .device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: COPIES,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                ..Default::default()
            })
            .expect("a view heap");
        let step = card.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
        let raw = |elements: u32| D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 {
                Buffer: D3D12_BUFFER_UAV { NumElements: elements, Flags: D3D12_BUFFER_UAV_FLAG_RAW, ..Default::default() },
            },
        };
        let mut at = views.GetCPUDescriptorHandleForHeapStart();
        card.device.CreateUnorderedAccessView(None, None, Some(&raw(1)), at);
        at.ptr += step;
        card.device.CreateUnorderedAccessView(&arena, None, Some(&raw(walker.arena.len() as u32)), at);
        (views.clone(), views.GetGPUDescriptorHandleForHeapStart())
    };
    let notes = shared(&card, 4 * u64::from(MEMORY_FAULTS));
    let no_guide = {
        let b = card.upload(&rays::guide_table(&[], &[]).0);
        // SAFETY: a live buffer.
        let at = unsafe { b.GetGPUVirtualAddress() };
        std::mem::forget(b);
        at
    };
    let numbers = walker.numbers.clone();
    let (far_top, far_ground) = (walker.far.top, walker.far.bottoms[0]);
    let page = |number: u64| numbers.iter().position(|&n| n == number).map(|k| (arena.clone(), k as u64 * rays::PAGE));
    // The "title" writes a structure anew: in the walker's own copy of
    // title memory and in the buffer on the adapter.
    let write = |walker: &mut Walker, address: u64, words: &[u32]| {
        walker.memory.write(address, words);
        let pages = address / rays::PAGE..(address + 4 * words.len() as u64).div_ceil(rays::PAGE);
        for number in pages {
            let k = numbers.iter().position(|&n| n == number).expect("a page the test has");
            let fresh = walker.memory.page(number).expect("the page").to_vec();
            walker.arena[k * (rays::PAGE / 4) as usize..(k + 1) * (rays::PAGE / 4) as usize].copy_from_slice(&fresh);
            let from = card.upload(&fresh);
            card.submit(None, |list| {
                // SAFETY: live buffers; the page lies inside the arena.
                unsafe {
                    list.ResourceBarrier(&[transition(&arena, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST)]);
                    list.CopyBufferRegion(&arena, k as u64 * rays::PAGE, &from, 0, rays::PAGE);
                    list.ResourceBarrier(&[transition(&arena, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON)]);
                }
            })
            .expect("the page is on the adapter");
        }
    };

    // (Built anew after eight refits, so that the test sees it happen.)
    let mut cfg = Config::of(8);
    (cfg.moving_rebuild, cfg.moving_cap) = (8, 4 << 20);
    let mut guide = Guide::new(&card.device, cfg, &top_pass, shared(&card, 8 << 20), shared(&card, guide_core::COUNTS_BYTES)).expect("the guide");
    guide.with_bottom_pass(&bottom_pass).expect("the bottom pass");
    // One frame of the layer: copies and builds; then what the title built
    // or refitted in it; then the top levels. The host's top levels for the
    // walk, the microseconds of the moving part on the CPU, and how long
    // the list ran.
    let frame_of = |guide: &mut Guide, frame: u64, moving: &[Moving], seen: &[u64]| -> (Vec<(u32, u64)>, f64, f64) {
        let (mut uses, mut micros) = (Vec::new(), 0.0);
        let began = std::time::Instant::now();
        // As the layer has it: the submissions' clock (here every list has
        // run when the next is recorded), and below the token of the list.
        guide.clock(frame, frame);
        card.submit(None, |list| {
            guide.begin(list, frame).expect("the first list");
            guide.copy_back(list, frame, &page);
            guide.work(list, frame, &mut |_| true).expect("the builds");
            // SAFETY: an open list and a live heap; live buffers.
            unsafe { list.SetDescriptorHeaps(&[Some(views.clone())]) };
            let io = |slot: u32| TopIo { directory: unsafe { directory.GetGPUVirtualAddress() }, faults: unsafe { notes.res.GetGPUVirtualAddress() }, copies, slot };
            let t = std::time::Instant::now();
            for m in moving {
                guide.moving(*m, frame, &mut |_| true);
            }
            for &address in seen {
                guide.seen(address, frame);
            }
            guide.record_moving(list, frame, io(0), 120, &mut |_| true).expect("the moving ones");
            micros = t.elapsed().as_secs_f64() * 1e6;
            for top in [NEAR, far_top] {
                if let Some(used) = guide.top(top, frame).expect("a top level") {
                    guide.record_top(list, &used, io(TOP_SLOT + used.index as u32));
                    uses.push((TOP_SLOT + used.index as u32, used.location));
                }
            }
        })
        .expect("the frame's list ran");
        if let Some(token) = guide.take_sent() {
            token.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        (uses, micros, began.elapsed().as_secs_f64() * 1e3)
    };
    let turns = |counts: &[u32]| counts.iter().map(|&c| u64::from(c)).sum::<u64>() as f64 / f64::from(PROBES);
    let light = [0.6, 0.64, 0.48];

    // The first list, the static grounds, and the frames before anything is
    // made.
    frame_of(&mut guide, 1, &[], &[]);
    for address in [probe.bottoms[0], far_ground] {
        let h = rays::header(&walker.memory, address).expect("a header");
        let mut w = [0u32; 2];
        assert!(rays::Memory::read(&walker.memory, address, &mut w));
        guide.came(address, (u64::from(h.nodes) + 1) * 64, u64::from(w[1]) << 32 | u64::from(w[0]), 1);
    }
    for frame in 2..24 {
        frame_of(&mut guide, frame, &[], &[]);
    }
    assert_eq!(guide.stats.built, 2, "the two grounds: {:?}", guide.stats);
    // (frame, node tests a ray without and with the guide, rays that
    // differ, the CPU's microseconds for the moving ones, the list's ms)
    let mut said = Vec::new();
    for frame in 24u64..84 {
        // (From frame 72 the ball's geometries meet at another row: the
        // counts the pass took for it are no longer its own.)
        let (ball_now, _) = ball(frame as u32, if frame >= 72 { 5 } else { 4 });
        let top_now = top_of(ground, cube, &ball_now);
        write(&mut walker, mover_at, &ball_now);
        write(&mut walker, NEAR, &top_now);
        // Nothing in the first frames (the ball is a stand-in box). From 28
        // the layer only sees that the builder wrote the ball and the large
        // ground: the pass counts them on the adapter. From 44 the title's
        // builder call describes them, and from 58 it tells too few
        // triangles for the ball. At 62 the ball is forgotten (a static
        // structure comes and goes at its address) and from then on only
        // seen again.
        let mut triangles = [0u32; rays::MOVING_GEOMETRIES];
        (triangles[0], triangles[1]) = (counts[0], if frame >= 58 { counts[1] - 10 } else { counts[1] });
        let mut big_triangles = [0u32; rays::MOVING_GEOMETRIES];
        big_triangles[0] = 7200;
        let moving = [Moving { address: mover_at, triangles }, Moving { address: big_at, triangles: big_triangles }];
        if frame == 62 {
            guide.came(mover_at, 64, 0, frame);
            guide.went(mover_at);
        }
        let described: &[Moving] = if (44..62).contains(&frame) { &moving } else { &[] };
        let seen: &[u64] = if frame >= 28 { &[mover_at, big_at] } else { &[] };
        let (uses, micros, ran) = frame_of(&mut guide, frame, described, seen);
        let (plain, plain_count) = walker.run(&card, &walker.plain, None, &[], light);
        let (out, count) = walker.run(&card, &walker.guided, Some(guide.table(frame).unwrap_or(no_guide)), &uses, light);
        let differ = out.iter().zip(&plain).filter(|(a, b)| a != b).count();
        said.push((frame, turns(&plain_count), turns(&count), differ, micros, ran));
    }
    let mean = |from: u64, to: u64, what: &dyn Fn(&(u64, f64, f64, usize, f64, f64)) -> f64| {
        let rows: Vec<f64> = said.iter().filter(|r| (from..to).contains(&r.0)).map(what).collect();
        rows.iter().sum::<f64>() / rows.len().max(1) as f64
    };
    let taken = guide.counted(guide_core::TOPS);
    eprintln!(
        "a ball of {} + {} triangles in two geometries, new every frame, instanced twice; 60 frames x 4,096 rays: rays that differ from the walk \
         without the guide: {}",
        counts[0],
        counts[1],
        said.iter().map(|r| r.3).sum::<usize>()
    );
    eprintln!(
        "node tests a ray, guide off {:.2}; guided with the ball a stand-in box (frames 24-27) {:.2}; seen by the layer and counted on the adapter, \
         then the host's own of it built and refitted (28-43: {:.2}; from 32: {:.2}); described by the title's call (44-57) {:.2}; told too few \
         triangles, so turned down on the adapter and a box again (58-61) {:.2}; forgotten, seen and counted again (66-71) {:.2}; its geometries \
         changed under the count: a box in that same list (72), counted again and back (78-83) {:.2}",
        mean(24, 84, &|r| r.1),
        mean(24, 28, &|r| r.2),
        mean(28, 44, &|r| r.2),
        mean(32, 44, &|r| r.2),
        mean(44, 58, &|r| r.2),
        mean(58, 62, &|r| r.2),
        mean(66, 72, &|r| r.2),
        mean(78, 84, &|r| r.2),
    );
    eprintln!("node tests a ray with the guide, frame by frame from 60: {:?}", said.iter().filter(|r| r.0 >= 60).map(|r| (r.0, (r.2 * 100.0).round() / 100.0)).collect::<Vec<_>>());
    eprintln!(
        "the CPU's part for the two moving ones a frame: {:.1} microseconds; the frame's list on the software adapter: {:.1} ms with nothing \
         moving, {:.1} ms with 7,368 triangles refitted (built anew every ninth frame)",
        mean(32, 58, &|r| r.4),
        mean(24, 28, &|r| r.5),
        mean(32, 58, &|r| r.5),
    );
    let (blocks, held, other) = guide.bytes();
    eprintln!(
        "host builds {} and refits {} of moving ones; the pass took {} and turned down {}; the pool {} KB in blocks, {} KB of structures; \
         everything else on the adapter {} KB; stats {:?}",
        guide.stats.moving_builds, guide.stats.moving_refits, taken[0], taken[1], blocks >> 10, held >> 10, other >> 10, guide.stats
    );
    assert!(said.iter().all(|r| r.3 == 0), "rays differ: {:?}", said.iter().filter(|r| r.3 != 0).collect::<Vec<_>>());
    assert!(mean(32, 44, &|r| r.2) < mean(24, 28, &|r| r.2), "the ball on the hardware saves turns");
    assert!(mean(44, 58, &|r| r.2) < mean(24, 28, &|r| r.2), "and so refitted");
    assert!(mean(58, 62, &|r| r.2) > mean(44, 58, &|r| r.2), "turned down: a box again");
    assert!(mean(66, 72, &|r| r.2) < mean(58, 62, &|r| r.2), "seen and counted again: on the hardware again");
    assert!(mean(78, 84, &|r| r.2) < mean(58, 62, &|r| r.2), "counted anew after its geometries changed");
    assert!(guide.stats.moving_builds >= 6 && guide.stats.moving_refits >= 30 && taken[0] > 0 && taken[1] >= 4, "{:?} {taken:?}", guide.stats);
    assert!(guide.stats.census_asked >= 4 && guide.stats.moving_misfit >= 1, "{:?}", guide.stats);
    // SAFETY: the notes' mapping.
    let noted = unsafe { std::slice::from_raw_parts(notes.data.cast::<u32>(), MEMORY_FAULTS as usize) };
    assert!(noted.iter().all(|&n| n == 0), "a pass asked for a page though every page has a copy");
}
