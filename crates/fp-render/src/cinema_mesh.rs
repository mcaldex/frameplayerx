//! The cinema rooms' meshes and textures, as build.rs embeds them from
//! `rooms/<room>` (`tools/cinema/make_cinema.py` writes them).
//!
//! The mesh file is `FPXC` version 4: meshes drawn at a list of places
//! each, such as a seat at every seat, and a tiling detail texture laid
//! over them.

use crate::{CinemaRoom, Error, Result};
use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy)]
pub(crate) struct CinemaAsset {
    geometry: &'static [u8],
    /// JPEG or PNG files, in the order the mesh file counts them.
    textures: &'static [&'static [u8]],
    pub screen_width: f32,
    pub screen_height: f32,
    pub screen_center: [f32; 3],
    /// The most the screen may curve, 1 over its radius in metres.
    pub max_curvature: f32,
    /// Eye positions, one per row, in the room's metres.
    pub seats: &'static [[f32; 3]],
}

include!(concat!(env!("OUT_DIR"), "/cinema_assets.rs"));

pub(crate) const fn asset(room: CinemaRoom) -> Option<CinemaAsset> {
    ASSETS[room.slot()]
}

/// The shader works out where in the detail texture a vertex lies from its
/// position, so a vertex stays 24 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct Vertex {
    pub position: [f32; 3],
    pub uv: [f32; 2],
    /// Bit 0: stays lit with the house lights down (step lights, exit
    /// signs). Bits 1–2: which of the detail texture's patterns (its
    /// channels) lies on it. Bits 3–4: the axis (x, y, z) the face mostly
    /// faces, which the pattern is projected along. Bit 5: goes out with the
    /// house lights (a light by the screen). Bit 6: plain white, not from
    /// the texture (lettering, finer than its texels). Bit 7: bends with the
    /// screen (the screen and its masking).
    pub flags: u8,
    /// How strongly the pattern shows, out of 255.
    pub strength: u8,
    /// The pattern's tile, centimetres.
    pub tile: u8,
    pub unused: u8,
}

/// Where one copy of a mesh stands: room metres and a turn about the
/// vertical, radians.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct Placement {
    pub at: [f32; 3],
    pub yaw: f32,
}

pub(crate) struct MeshData {
    pub texture: usize,
    /// Copies out of view are left out and the rest drawn nearest first;
    /// otherwise it is drawn whole, its triangles ordered from the back of
    /// the room.
    pub culled: bool,
    /// The distance from the eyes, metres, its copies are drawn from and to.
    pub near: f32,
    pub far: f32,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub placements: Vec<Placement>,
}

pub(crate) struct Texture {
    pub width: u32,
    pub height: u32,
    /// RGBA, 8 bits a channel.
    pub pixels: Vec<u8>,
}

/// A room read, opened and decoded, ready to upload.
pub(crate) struct RoomData {
    pub meshes: Vec<MeshData>,
    pub textures: Vec<Texture>,
    /// Which texture is the tiling detail texture.
    pub detail: Option<usize>,
}

fn damaged() -> Error {
    Error::Unsupported("damaged cinema room".into())
}

/// Everything about a room that takes time, to run off the render thread.
pub(crate) fn load(room: CinemaRoom) -> Result<RoomData> {
    let asset =
        asset(room).ok_or_else(|| Error::Unsupported("cinema assets are unavailable".into()))?;
    let textures = asset
        .textures
        .iter()
        .map(|file| {
            let picture = image::load_from_memory(file)
                .map_err(|error| Error::Unsupported(format!("cinema texture: {error}")))?
                .into_rgba8();
            Ok(Texture {
                width: picture.width(),
                height: picture.height(),
                pixels: picture.into_raw(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let (mut meshes, detail) =
        placed_meshes(asset.geometry.strip_prefix(b"FPXC").ok_or_else(damaged)?)?;
    if detail.is_some_and(|d| d >= textures.len()) {
        return Err(damaged());
    }
    for mesh in &mut meshes {
        if mesh.texture >= textures.len() || Some(mesh.texture) == detail {
            return Err(damaged());
        }
        if !mesh.culled {
            nearest_first(&mesh.vertices, &mut mesh.indices);
        }
    }
    Ok(RoomData {
        meshes,
        textures,
        detail,
    })
}

/// Little-endian fields from the front of a byte slice.
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let (head, rest) = self.0.split_at_checked(n).ok_or_else(damaged)?;
        self.0 = rest;
        Ok(head)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn items<T: Pod>(&mut self, count: usize) -> Result<Vec<T>> {
        let bytes = self.take(count.checked_mul(size_of::<T>()).ok_or_else(damaged)?)?;
        Ok(bytemuck::pod_collect_to_vec(bytes))
    }
}

/// The meshes, and the detail texture's index unless it is `u32::MAX`.
fn placed_meshes(bytes: &[u8]) -> Result<(Vec<MeshData>, Option<usize>)> {
    let mut r = Reader(bytes);
    if r.u32()? != 4 {
        return Err(Error::Unsupported("unknown cinema room version".into()));
    }
    let count = r.u32()?;
    let detail = Some(r.u32()?).filter(|&d| d != u32::MAX);
    let meshes = (0..count)
        .map(|_| {
            let texture = r.u32()? as usize;
            let culled = r.u32()? != 0;
            let (near, far) = (r.f32()?, r.f32()?);
            let (vertices, indices, placements) =
                (r.u32()? as usize, r.u32()? as usize, r.u32()? as usize);
            let vertices: Vec<Vertex> = r.items(vertices)?;
            let indices: Vec<u32> = r.items(indices)?;
            let placements: Vec<Placement> = r.items(placements)?;
            check(&vertices, &indices)?;
            if placements.is_empty()
                || placements
                    .iter()
                    .any(|p| p.at.iter().chain([&p.yaw]).any(|c| !c.is_finite()))
            {
                return Err(damaged());
            }
            Ok(MeshData {
                texture,
                culled,
                near,
                far,
                vertices,
                indices,
                placements,
            })
        })
        .collect::<Result<_>>()?;
    Ok((meshes, detail.map(|d| d as usize)))
}

fn check(vertices: &[Vertex], indices: &[u32]) -> Result<()> {
    if vertices.is_empty()
        || indices.is_empty()
        || !indices.len().is_multiple_of(3)
        || indices.iter().any(|&i| i as usize >= vertices.len())
        || vertices.iter().any(|v| {
            v.position.iter().any(|p| !p.is_finite())
                || v.uv.iter().any(|c| !(0.0..=1.0).contains(c))
                || (v.flags >> 3) & 3 > 2
        })
    {
        return Err(Error::Unsupported("invalid cinema mesh".into()));
    }
    Ok(())
}

/// Orders the triangles from the back of the room to the screen, which is
/// nearest first from any seat looking at the screen. The depth test then
/// skips shading what nearer chairs hide: about 1.5 ms a frame at 3744
/// pixels an eye on the Frame.
fn nearest_first(vertices: &[Vertex], indices: &mut Vec<u32>) {
    let depth = |t: &[u32; 3]| -> f32 { t.iter().map(|&i| vertices[i as usize].position[2]).sum() };
    let mut triangles: Vec<[u32; 3]> = indices.as_chunks::<3>().0.to_vec();
    triangles.sort_by(|a, b| depth(b).total_cmp(&depth(a)));
    *indices = triangles.into_iter().flatten().collect();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_room_built_in_loads() {
        for room in CinemaRoom::ALL.into_iter().filter(|r| r.is_available()) {
            let RoomData {
                meshes,
                textures,
                detail,
            } = load(room).unwrap();
            for t in &textures {
                assert_eq!(t.pixels.len(), (t.width * t.height * 4) as usize);
            }
            assert!(detail.is_none_or(|d| d < textures.len()));
            for mesh in meshes.iter().filter(|m| !m.culled) {
                let depths: Vec<f32> = mesh
                    .indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|t| {
                        t.iter()
                            .map(|&i| mesh.vertices[i as usize].position[2])
                            .sum()
                    })
                    .collect();
                assert!(depths.is_sorted_by(|a, b| a >= b), "back of the room first");
            }
        }
    }
}
