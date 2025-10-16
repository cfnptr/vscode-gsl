# Changelog

## [1.3.0] - 2025-10-16

- Added VK_KHR_ray_query extension support.
- Fixed some docs, namings and variable types.
- Fixed missing `default` keyword highlight.

* ext.rayQuery - [GLSL_EXT_ray_query](https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GLSL_EXT_ray_query.txt)

## [1.2.0] - 2025-06-24

- Added GLSL `invariant` keyword.
- Added GSL extension changelog.
- Added `#rayRecusrionDepth` and `gsl.rayRecursionDepth` support.
- Added `printf(fmt, ...)` support.
- Added `uintR8, sintR16G16...` image formats support.
- Added `ext.bindless, ext.debugPrintf...` extensions support.
- Fixed `gsl.variant` built-in name.
- And other docs fixes and improvements.

* ext.debugPrintf - [GLSL_EXT_debug_printf](https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GLSL_EXT_debug_printf.txt)
* ext.explicitTypes - [GL_EXT_shader_explicit_arithmetic_types](https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GL_EXT_shader_explicit_arithmetic_types.txt)
* ext.int8BitStorage - [GL_EXT_shader_8bit_storage](https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GL_EXT_shader_16bit_storage.txt)
* ext.int16BitStorage - [GL_EXT_shader_16bit_storage](https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GL_EXT_shader_16bit_storage.txt)
* ext.subgroupBasic - [GL_KHR_shader_subgroup_basic](https://github.com/KhronosGroup/GLSL/blob/main/extensions/khr/GL_KHR_shader_subgroup.txt)
* ext.subgroupVote - [GL_KHR_shader_subgroup_vote](https://github.com/KhronosGroup/GLSL/blob/main/extensions/khr/GL_KHR_shader_subgroup.txt)

## [1.1.0] - 2025-06-01

### Added

- `earlyFragmentTests` keyword support.
- `scalar` and `reference` keywords support.
- `uint8`, `uint16`, `half`, etc. keywords support.
- Input vertex attribute types docs. (`f8`, `f16`...)

Scalar layout - https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GL_EXT_scalar_block_layout.txt<br>
Buffer reference - https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GLSL_EXT_buffer_reference.txt<br>
Shader 16-bit storage - https://github.com/KhronosGroup/GLSL/blob/main/extensions/ext/GL_EXT_shader_16bit_storage.txt

### Fixed

- Incorrect code highlight docs.
- Duplicate `gl.primitiveID` declaration.

## [1.0.0] - 2025-05-19

### Added

- All base GLSL keywords and functions support.
- Buil-in keywords and functions hover tooltip support.
- Buil-in keywords and functions auto completion support.
- Ray tracing shader keywords and functions.
- Illegal HLSL functions highlight.
- README GSL vertex and fragment shaders example.

## [0.2.2] - 2025-04-23

- Added `mutable` sampler support.

## [0.2.1] - 2025-03-22

- Fixed vertex input attributes remove.

## [0.2.0] - 2025-03-17

- Refactored imageX formats.
- Added support for more formats.

## [0.1.2] - 2024-12-12

- Added f8 and f16 vertex attribute support.

## [0.1.1] - 2023-12-23

### Added

- GSL shading language logo.
- `spec` const keyword highlight.

### Fixed

- Extension name and config.

## [0.1.0] - 2023-11-16

- Added Base GSL dialect syntax highlight.
