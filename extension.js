const vscode = require('vscode');

const builtins =
[
	{
		label: '#include', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Includes the contents of another GSL source file.<br>Used to modularize shader code by reusing common functions or definitions.', 
		signature: '#include "file-name.gsl"', insertText: new vscode.SnippetString('#include $1')
	},
	{
		label: '#feature', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Requests the availability of GLSL or vendor-specific extension.', 
		signature: '#feature name', insertText: new vscode.SnippetString('#feature $1')
	},
	{
		label: 'void', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Represents the absence of a return value in functions.', 
		signature: 'void', insertText: new vscode.SnippetString('void ')
	},
	{
		label: 'main', kind: vscode.CompletionItemKind.Function, 
		documentation: 'The entry point function of a GSL shader. Execution starts here.', 
		signature: 'void main()\n{\n\t...\n}', insertText: new vscode.SnippetString('main ')
	},
	{
		label: 'discard', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Terminates processing of the current fragment and prevents it from being written to the framebuffer. <br>Used to create transparency effects or conditional fragment output.', 
		signature: 'discard;', insertText: new vscode.SnippetString('discard;')
	},

	{
		label: 'bool', kind: vscode.CompletionItemKind.Class, 
		documentation: 'Conditional type, values may be either true or false.', 
		signature: 'bool', insertText: new vscode.SnippetString('bool')
	},
	{
		label: 'int32', kind: vscode.CompletionItemKind.Class, 
		documentation: "A signed, [two's complement](https://en.wikipedia.org/wiki/Two%27s_complement), 32-bit integer.", 
		signature: 'int32', insertText: new vscode.SnippetString('int32')
	},
	{
		label: 'uint32', kind: vscode.CompletionItemKind.Class, 
		documentation: "An unsigned, [two's complement](https://en.wikipedia.org/wiki/Two%27s_complement), 32-bit integer.", 
		signature: 'uint32', insertText: new vscode.SnippetString('uint32')
	},
	{
		label: 'float', kind: vscode.CompletionItemKind.Class, 
		documentation: 'An [IEEE-754](https://en.wikipedia.org/wiki/IEEE_754) single-precision floating point number.', 
		signature: 'float', insertText: new vscode.SnippetString('float')
	},
	{
		label: 'double', kind: vscode.CompletionItemKind.Class, 
		documentation: 'An [IEEE-754](https://en.wikipedia.org/wiki/IEEE_754) double-precision floating point number.', 
		signature: 'doulbe', insertText: new vscode.SnippetString('doulbe')
	},

	{
		label: 'bool2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2-component vector of booleans.', 
		signature: 'bool2(bool x, bool y)', insertText: new vscode.SnippetString('bool2($1, $2)')
	},
	{
		label: 'bool3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3-component vector of booleans.', 
		signature: 'bool3(bool x, bool y, bool z)', insertText: new vscode.SnippetString('bool3($1, $2, $3)')
	},
	{
		label: 'bool4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 4-component vector of booleans.', 
		signature: 'bool4(bool x, bool y, bool z, bool w)', insertText: new vscode.SnippetString('bool4($1, $2, $3, $4)')
	},

	{
		label: 'int2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2-component vector of 32-bit signed integers.', 
		signature: 'int2(int32 x, int32 y)', insertText: new vscode.SnippetString('int2($1, $2)')
	},
	{
		label: 'int3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3-component vector of 32-bit signed integers.', 
		signature: 'int3(int32 x, int32 y, int32 z)', insertText: new vscode.SnippetString('int3($1, $2, $3)')
	},
	{
		label: 'int4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 4-component vector of 32-bit signed integers.', 
		signature: 'int4(int32 x, int32 y, int32 z, int32 w)', insertText: new vscode.SnippetString('int4($1, $2, $3, $4)')
	},

	{
		label: 'uint2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2-component vector of 32-bit unsigned integers.', 
		signature: 'uint2(uint32 x, uint32 y)', insertText: new vscode.SnippetString('uint2($1, $2)')
	},
	{
		label: 'uint3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3-component vector of 32-bit unsigned integers.', 
		signature: 'uint3(uint32 x, uint32 y, uint32 z)', insertText: new vscode.SnippetString('uint3($1, $2, $3)')
	},
	{
		label: 'uint4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 4-component vector of 32-bit unsigned integers.', 
		signature: 'uint4(uint32 x, uint32 y, uint32 z, uint32 w)', insertText: new vscode.SnippetString('uint4($1, $2, $3, $4)')
	},

	{
		label: 'float2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2-component vector of single-precision floating-point numbers.', 
		signature: 'float2(float x, float y)', insertText: new vscode.SnippetString('float2($1, $2)')
	},
	{
		label: 'float3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3-component vector of single-precision floating-point numbers.', 
		signature: 'float3(float x, float y, float z)', insertText: new vscode.SnippetString('float3($1, $2, $3)')
	},
	{
		label: 'float4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 4-component vector of single-precision floating-point numbers.', 
		signature: 'float4(float x, float y, float z, float w)', insertText: new vscode.SnippetString('float4($1, $2, $3, $4)')
	},

	{
		label: 'double2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2-component vector of double-precision floating-point numbers.', 
		signature: 'double2(double x, double y)', insertText: new vscode.SnippetString('double2($1, $2)')
	},
	{
		label: 'double3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3-component vector of double-precision floating-point numbers.', 
		signature: 'double3(double x, double y, double z)', insertText: new vscode.SnippetString('double3($1, $2, $3)')
	},
	{
		label: 'double4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 4-component vector of double-precision floating-point numbers.', 
		signature: 'double4(double x, double y, double z, double w)', insertText: new vscode.SnippetString('double4($1, $2, $3, $4)')
	},

	{
		label: 'float2x2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 2 columns and 2 rows of single-precision floating-point numbers.', 
		signature: 'float2x2(float2 c0, float2 c1)', insertText: new vscode.SnippetString('float2x2($1, $2)')
	},
	{
		label: 'float2x3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 2 columns and 3 rows of single-precision floating-point numbers.', 
		signature: 'float2x3(float3 c0, float3 c1)', insertText: new vscode.SnippetString('float2x3($1, $2)')
	},
	{
		label: 'float2x4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 2 columns and 4 rows of single-precision floating-point numbers.', 
		signature: 'float2x4(float4 c0, float4 c1)', insertText: new vscode.SnippetString('float2x4($1, $2)')
	},
	{
		label: 'float3x2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 3 columns and 2 rows of single-precision floating-point numbers.', 
		signature: 'float3x2(float2 c0, float2 c1, float2 c2)', insertText: new vscode.SnippetString('float3x2($1, $2, $3)')
	},
	{
		label: 'float3x3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 3 columns and 3 rows of single-precision floating-point numbers.', 
		signature: 'float3x3(float3 c0, float3 c1, float3 c2)', insertText: new vscode.SnippetString('float3x3($1, $2, $3)')
	},
	{
		label: 'float3x4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 3 columns and 4 rows of single-precision floating-point numbers.', 
		signature: 'float3x4(float4 c0, float4 c1, float4 c2)', insertText: new vscode.SnippetString('float3x4($1, $2, $3)')
	},
	{
		label: 'float4x2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 4 columns and 2 rows of single-precision floating-point numbers.', 
		signature: 'float4x2(float2 c0, float2 c1, float2 c2, float2 c3)', insertText: new vscode.SnippetString('float4x2($1, $2, $3, $4)')
	},
	{
		label: 'float4x3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 4 columns and 3 rows of single-precision floating-point numbers.', 
		signature: 'float4x3(float3 c0, float3 c1, float3 c2, float3 c3)', insertText: new vscode.SnippetString('float4x3($1, $2, $3)')
	},
	{
		label: 'float4x4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 4 columns and 4 rows of single-precision floating-point numbers.', 
		signature: 'float4x4(float4 c0, float4 c1, float4 c2, float4 c3)', insertText: new vscode.SnippetString('float4x4($1, $2, $3)')
	},

	{
		label: 'double2x2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 2 columns and 2 rows of double-precision floating-point numbers.', 
		signature: 'double2x2(double2 c0, double2 c1)', insertText: new vscode.SnippetString('double2x2($1, $2)')
	},
	{
		label: 'double2x3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 2 columns and 3 rows of double-precision floating-point numbers.', 
		signature: 'double2x3(double3 c0, double3 c1)', insertText: new vscode.SnippetString('double2x3($1, $2)')
	},
	{
		label: 'double2x4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 2 columns and 4 rows of double-precision floating-point numbers.', 
		signature: 'double2x4(double4 c0, double4 c1)', insertText: new vscode.SnippetString('double2x4($1, $2)')
	},
	{
		label: 'double3x2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 3 columns and 2 rows of double-precision floating-point numbers.', 
		signature: 'double3x2(double2 c0, double2 c1, double2 c2)', insertText: new vscode.SnippetString('double3x2($1, $2, $3)')
	},
	{
		label: 'double3x3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 3 columns and 3 rows of double-precision floating-point numbers.', 
		signature: 'double3x3(double3 c0, double3 c1, double3 c2)', insertText: new vscode.SnippetString('double3x3($1, $2, $3)')
	},
	{
		label: 'double3x4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 3 columns and 4 rows of double-precision floating-point numbers.', 
		signature: 'double3x4(double4 c0, double4 c1, double4 c2)', insertText: new vscode.SnippetString('double3x4($1, $2, $3)')
	},
	{
		label: 'double4x2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 4 columns and 2 rows of double-precision floating-point numbers.', 
		signature: 'double4x2(double2 c0, double2 c1, double2 c2, double2 c3)', insertText: new vscode.SnippetString('double4x2($1, $2, $3, $4)')
	},
	{
		label: 'double4x3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 4 columns and 3 rows of double-precision floating-point numbers.', 
		signature: 'double4x3(double3 c0, double3 c1, double3 c2, double3 c3)', insertText: new vscode.SnippetString('double4x3($1, $2, $3)')
	},
	{
		label: 'double4x4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 4 columns and 4 rows of double-precision floating-point numbers.', 
		signature: 'double4x4(double4 c0, double4 c1, double4 c2, double4 c3)', insertText: new vscode.SnippetString('double4x4($1, $2, $3)')
	},

	{
		label: 'in', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares an input variable to the shader stage or function.', 
		signature: 'in Type name;', insertText: new vscode.SnippetString('in ')
	},
	{
		label: 'out', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares an output variable from the shader stage or function.', 
		signature: 'out Type name;', insertText: new vscode.SnippetString('out ')
	},
	{
		label: 'inout', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares an input and output function variable.', 
		signature: 'inout Type name;', insertText: new vscode.SnippetString('inout ')
	},
	{
		label: 'smooth', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Default interpolation qualifier. Performs perspective-correct interpolation.', 
		signature: '<in|out> smooth Type name;', insertText: new vscode.SnippetString('smooth ')
	},
	{
		label: 'flat', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Disables interpolation. The value is passed unmodified from the provoking vertex to all fragments.', 
		signature: '<in|out> flat Type name;', insertText: new vscode.SnippetString('flat ')
	},
	{
		label: 'noperspective', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Performs linear (non-perspective-correct) interpolation across the primitive.', 
		signature: '<in|out> noperspective Type name;', insertText: new vscode.SnippetString('noperspective ')
	},
	{
		label: 'centroid', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Interpolates using values at the centroid of the covered fragment area. Helps avoid edge artifacts in multisampling.', 
		signature: '<in|out> centroid Type name;', insertText: new vscode.SnippetString('centroid ')
	},
	{
		label: 'sample', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Forces interpolation at the sample location in multisample rendering.', 
		signature: '<in|out> sample Type name;', insertText: new vscode.SnippetString('sample ')
	},

	{
		label: 'struct', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares a user-defined data structure.', 
		signature: 'struct StructName\n{\n\t...\n};', insertText: new vscode.SnippetString('struct $1\n{\n\t$2\n};')
	},
	{
		label: 'const', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares a constant value that cannot be changed after initialization.', 
		signature: 'const Type name = ...;', insertText: new vscode.SnippetString('const ')
	},
	{
		label: 'uniform', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares a global read-only shader variable shared between CPU and GPU. <br>Used to pass data from the application to the shader.', 
		signature: 'uniform set0 BufferName\n{\n\t...\n} name;', insertText: new vscode.SnippetString('uniform ')
	},
	{
		label: 'buffer', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares a shader storage buffer for reading and/or writing large data buffers.', 
		signature: 'buffer <readonly|writeonly|coherent|volatile|restrict> set0 BufferName\n{\n\t...\n} name;', insertText: new vscode.SnippetString('buffer ')
	},
	{
		label: 'coherent', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Ensures memory accesses are visible across shader invocations and GPU stages without additional synchronization.', 
		signature: 'coherent Type name;', insertText: new vscode.SnippetString('coherent ')
	},
	{
		label: 'volatile', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Prevents the compiler from optimizing away memory accesses, ensuring every read/write occurs as written.', 
		signature: 'volatile Type name;', insertText: new vscode.SnippetString('volatile ')
	},
	{
		label: 'restrict', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Indicates that memory pointed to by the variable is not <br>aliased (overlapped) with any other memory or variable used in the shader.', 
		signature: 'restrict Type name;', insertText: new vscode.SnippetString('restrict ')
	},
	{
		label: 'shared', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A variable shared between threads in a work group, stored in shared memory.', 
		signature: 'shared Type name;', insertText: new vscode.SnippetString('shared ')
	},
	{
		label: 'readonly', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Marks an image or buffer as read-only in shaders.', 
		signature: 'uniform readonly Type name;', insertText: new vscode.SnippetString('readonly ')
	},
	{
		label: 'writeonly', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Marks an image or buffer as as write-only in shaders.', 
		signature: 'uniform writeonly Type name;', insertText: new vscode.SnippetString('writeonly ')
	},
	{
		label: 'mutable', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Indicates that texture sampler state can be changed at runtime.', 
		signature: 'uniform mutable Type name;', insertText: new vscode.SnippetString('mutable ')
	},
	{
		label: 'depthLess', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Indicates fragment depth is less than stored depth.', 
		signature: 'depthLess out float gl.fragDepth;', insertText: new vscode.SnippetString('depthLess out float gl.fragDepth;')
	},
	{
		label: 'depthGreater', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Indicates fragment depth is greater than stored depth.', 
		signature: 'depthGreater out float gl.fragDepth;', insertText: new vscode.SnippetString('depthGreater out float gl.fragDepth;')
	},
	{
		label: 'localSize', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Specifies the size of a compute shader workgroup in the X, Y, and Z dimensions. <br>The values determine how many shader invocations will be launched per workgroup.', 
		signature: 'localSize = 16, 16, 1;', insertText: new vscode.SnippetString('localSize = $1, $2, $3;')
	},
	{
		label: 'spec', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares a specialization constant that can be overridden at pipeline creation time. <br>Specialization constants behave like regular constants in shaders but allow changing their values without recompiling the shader.', 
		signature: 'spec const Type NAME = ...;', insertText: new vscode.SnippetString('spec const $1 $2 = $3;')
	},
	{
		label: 'pipelineState', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares a pipeline state that configures fixed-function pipeline behavior.', 
		signature: 'pipelineState\n{\n\t[...](https://github.com/cfnptr/garden/blob/main/docs/GSL.md#pipeline-state)\n}', insertText: new vscode.SnippetString('pipelineState\n{\n\t$1\n}')
	},
	{
		label: 'pushConstants', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Defines a block of fast, read-only memory that can be updated <br>frequently from the host without rebinding descriptor sets.', 
		signature: 'uniform pushConstants\n{\n\t...\n} pc;', insertText: new vscode.SnippetString('pushConstants ')
	},
	{
		label: 'rayPayload', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares the ray payload output variable that carries data between ray tracing shader stages.', 
		signature: 'rayPayload Type name;', insertText: new vscode.SnippetString('rayPayload $1 $2;')
	},
	{
		label: 'rayPayloadIn', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares an input ray payload variable for hit group shaders.', 
		signature: 'rayPayloadIn Type name;', insertText: new vscode.SnippetString('rayPayloadIn $1 $2;')
	},
	{
		label: 'hitAttribute', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares a variable that carries per-hit information such as barycentric coordinates to hit shaders.', 
		signature: 'hitAttribute Type name;', insertText: new vscode.SnippetString('hitAttribute $1 $2;')
	},
	{
		label: 'callableData', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares data output by callable shaders, which are invoked from ray tracing shaders.', 
		signature: 'callableData Type name;', insertText: new vscode.SnippetString('callableData $1 $2;')
	},
	{
		label: 'callableDataIn', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares an input data for callable shaders.', 
		signature: 'callableDataIn Type name;', insertText: new vscode.SnippetString('callableDataIn $1 $2;')
	},
	{
		label: '#attributeOffset', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Adds offset to the input vertex attributes in bytes.', 
		signature: '#attributeOffset ...', insertText: new vscode.SnippetString('#attributeOffset $1')
	},
	{
		label: '#attachmentOffset', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Adds offset to the `subpassInput` index.', 
		signature: '#attachmentOffset ...', insertText: new vscode.SnippetString('#attachmentOffset $1')
	},
	{
		label: '#payloadOffset', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Adds offset to the `rayPayload` or `rayPayloadIn` index.', 
		signature: '#payloadOffset ...', insertText: new vscode.SnippetString('#payloadOffset $1')
	},
	{
		label: '#callableOffset', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Adds offset to the `callableData` or `callableDataIn` index.', 
		signature: '#callableOffset ...', insertText: new vscode.SnippetString('#callableOffset $1')
	},

	{
		label: 'set0', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Specifies default uniform variable first descriptor set index.', 
		signature: 'uniform set0 ...', insertText: new vscode.SnippetString('set0 ')
	},
	{
		label: 'set1', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Specifies uniform variable second descriptor set index.', 
		signature: 'uniform set1 ...', insertText: new vscode.SnippetString('set1 ')
	},
	{
		label: 'set2', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Specifies uniform variable third descriptor set index.', 
		signature: 'uniform set2 ...', insertText: new vscode.SnippetString('set2 ')
	},
	{
		label: 'set3', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Specifies uniform variable fourth descriptor set index.', 
		signature: 'uniform set3 ...', insertText: new vscode.SnippetString('set3 ')
	},

	{
		label: 'sampler1D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 1D texture sampler used to sample floating-point data in shaders.', 
		signature: 'uniform set0 sampler1D\n{\n\t...\n} name;', insertText: new vscode.SnippetString('sampler1D ')
	},
	{
		label: 'sampler2D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D texture sampler used to sample floating-point data in shaders.', 
		signature: 'uniform set0 sampler2D\n{\n\t...\n} name;', insertText: new vscode.SnippetString('sampler2D ')
	},
	{
		label: 'sampler3D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3D texture sampler used to sample floating-point data in shaders.', 
		signature: 'uniform set0 sampler3D\n{\n\t...\n} name;', insertText: new vscode.SnippetString('sampler3D ')
	},
	{
		label: 'samplerCube', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A cubemap texture sampler used to sample floating-point data in shaders.', 
		signature: 'uniform set0 samplerCube\n{\n\t...\n} name;', insertText: new vscode.SnippetString('samplerCube ')
	},
	{
		label: 'sampler1DArray', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 1D texture array sampler used to sample floating-point data in shaders.', 
		signature: 'uniform set0 sampler1DArray\n{\n\t...\n} name;', insertText: new vscode.SnippetString('sampler1DArray ')
	},
	{
		label: 'sampler2DArray', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D texture array sampler used to sample floating-point data in shaders.', 
		signature: 'uniform set0 sampler2DArray\n{\n\t...\n} name;', insertText: new vscode.SnippetString('sampler2DArray ')
	},

	{
		label: 'isampler1D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 1D texture sampler used to sample signed integer data in shaders.', 
		signature: 'uniform set0 isampler1D\n{\n\t...\n} name;', insertText: new vscode.SnippetString('isampler1D ')
	},
	{
		label: 'isampler2D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D texture sampler used to sample signed integer data in shaders.', 
		signature: 'uniform set0 isampler2D\n{\n\t...\n} name;', insertText: new vscode.SnippetString('isampler2D ')
	},
	{
		label: 'isampler3D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3D texture sampler used to sample signed integer data in shaders.', 
		signature: 'uniform set0 isampler3D\n{\n\t...\n} name;', insertText: new vscode.SnippetString('isampler3D ')
	},
	{
		label: 'isamplerCube', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A cubemap texture sampler used to sample signed integer data in shaders.', 
		signature: 'uniform set0 isamplerCube\n{\n\t...\n} name;', insertText: new vscode.SnippetString('isamplerCube ')
	},
	{
		label: 'isampler1DArray', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 1D texture array sampler used to sample signed integer data in shaders.', 
		signature: 'uniform set0 isampler1DArray\n{\n\t...\n} name;', insertText: new vscode.SnippetString('isampler1DArray ')
	},
	{
		label: 'isampler2DArray', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D texture array sampler used to sample signed integer data in shaders.', 
		signature: 'uniform set0 isampler2DArray\n{\n\t...\n} name;', insertText: new vscode.SnippetString('isampler2DArray ')
	},

	{
		label: 'usampler1D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 1D texture sampler used to sample signed integer data in shaders.', 
		signature: 'uniform set0 usampler1D\n{\n\t...\n} name;', insertText: new vscode.SnippetString('usampler1D ')
	},
	{
		label: 'usampler2D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D texture sampler used to sample signed integer data in shaders.', 
		signature: 'uniform set0 usampler2D\n{\n\t...\n} name;', insertText: new vscode.SnippetString('usampler2D ')
	},
	{
		label: 'usampler3D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3D texture sampler used to sample signed integer data in shaders.', 
		signature: 'uniform set0 usampler3D\n{\n\t...\n} name;', insertText: new vscode.SnippetString('usampler3D ')
	},
	{
		label: 'usamplerCube', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A cubemap texture sampler used to sample signed integer data in shaders.', 
		signature: 'uniform set0 usamplerCube\n{\n\t...\n} name;', insertText: new vscode.SnippetString('usamplerCube ')
	},
	{
		label: 'usampler1DArray', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 1D texture array sampler used to sample signed integer data in shaders.', 
		signature: 'uniform set0 usampler1DArray\n{\n\t...\n} name;', insertText: new vscode.SnippetString('usampler1DArray ')
	},
	{
		label: 'usampler2DArray', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D texture array sampler used to sample signed integer data in shaders.', 
		signature: 'uniform set0 usampler2DArray\n{\n\t...\n} name;', insertText: new vscode.SnippetString('usampler2DArray ')
	},

	{
		label: 'sampler1DShadow', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 1D texture sampler used to sample depth data in shaders.', 
		signature: 'uniform set0 sampler1DShadow\n{\n\t...\n} name;', insertText: new vscode.SnippetString('sampler1DShadow ')
	},
	{
		label: 'sampler2DShadow', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D texture sampler used to sample depth data in shaders.', 
		signature: 'uniform set0 sampler2DShadow\n{\n\t...\n} name;', insertText: new vscode.SnippetString('sampler2DShadow ')
	},
	{
		label: 'sampler3DShadow', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3D texture sampler used to sample depth data in shaders.', 
		signature: 'uniform set0 sampler3DShadow\n{\n\t...\n} name;', insertText: new vscode.SnippetString('sampler3DShadow ')
	},
	{
		label: 'samplerCubeShadow', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A cubemap texture sampler used to sample depth data in shaders.', 
		signature: 'uniform set0 samplerCubeShadow\n{\n\t...\n} name;', insertText: new vscode.SnippetString('samplerCubeShadow ')
	},
	{
		label: 'sampler1DArrayShadow', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 1D texture array sampler used to sample depth data in shaders.', 
		signature: 'uniform set0 sampler1DArrayShadow\n{\n\t...\n} name;', insertText: new vscode.SnippetString('sampler1DArrayShadow ')
	},
	{
		label: 'sampler2DArrayShadow', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D texture array sampler used to sample depth data in shaders.', 
		signature: 'uniform set0 sampler2DArrayShadow\n{\n\t...\n} name;', insertText: new vscode.SnippetString('sampler2DArrayShadow ')
	},

	{
		label: 'image1D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 1D image that can be read from or written to in shaders, with signed integer data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 image1D name : Format;', insertText: new vscode.SnippetString('image1D ')
	},
	{
		label: 'image2D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D image that can be read from or written to in shaders, with signed integer data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 image2D name : Format;', insertText: new vscode.SnippetString('image2D ')
	},
	{
		label: 'image3D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3D image that can be read from or written to in shaders, with signed integer data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 image3D name : Format;', insertText: new vscode.SnippetString('image3D ')
	},
	{
		label: 'imageCube', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A cubemap image that can be read from or written to in shaders, with signed integer data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 imageCube name : Format;', insertText: new vscode.SnippetString('imageCube ')
	},
	{
		label: 'image1DArray', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 1D image array that can be read from or written to in shaders, with signed integer data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 image1DArray name : Format;', insertText: new vscode.SnippetString('image1DArray ')
	},
	{
		label: 'image2DArray', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D image array that can be read from or written to in shaders, with signed integer data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 image2DArray name : Format;', insertText: new vscode.SnippetString('image2DArray ')
	},

	{
		label: 'iimage1D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 1D image that can be read from or written to in shaders, with signed integer data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 iimage1D name : Format;', insertText: new vscode.SnippetString('iimage1D ')
	},
	{
		label: 'iimage2D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D image that can be read from or written to in shaders, with signed integer data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 iimage2D name : Format;', insertText: new vscode.SnippetString('iimage2D ')
	},
	{
		label: 'iimage3D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3D image that can be read from or written to in shaders, with signed integer data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 iimage3D name : Format;', insertText: new vscode.SnippetString('iimage3D ')
	},
	{
		label: 'iimageCube', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A cubemap image that can be read from or written to in shaders, with signed integer data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 iimageCube name : Format;', insertText: new vscode.SnippetString('iimageCube ')
	},
	{
		label: 'iimage1DArray', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 1D image array that can be read from or written to in shaders, with signed integer data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 iimage1DArray name : Format;', insertText: new vscode.SnippetString('iimage1DArray ')
	},
	{
		label: 'iimage2DArray', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D image array that can be read from or written to in shaders, with signed integer data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 iimage2DArray name : Format;', insertText: new vscode.SnippetString('iimage2DArray ')
	},

	{
		label: 'uimage1D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 1D image that can be read from or written to in shaders, with unsigned integer data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 uimage1D name : Format;', insertText: new vscode.SnippetString('uimage1D ')
	},
	{
		label: 'uimage2D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D image that can be read from or written to in shaders, with unsigned integer data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 uimage2D name : Format;', insertText: new vscode.SnippetString('uimage2D ')
	},
	{
		label: 'uimage3D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3D image that can be read from or written to in shaders, with unsigned integer data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 uimage3D name : Format;', insertText: new vscode.SnippetString('uimage3D ')
	},
	{
		label: 'uimageCube', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A cubemap image that can be read from or written to in shaders, with unsigned integerdata.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 uimageCube name : Format;', insertText: new vscode.SnippetString('uimageCube ')
	},
	{
		label: 'uimage1DArray', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 1D image array that can be read from or written to in shaders, with unsigned integer data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 uimage1DArray name : Format;', insertText: new vscode.SnippetString('uimage1DArray ')
	},
	{
		label: 'uimage2DArray', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D image array that can be read from or written to in shaders, with unsigned integer data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 uimage2DArray name : Format;', insertText: new vscode.SnippetString('uimage2DArray ')
	},

	{
		label: 'subpassInput', kind: vscode.CompletionItemKind.Class, 
		documentation: 'An image type that allows reading from a framebuffer attachment during a subpass.', 
		signature: 'uniform subpassInput name;', insertText: new vscode.SnippetString('subpassInput ')
	},
	{
		label: 'accelerationStructure', kind: vscode.CompletionItemKind.Class, 
		documentation: "A structure that holds the scene's spatial hierarchy for ray tracing.", 
		signature: 'uniform accelerationStructure name;', insertText: new vscode.SnippetString('accelerationStructure ')
	},
	
	{
		label: 'if', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Conditional branch. Executes the block if the condition is true.', 
		signature: 'if (condition)\n{\n\t...\n}', insertText: new vscode.SnippetString('if ($1)')
	},
	{
		label: 'else', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Alternative block executed if the preceding "if" condition is false.', 
		signature: 'else\n{\n\t...\n}', insertText: new vscode.SnippetString('else')
	},
	{
		label: 'for', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Loop that runs for a specific number of iterations.', 
		signature: 'for (Type i = 0; i < count; i++)\n{\n\t...\n}', insertText: new vscode.SnippetString('for ($1 i = 0; i < $2; i++)')
	},
	{
		label: 'while', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Loop that runs as long as a condition is true.', 
		signature: 'while (condition)\n{\n\t...\n}', insertText: new vscode.SnippetString('while ($1)')
	},
	{
		label: 'do', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Starts a do-while loop. The loop body executes at least once.', 
		signature: 'do\n{\n\t...\n}\nwhile (condition);', insertText: new vscode.SnippetString('do')
	},
	{
		label: 'switch', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Selects a block of code to run based on a matching value.', 
		signature: 'switch (value)\n{\n\t...\n}', insertText: new vscode.SnippetString('switch ($1)')
	},
	{
		label: 'case', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Specifies a branch in a switch statement.', 
		signature: 'case ...:', insertText: new vscode.SnippetString('case $1:')
	},
	{
		label: 'default', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Default branch in a switch statement if no case matches.', 
		signature: 'default:', insertText: new vscode.SnippetString('default:')
	},
	{
		label: 'break', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Exits the nearest enclosing loop or switch.', 
		signature: 'break;', insertText: new vscode.SnippetString('break;')
	},
	{
		label: 'continue', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Skips the current loop iteration and continues with the next.', 
		signature: 'continue;', insertText: new vscode.SnippetString('continue;')
	},
	{
		label: 'return', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Returns a value and exits the current function.', 
		signature: 'return value;', insertText: new vscode.SnippetString('return')
	},

	{
		label: 'abs', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return the absolute value of the `x`.',
		signature: 'Type abs(Type x)', insertText: new vscode.SnippetString('abs($1)')
	},
	{
		label: 'sign', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Extract the sign of the `x`.',
		signature: 'Type sign(Type x)', insertText: new vscode.SnippetString('sign($1)')
	},
	{
		label: 'floor', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Find the nearest integer less than or equal to the `x`.',
		signature: 'Type floor(Type x)', insertText: new vscode.SnippetString('floor($1)')
	},
	{
		label: 'ceil', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Find the nearest integer that is greater than or equal to the `x`.',
		signature: 'Type ceil(Type x)', insertText: new vscode.SnippetString('ceil($1)')
	},
	{
		label: 'round', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Find the nearest integer to the `x`.',
		signature: 'Type round(Type x)', insertText: new vscode.SnippetString('round($1)')
	},
	{
		label: 'roundEven', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Find the nearest even integer to the `x`.',
		signature: 'Type roundEven(Type x)', insertText: new vscode.SnippetString('roundEven($1)')
	},

	{
		label: 'sin', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the sine of the `angle` in radians.',
		signature: 'Type sin(Type angle)', insertText: new vscode.SnippetString('sin($1)')
	},
	{
		label: 'cos', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the cosine of the `angle` in radians.',
		signature: 'Type cos(Type angle)', insertText: new vscode.SnippetString('cos($1)')
	},
	{
		label: 'tan', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the tangent of the `angle` in radians.',
		signature: 'Type tan(Type angle)', insertText: new vscode.SnippetString('tan($1)')
	},
	{
		label: 'asin', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the arcsine of the `angle` in radians.',
		signature: 'Type asin(Type angle)', insertText: new vscode.SnippetString('asin($1)')
	},
	{
		label: 'acos', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the arccosine of the `angle` in radians.',
		signature: 'Type acos(Type angle)', insertText: new vscode.SnippetString('acos($1)')
	},
	{
		label: 'atan', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the arc-tangent of the `angle` in radians.',
		signature: 'Type atan(Type angle)', insertText: new vscode.SnippetString('atan($1)')
	},
	{
		label: 'sinh', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the hyperbolic sine of the `angle` in radians.',
		signature: 'Type sinh(Type angle)', insertText: new vscode.SnippetString('sinh($1)')
	},
	{
		label: 'cosh', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the hyperbolic cosine of the `angle` in radians.',
		signature: 'Type cosh(Type angle)', insertText: new vscode.SnippetString('cosh($1)')
	},
	{
		label: 'tanh', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the hyperbolic tangent of the `angle` in radians.',
		signature: 'Type tanh(Type angle)', insertText: new vscode.SnippetString('tanh($1)')
	},
	{
		label: 'asinh', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the arc hyperbolic sine of the `angle` in radians.',
		signature: 'Type asinh(Type angle)', insertText: new vscode.SnippetString('asinh($1)')
	},
	{
		label: 'acosh', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the arc hyperbolic cosine of the `angle` in radians.',
		signature: 'Type acosh(Type angle)', insertText: new vscode.SnippetString('acosh($1)')
	},
	{
		label: 'atanh', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the arc hyperbolic tangent of the `angle` in radians.',
		signature: 'Type atanh(Type angle)', insertText: new vscode.SnippetString('atanh($1)')
	},

	{
		label: 'gl.vertexIndex', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains index of the current vertex. (Vertex Shader)', 
		signature: 'in int32 gl.vertexIndex;', insertText: new vscode.SnippetString('gl.vertexIndex')
	},
	{
		label: 'gl.baseVertex', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains base vertex offset. (Vertex Shader)', 
		signature: 'in int32 gl.baseVertex;', insertText: new vscode.SnippetString('gl.baseVertex')
	},
	{
		label: 'gl.instanceIndex', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains index of the current instance. (Vertex Shader)', 
		signature: 'in int32 gl.instanceIndex;', insertText: new vscode.SnippetString('gl.instanceIndex')
	},
	{
		label: 'gl.baseInstance', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains base instance offset. (Vertex Shader)', 
		signature: 'in int32 gl.baseInstance;', insertText: new vscode.SnippetString('gl.baseInstance')
	},
	{
		label: 'gl.drawIndex', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains index of the current draw call. (Vertex Shader)', 
		signature: 'in int32 gl.drawIndex;', insertText: new vscode.SnippetString('gl.drawIndex')
	},
	{
		label: 'gl.position', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains clip-space position of the current vertex. (Vertex Shader)', 
		signature: 'out float4 gl.position;', insertText: new vscode.SnippetString('gl.position')
	},
	{
		label: 'gl.pointSize', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains pixel size of the point being rasterized. (Vertex Shader)', 
		signature: 'out float gl.pointSize;', insertText: new vscode.SnippetString('gl.pointSize')
	},

	{
		label: 'gl.fragCoord', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the window-relative coordinates of the current fragment. (Fragment Shader)', 
		signature: 'in float4 gl.fragCoord;', insertText: new vscode.SnippetString('gl.fragCoord')
	},
	{
		label: 'gl.frontFacing', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Indicates whether a primitive is front or back facing. (Fragment Shader)', 
		signature: 'in bool gl.frontFacing;', insertText: new vscode.SnippetString('gl.frontFacing')
	},
	{
		label: 'gl.pointCoord', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the coordinate of a fragment within a point. (Fragment Shader)', 
		signature: 'in float2 gl.pointCoord;', insertText: new vscode.SnippetString('gl.pointCoord')
	},
	{
		label: 'gl.numSamples', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the total number of samples in the framebuffer. (Fragment Shader)', 
		signature: 'in int32 gl.numSamples;', insertText: new vscode.SnippetString('gl.numSamples')
	},
	{
		label: 'gl.sampleID', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Index of the current sample that this fragment is rasterized for. (Fragment Shader)', 
		signature: 'in int32 gl.sampleID;', insertText: new vscode.SnippetString('gl.sampleID')
	},
	{
		label: 'gl.samplePositions', kind: vscode.CompletionItemKind.Variable, 
		documentation: "Location of the current sample for the fragment within the pixel's area. (Fragment Shader)", 
		signature: 'in float2 gl.samplePositions;', insertText: new vscode.SnippetString('gl.samplePositions')
	},
	{
		label: 'gl.sampleMaskIn', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains a bitfield for the sample mask of the fragment being generated. (Fragment Shader)', 
		signature: 'in int32 gl.sampleMaskIn[];', insertText: new vscode.SnippetString('gl.sampleMaskIn[$1]')
	},
	{
		label: 'gl.helperInvocation', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Indicates whether a shader invocation is a helper invocation. (Fragment Shader)', 
		signature: 'in bool gl.helperInvocation;', insertText: new vscode.SnippetString('gl.helperInvocation')
	},
	
	{
		label: 'gl.fragDepth', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Establishes a depth value for the current fragment. (Fragment Shader)', 
		signature: 'out float gl.fragDepth;', insertText: new vscode.SnippetString('gl.fragDepth')
	},
	{
		label: 'gl.sampleMask', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Specifies the sample coverage mask for the current fragment. (Fragment Shader)', 
		signature: 'out int32 gl.sampleMask[];', insertText: new vscode.SnippetString('gl.sampleMask[$1]')
	},

	{
		label: 'gl.clipDistance', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the user-defined clipping planes. (Vertex and Fragment Shaders)', 
		signature: 'inout float gl.clipDistance[];', insertText: new vscode.SnippetString('gl.clipDistance[$1]')
	},
	{
		label: 'gl.cullDistance', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the user-defined culling distances. (Vertex and Fragment Shaders)', 
		signature: 'inout float gl.cullDistance[];', insertText: new vscode.SnippetString('gl.cullDistance[$1]')
	},
	{
		label: 'gl.primitiveID', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the index of the current primitive. (Fragment and Mesh Shaders)', 
		signature: 'inout int32 gl.primitiveID;', insertText: new vscode.SnippetString('gl.primitiveID')
	},
	{
		label: 'gl.layerID', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains a layer index of the current fragment. (Fragment and Mesh Shaders)', 
		signature: 'inout int32 gl.layerID;', insertText: new vscode.SnippetString('gl.layerID')
	},
	{
		label: 'gl.viewportIndex', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains a viewport index of the currents fragment. (Fragment and Mesh Shaders)', 
		signature: 'inout int32 gl.viewportIndex;', insertText: new vscode.SnippetString('gl.viewportIndex')
	},

	{
		label: 'gl.workGroupSize', kind: vscode.CompletionItemKind.Variable, 
		documentation: "Contains the `localSize` of the workgroup. (Compute Shader)", 
		signature: 'in uint3 gl.workGroupSize;', insertText: new vscode.SnippetString('gl.workGroupSize')
	},
	{
		label: 'gl.numWorkGroups', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the number of workgroups that have been dispatched. (Compute Shader)', 
		signature: 'in uint3 gl.numWorkGroups;', insertText: new vscode.SnippetString('gl.numWorkGroups')
	},
	{
		label: 'gl.workGroupID', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the index of the workgroup currently being operated on. (Compute Shader)', 
		signature: 'in uint3 gl.workGroupID;', insertText: new vscode.SnippetString('gl.workGroupID')
	},
	{
		label: 'gl.localInvocationID', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the index of work item currently being operated on. (Compute Shader)', 
		signature: 'in uint3 gl.localInvocationID;', insertText: new vscode.SnippetString('gl.localInvocationID')
	},
	{
		label: 'gl.globalInvocationID', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the global index of work item currently being operated on. (Compute Shader) <br>`gl.globalInvocationID = gl.workGroupID * gl.workGroupSize + gl.localInvocationID;`', 
		signature: 'in uint3 gl.globalInvocationID;', insertText: new vscode.SnippetString('gl.globalInvocationID')
	},
	{
		label: 'gl.localInvocationIndex', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the local linear index of work item currently being operated on. (Compute Shader)', 
		signature: 'in uint32 gl.localInvocationIndex;', insertText: new vscode.SnippetString('gl.localInvocationIndex')
	},
];
const builtinMap = new Map(builtins.map(item => [item.label, item]));

function activate(context)
{
    console.log('GSL extension activated');

	let hoverProvider = vscode.languages.registerHoverProvider('gsl',
	{
		provideHover(document, position, token)
		{
			const wordRange = document.getWordRangeAtPosition(position);
			if (!wordRange) return undefined;

			const word = document.getText(wordRange);
			if (builtinMap.has(word))
			{
				const builtin = builtinMap.get(word);
				const markdown = new vscode.MarkdownString();
				markdown.appendCodeblock(`${builtin.signature}`, 'gsl');
				markdown.appendMarkdown('\n\n' + builtin.documentation);
				markdown.supportHtml = true; markdown.isTrusted = true;
				return new vscode.Hover(markdown, wordRange);
			}
			return undefined;
		}
	});

	let completionProvider = vscode.languages.registerCompletionItemProvider('gsl',
	{
		provideCompletionItems(document, position, token, context)
		{
			return builtins.map(itemData =>
			{
				const item = new vscode.CompletionItem(itemData.label, itemData.kind);
				item.detail = itemData.signature;
				item.documentation = new vscode.MarkdownString(itemData.documentation);
				item.insertText = itemData.insertText;
				return item;
			});
		}
	}, '');

	context.subscriptions.push(hoverProvider, completionProvider);
}

module.exports =
{
	activate
};