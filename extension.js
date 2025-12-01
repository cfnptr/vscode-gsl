const vscode = require('vscode');

const builtins =
[
	{
		label: '#defines', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Defines constant global shader preprocessor macros.', 
		signature: '#define NAME value', insertText: new vscode.SnippetString('#define $1 $2')
	},
	{
		label: '#include', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Includes the contents of another GSL source file.<br>Used to modularize shader code by reusing common functions or definitions.', 
		signature: '#include "file-name.gsl"', insertText: new vscode.SnippetString('#include "$1"')
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
		signature: 'discard;', insertText: new vscode.SnippetString('discard')
	},

	{
		label: 'bool', kind: vscode.CompletionItemKind.Class, 
		documentation: 'Conditional type, values may be either true or false.', 
		signature: 'bool', insertText: new vscode.SnippetString('bool')
	},
	{
		label: 'int8', kind: vscode.CompletionItemKind.Class, 
		documentation: "A signed 8-bit integer. (-128 to 127)", 
		signature: 'int8', insertText: new vscode.SnippetString('int8')
	},
	{
		label: 'uint8', kind: vscode.CompletionItemKind.Class, 
		documentation: "An unsigned 8-bit integer. (0 to 255)", 
		signature: 'uint8', insertText: new vscode.SnippetString('uint8')
	},
	{
		label: 'int16', kind: vscode.CompletionItemKind.Class, 
		documentation: "A signed 16-bit integer. (-32,768 to 32,767)", 
		signature: 'int16', insertText: new vscode.SnippetString('int16')
	},
	{
		label: 'uint16', kind: vscode.CompletionItemKind.Class, 
		documentation: "An unsigned 16-bit integer. (0 to 65,535)", 
		signature: 'uint16', insertText: new vscode.SnippetString('uint16')
	},
	{
		label: 'int32', kind: vscode.CompletionItemKind.Class, 
		documentation: "A signed, [two's complement](https://en.wikipedia.org/wiki/Two%27s_complement), 32-bit integer. (-2,147,483,648 to 2,147,483,647)", 
		signature: 'int32', insertText: new vscode.SnippetString('int32')
	},
	{
		label: 'uint32', kind: vscode.CompletionItemKind.Class, 
		documentation: "An unsigned, [two's complement](https://en.wikipedia.org/wiki/Two%27s_complement), 32-bit integer. (0 to 4,294,967,295)", 
		signature: 'uint32', insertText: new vscode.SnippetString('uint32')
	},
	{
		label: 'int64', kind: vscode.CompletionItemKind.Class, 
		documentation: "A signed 64-bit integer. (-9,223,372,036,854,775,808 to 9,223,372,036,854,775,807)", 
		signature: 'int64', insertText: new vscode.SnippetString('int64')
	},
	{
		label: 'uint64', kind: vscode.CompletionItemKind.Class, 
		documentation: "An unsigned 64-bit integer. (0 to 18,446,744,073,709,551,615)", 
		signature: 'uint64', insertText: new vscode.SnippetString('uint64')
	},
	{
		label: 'half', kind: vscode.CompletionItemKind.Class, 
		documentation: 'An [IEEE-754](https://en.wikipedia.org/wiki/IEEE_754) half-precision 16-bit floating point number.', 
		signature: 'half', insertText: new vscode.SnippetString('half')
	},
	{
		label: 'float', kind: vscode.CompletionItemKind.Class, 
		documentation: 'An [IEEE-754](https://en.wikipedia.org/wiki/IEEE_754) single-precision 32-bit floating point number.', 
		signature: 'float', insertText: new vscode.SnippetString('float')
	},
	{
		label: 'double', kind: vscode.CompletionItemKind.Class, 
		documentation: 'An [IEEE-754](https://en.wikipedia.org/wiki/IEEE_754) double-precision 64-bit floating point number.', 
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
		label: 'sbyte2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2-component vector of 8-bit signed integers.', 
		signature: 'sbyte2(int8 x, int8 y)', insertText: new vscode.SnippetString('sbyte2($1, $2)')
	},
	{
		label: 'sbyte3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3-component vector of 8-bit signed integers.', 
		signature: 'sbyte3(int8 x, int8 y, int8 z)', insertText: new vscode.SnippetString('sbyte3($1, $2, $3)')
	},
	{
		label: 'sbyte4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 4-component vector of 8-bit signed integers.', 
		signature: 'sbyte4(int8 x, int8 y, int8 z, int8 w)', insertText: new vscode.SnippetString('sbyte4($1, $2, $3, $4)')
	},
	{
		label: 'byte2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2-component vector of 8-bit unsigned integers.', 
		signature: 'byte2(uint8 x, uint8 y)', insertText: new vscode.SnippetString('byte2($1, $2)')
	},
	{
		label: 'byte3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3-component vector of 8-bit unsigned integers.', 
		signature: 'byte3(uint8 x, uint8 y, uint8 z)', insertText: new vscode.SnippetString('byte3($1, $2, $3)')
	},
	{
		label: 'byte4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 4-component vector of 8-bit unsigned integers.', 
		signature: 'byte4(uint8 x, uint8 y, uint8 z, uint8 w)', insertText: new vscode.SnippetString('byte4($1, $2, $3, $4)')
	},

	{
		label: 'short2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2-component vector of 16-bit signed integers.', 
		signature: 'short2(int16 x, int16 y)', insertText: new vscode.SnippetString('short2($1, $2)')
	},
	{
		label: 'short3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3-component vector of 16-bit signed integers.', 
		signature: 'short3(int16 x, int16 y, int16 z)', insertText: new vscode.SnippetString('short3($1, $2, $3)')
	},
	{
		label: 'short4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 4-component vector of 16-bit signed integers.', 
		signature: 'short4(int16 x, int16 y, int16 z, int16 w)', insertText: new vscode.SnippetString('short4($1, $2, $3, $4)')
	},
	{
		label: 'ushort2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2-component vector of 16-bit unsigned integers.', 
		signature: 'ushort2(uint16 x, uint16 y)', insertText: new vscode.SnippetString('ushort2($1, $2)')
	},
	{
		label: 'ushort3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3-component vector of 16-bit unsigned integers.', 
		signature: 'ushort3(uint16 x, uint16 y, uint16 z)', insertText: new vscode.SnippetString('ushort3($1, $2, $3)')
	},
	{
		label: 'ushort4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 4-component vector of 16-bit unsigned integers.', 
		signature: 'ushort4(uint16 x, uint16 y, uint16 z, uint16 w)', insertText: new vscode.SnippetString('ushort4($1, $2, $3, $4)')
	},

	{
		label: 'long2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2-component vector of 64-bit signed integers.', 
		signature: 'long2(int64 x, int64 y)', insertText: new vscode.SnippetString('long2($1, $2)')
	},
	{
		label: 'long3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3-component vector of 64-bit signed integers.', 
		signature: 'long3(int64 x, int64 y, int64 z)', insertText: new vscode.SnippetString('long3($1, $2, $3)')
	},
	{
		label: 'long4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 4-component vector of 64-bit signed integers.', 
		signature: 'long4(int64 x, int64 y, int64 z, int64 w)', insertText: new vscode.SnippetString('long4($1, $2, $3, $4)')
	},
	{
		label: 'ulong2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2-component vector of 64-bit unsigned integers.', 
		signature: 'ulong2(uint64 x, uint64 y)', insertText: new vscode.SnippetString('ulong2($1, $2)')
	},
	{
		label: 'ulong3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3-component vector of 64-bit unsigned integers.', 
		signature: 'ulong3(uint64 x, uint64 y, uint64 z)', insertText: new vscode.SnippetString('ulong3($1, $2, $3)')
	},
	{
		label: 'ulong4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 4-component vector of 64-bit unsigned integers.', 
		signature: 'ulong4(uint64 x, uint64 y, uint64 z, uint64 w)', insertText: new vscode.SnippetString('ulong4($1, $2, $3, $4)')
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
		label: 'half2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2-component vector of half-precision 16-bit floating-point numbers.', 
		signature: 'half2(half x, half y)', insertText: new vscode.SnippetString('half2($1, $2)')
	},
	{
		label: 'half3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3-component vector of half-precision 16-bit floating-point numbers.', 
		signature: 'half3(half x, half y, half z)', insertText: new vscode.SnippetString('half3($1, $2, $3)')
	},
	{
		label: 'half4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 4-component vector of half-precision 16-bit floating-point numbers.', 
		signature: 'half4(half x, half y, half z, half w)', insertText: new vscode.SnippetString('half4($1, $2, $3, $4)')
	},

	{
		label: 'float2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2-component vector of single-precision 32-bit floating-point numbers.', 
		signature: 'float2(float x, float y)', insertText: new vscode.SnippetString('float2($1, $2)')
	},
	{
		label: 'float3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3-component vector of single-precision 32-bit floating-point numbers.', 
		signature: 'float3(float x, float y, float z)', insertText: new vscode.SnippetString('float3($1, $2, $3)')
	},
	{
		label: 'float4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 4-component vector of single-precision 32-bit floating-point numbers.', 
		signature: 'float4(float x, float y, float z, float w)', insertText: new vscode.SnippetString('float4($1, $2, $3, $4)')
	},

	{
		label: 'double2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2-component vector of double-precision 64-bit floating-point numbers.', 
		signature: 'double2(double x, double y)', insertText: new vscode.SnippetString('double2($1, $2)')
	},
	{
		label: 'double3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3-component vector of double-precision 64-bit floating-point numbers.', 
		signature: 'double3(double x, double y, double z)', insertText: new vscode.SnippetString('double3($1, $2, $3)')
	},
	{
		label: 'double4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 4-component vector of double-precision 64-bit floating-point numbers.', 
		signature: 'double4(double x, double y, double z, double w)', insertText: new vscode.SnippetString('double4($1, $2, $3, $4)')
	},

	{
		label: 'half2x2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 2 columns and 2 rows of half-precision 16-bit floating-point numbers.', 
		signature: 'half2x2(half2 c0, half2 c1)', insertText: new vscode.SnippetString('half2x2($1, $2)')
	},
	{
		label: 'half2x3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 2 columns and 3 rows of half-precision 16-bit floating-point numbers.', 
		signature: 'halfx3(half3 c0, half3 c1)', insertText: new vscode.SnippetString('half2x3($1, $2)')
	},
	{
		label: 'half2x4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 2 columns and 4 rows of half-precision 16-bit floating-point numbers.', 
		signature: 'half2x4(half4 c0, half4 c1)', insertText: new vscode.SnippetString('half2x4($1, $2)')
	},
	{
		label: 'half3x2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 3 columns and 2 rows of half-precision 16-bit floating-point numbers.', 
		signature: 'half3x2(half2 c0, half2 c1, half2 c2)', insertText: new vscode.SnippetString('half3x2($1, $2, $3)')
	},
	{
		label: 'half3x3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 3 columns and 3 rows of half-precision 16-bit floating-point numbers.', 
		signature: 'half3x3(half3 c0, half3 c1, half3 c2)', insertText: new vscode.SnippetString('half3x3($1, $2, $3)')
	},
	{
		label: 'half3x4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 3 columns and 4 rows of half-precision 16-bit floating-point numbers.', 
		signature: 'half3x4(half4 c0, half4 c1, half4 c2)', insertText: new vscode.SnippetString('half3x4($1, $2, $3)')
	},
	{
		label: 'half4x2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 4 columns and 2 rows of half-precision 16-bit floating-point numbers.', 
		signature: 'half4x2(half2 c0, half2 c1, half2 c2, half2 c3)', insertText: new vscode.SnippetString('half4x2($1, $2, $3, $4)')
	},
	{
		label: 'half4x3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 4 columns and 3 rows of half-precision 16-bit floating-point numbers.', 
		signature: 'half4x3(half3 c0, half3 c1, half3 c2, half3 c3)', insertText: new vscode.SnippetString('half4x3($1, $2, $3)')
	},
	{
		label: 'half4x4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 4 columns and 4 rows of half-precision 16-bit floating-point numbers.', 
		signature: 'half4x4(half4 c0, half4 c1, half4 c2, half4 c3)', insertText: new vscode.SnippetString('half4x4($1, $2, $3)')
	},

	{
		label: 'float2x2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 2 columns and 2 rows of single-precision 32-bit floating-point numbers.', 
		signature: 'float2x2(float2 c0, float2 c1)', insertText: new vscode.SnippetString('float2x2($1, $2)')
	},
	{
		label: 'float2x3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 2 columns and 3 rows of single-precision 32-bit floating-point numbers.', 
		signature: 'float2x3(float3 c0, float3 c1)', insertText: new vscode.SnippetString('float2x3($1, $2)')
	},
	{
		label: 'float2x4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 2 columns and 4 rows of single-precision 32-bit floating-point numbers.', 
		signature: 'float2x4(float4 c0, float4 c1)', insertText: new vscode.SnippetString('float2x4($1, $2)')
	},
	{
		label: 'float3x2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 3 columns and 2 rows of single-precision 32-bit floating-point numbers.', 
		signature: 'float3x2(float2 c0, float2 c1, float2 c2)', insertText: new vscode.SnippetString('float3x2($1, $2, $3)')
	},
	{
		label: 'float3x3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 3 columns and 3 rows of single-precision 32-bit floating-point numbers.', 
		signature: 'float3x3(float3 c0, float3 c1, float3 c2)', insertText: new vscode.SnippetString('float3x3($1, $2, $3)')
	},
	{
		label: 'float3x4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 3 columns and 4 rows of single-precision 32-bit floating-point numbers.', 
		signature: 'float3x4(float4 c0, float4 c1, float4 c2)', insertText: new vscode.SnippetString('float3x4($1, $2, $3)')
	},
	{
		label: 'float4x2', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 4 columns and 2 rows of single-precision 32-bit floating-point numbers.', 
		signature: 'float4x2(float2 c0, float2 c1, float2 c2, float2 c3)', insertText: new vscode.SnippetString('float4x2($1, $2, $3, $4)')
	},
	{
		label: 'float4x3', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 4 columns and 3 rows of single-precision 32-bit floating-point numbers.', 
		signature: 'float4x3(float3 c0, float3 c1, float3 c2, float3 c3)', insertText: new vscode.SnippetString('float4x3($1, $2, $3)')
	},
	{
		label: 'float4x4', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A matrix with 4 columns and 4 rows of single-precision 32-bit floating-point numbers.', 
		signature: 'float4x4(float4 c0, float4 c1, float4 c2, float4 c3)', insertText: new vscode.SnippetString('float4x4($1, $2, $3)')
	},

	{
		label: 'f8', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 8-bit normalized uint as float vertex shader input attribute type.', 
		signature: 'in Type vs.name : f8;', insertText: new vscode.SnippetString('f8;')
	},
	{
		label: 'f16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit normalized uint as float vertex shader input attribute type.', 
		signature: 'in Type vs.name : f16;', insertText: new vscode.SnippetString('f16;')
	},
	{
		label: 'f32', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 32-bit floating-point vertex shader input attribute type.', 
		signature: 'in Type vs.name : f32;', insertText: new vscode.SnippetString('f32;')
	},
	{
		label: 'i8', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 8-bit signed integer vertex shader input attribute type.', 
		signature: 'in Type vs.name : i8;', insertText: new vscode.SnippetString('i8;')
	},
	{
		label: 'i16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit signed integer vertex shader input attribute type.', 
		signature: 'in Type vs.name : i16;', insertText: new vscode.SnippetString('i16;')
	},
	{
		label: 'i32', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 32-bit signed integer vertex shader input attribute type.', 
		signature: 'in Type vs.name : i32;', insertText: new vscode.SnippetString('i32;')
	},
	{
		label: 'u8', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 8-bit unsigned integer vertex shader input attribute type.', 
		signature: 'in Type vs.name : u8;', insertText: new vscode.SnippetString('u8;')
	},
	{
		label: 'u16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit unsigned integer vertex shader input attribute type.', 
		signature: 'in Type vs.name : u16;', insertText: new vscode.SnippetString('u16;')
	},
	{
		label: 'u32', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 32-bit unsigned integer vertex shader input attribute type.', 
		signature: 'in Type vs.name : u32;', insertText: new vscode.SnippetString('u32;')
	},

	{
		label: 'uintR8', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 8-bit unsigned integer (red only channel) image format.', 
		signature: 'uniform Image name : uintR8;', insertText: new vscode.SnippetString('uintR8;')
	},
	{
		label: 'uintR8G8', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 8-bit unsigned integer (red and green channel) image format.', 
		signature: 'uniform Image name : uintR8G8;', insertText: new vscode.SnippetString('uintR8G8;')
	},
	{
		label: 'uintR8G8B8A8', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 8-bit unsigned integer (red, green, blue, alpha channel) image format.', 
		signature: 'uniform Image name : uintR8G8B8A8;', insertText: new vscode.SnippetString('uintR8G8B8A8;')
	},
	{
		label: 'uintR16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit unsigned integer (red only channel) image format.', 
		signature: 'uniform Image name : uintR16;', insertText: new vscode.SnippetString('uintR16;')
	},
	{
		label: 'uintR16G16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit unsigned integer (red and green channel) image format.', 
		signature: 'uniform Image name : uintR16G16;', insertText: new vscode.SnippetString('uintR16G16;')
	},
	{
		label: 'uintR16G16B16A16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit unsigned integer (red, green, blue, alpha channel) image format.', 
		signature: 'uniform Image name : uintR16G16B16A16;', insertText: new vscode.SnippetString('uintR16G16B16A16;')
	},
	{
		label: 'uintR32', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 32-bit unsigned integer (red only channel) image format.', 
		signature: 'uniform Image name : uintR32;', insertText: new vscode.SnippetString('uintR32;')
	},
	{
		label: 'uintR32G32', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 32-bit unsigned integer (red and green channel) image format.', 
		signature: 'uniform Image name : uintR32G32;', insertText: new vscode.SnippetString('uintR32G32;')
	},
	{
		label: 'uintR32G32B32A32', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 32-bit unsigned integer (red, green, blue, alpha channel) image format.', 
		signature: 'uniform Image name : uintR32G32B32A32;', insertText: new vscode.SnippetString('uintR32G32B32A32;')
	},
	{
		label: 'uintA2R10G10B10', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'An unsigned integer (2-bit alpha, 10-bit red/green/blue channel) format.', 
		signature: 'uniform Image name : uintA2R10G10B10;', insertText: new vscode.SnippetString('uintA2R10G10B10;')
	},

	{
		label: 'sintR8', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 8-bit signed integer (red only channel) image format.', 
		signature: 'uniform Image name : sintR8;', insertText: new vscode.SnippetString('sintR8;')
	},
	{
		label: 'sintR8G8', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 8-bit signed integer (red and green channel) image format.', 
		signature: 'uniform Image name : sintR8G8;', insertText: new vscode.SnippetString('sintR8G8;')
	},
	{
		label: 'sintR8G8B8A8', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 8-bit signed integer (red, green, blue, alpha channel) image format.', 
		signature: 'uniform Image name : sintR8G8B8A8;', insertText: new vscode.SnippetString('sintR8G8B8A8;')
	},
	{
		label: 'sintR16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit signed integer (red only channel) image format.', 
		signature: 'uniform Image name : sintR16;', insertText: new vscode.SnippetString('sintR16;')
	},
	{
		label: 'sintR16G16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit signed integer (red and green channel) image format.', 
		signature: 'uniform Image name : sintR16G16;', insertText: new vscode.SnippetString('sintR16G16;')
	},
	{
		label: 'sintR16G16B16A16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit signed integer (red, green, blue, alpha channel) image format.', 
		signature: 'uniform Image name : sintR16G16B16A16;', insertText: new vscode.SnippetString('sintR16G16B16A16;')
	},
	{
		label: 'sintR32', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 32-bit signed integer (red only channel) image format.', 
		signature: 'uniform Image name : sintR32;', insertText: new vscode.SnippetString('sintR32;')
	},
	{
		label: 'sintR32G32', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 32-bit signed integer (red and green channel) image format.', 
		signature: 'uniform Image name : sintR32G32;', insertText: new vscode.SnippetString('sintR32G32;')
	},
	{
		label: 'sintR32G32B32A32', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 32-bit signed integer (red, green, blue, alpha channel) image format.', 
		signature: 'uniform Image name : sintR32G32B32A32;', insertText: new vscode.SnippetString('sintR32G32B32A32;')
	},

	{
		label: 'unormR8', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 8-bit normalized uint as float (red only channel) image format. [0.0, 1.0]', 
		signature: 'uniform Image name : unormR8;', insertText: new vscode.SnippetString('unormR8;')
	},
	{
		label: 'unormR8G8', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 8-bit normalized uint as float (red and green channel) image format. [0.0, 1.0]', 
		signature: 'uniform Image name : unormR8G8;', insertText: new vscode.SnippetString('unormR8G8;')
	},
	{
		label: 'unormR8G8B8A8', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 8-bit normalized uint as float (red, green, blue, alpha channel) image format. [0.0, 1.0]', 
		signature: 'uniform Image name : unormR8G8B8A8;', insertText: new vscode.SnippetString('unormR8G8B8A8;')
	},
	{
		label: 'unormR16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit normalized uint as float (red only channel) image format. [0.0, 1.0]', 
		signature: 'uniform Image name : unormR16;', insertText: new vscode.SnippetString('unormR16;')
	},
	{
		label: 'unormR16G16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit normalized uint as float (red and green channel) image format. [0.0, 1.0]', 
		signature: 'uniform Image name : unormR16G16;', insertText: new vscode.SnippetString('unormR16G16;')
	},
	{
		label: 'unormR16G16B16A16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit normalized uint as float (red, green, blue, alpha channel) image format. [0.0, 1.0]', 
		signature: 'uniform Image name : unormR16G16B16A16;', insertText: new vscode.SnippetString('unormR16G16B16A16;')
	},
	{
		label: 'unormA2R10G10B10', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'An normalized uint as float (2-bit alpha, 10-bit red/green/blue channel) format. [0.0, 1.0]', 
		signature: 'uniform Image name : unormA2R10G10B10;', insertText: new vscode.SnippetString('unormA2R10G10B10;')
	},

	{
		label: 'snormR8', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 8-bit normalized int as float (red only channel) image format. [-1.0, 1.0]', 
		signature: 'uniform Image name : snormR8;', insertText: new vscode.SnippetString('snormR8;')
	},
	{
		label: 'snormR8G8', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 8-bit normalized int as float (red and green channel) image format. [-1.0, 1.0]', 
		signature: 'uniform Image name : snormR8G8;', insertText: new vscode.SnippetString('snormR8G8;')
	},
	{
		label: 'snormR8G8B8A8', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 8-bit normalized int as float (red, green, blue, alpha channel) image format. [-1.0, 1.0]', 
		signature: 'uniform Image name : snormR8G8B8A8;', insertText: new vscode.SnippetString('snormR8G8B8A8;')
	},
	{
		label: 'snormR16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit normalized int as float (red only channel) image format. [-1.0, 1.0]', 
		signature: 'uniform Image name : snormR16;', insertText: new vscode.SnippetString('snormR16;')
	},
	{
		label: 'snormR16G16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit normalized int as float (red and green channel) image format. [-1.0, 1.0]', 
		signature: 'uniform Image name : snormR16G16;', insertText: new vscode.SnippetString('snormR16G16;')
	},
	{
		label: 'snormR16G16B16A16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit normalized int as float (red, green, blue, alpha channel) image format. [-1.0, 1.0]', 
		signature: 'uniform Image name : snormR16G16B16A16;', insertText: new vscode.SnippetString('snormR16G16B16A16;')
	},

	{
		label: 'sfloatR16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit signed floating point (red only channel) image format.', 
		signature: 'uniform Image name : sfloatR16;', insertText: new vscode.SnippetString('sfloatR16;')
	},
	{
		label: 'sfloatR16G16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit signed floating point (red and green channel) image format.', 
		signature: 'uniform Image name : sfloatR16G16;', insertText: new vscode.SnippetString('sfloatR16G16;')
	},
	{
		label: 'sfloatR16G16B16A16', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 16-bit signed floating point (red, green, blue, alpha channel) image format.', 
		signature: 'uniform Image name : sfloatR16G16B16A16;', insertText: new vscode.SnippetString('sfloatR16G16B16A16;')
	},
	{
		label: 'sfloatR32', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 32-bit signed floating point (red only channel) image format.', 
		signature: 'uniform Image name : sfloatR32;', insertText: new vscode.SnippetString('sfloatR32;')
	},
	{
		label: 'sfloatR32G32', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 32-bit signed floating point (red and green channel) image format.', 
		signature: 'uniform Image name : sfloatR32G32;', insertText: new vscode.SnippetString('sfloatR32G32;')
	},
	{
		label: 'sfloatR32G32B32A32', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'A 32-bit signed floating point (red, green, blue, alpha channel) image format.', 
		signature: 'uniform Image name : sfloatR32G32B32A32;', insertText: new vscode.SnippetString('sfloatR32G32B32A32;')
	},
	{
		label: 'ufloatB10G11R11', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'An unsigned floating point (10-bit blue, 11-bit green, 10-bit red channel) image format.', 
		signature: 'uniform Image name : ufloatB10G11R11;', insertText: new vscode.SnippetString('ufloatB10G11R11;')
	},

	{
		label: 'in', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares an input variable to the shader stage or function.', 
		signature: 'in Type name;', insertText: new vscode.SnippetString('in ')
	},
	{
		label: 'out', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares an output variable from the shader stage or function.', 
		signature: '<invariant> out Type name;', insertText: new vscode.SnippetString('out ')
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
		signature: 'buffer <readonly|writeonly|coherent|volatile|restrict|std430> set0 BufferName\n{\n\t...\n} name;', insertText: new vscode.SnippetString('buffer ')
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
		documentation: 'Indicates that memory pointed to by the variable is not aliased <br>(overlapped) with any other memory or variable used in the shader.', 
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
		label: 'invariant', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Ensures consistent rasterization results across shader stages <br>by preventing small differences in floating-point calculations.', 
		signature: 'invariant out Type name;', insertText: new vscode.SnippetString('invariant ')
	},
	{
		label: 'mutable', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Indicates that texture sampler state can be changed at runtime.', 
		signature: 'uniform mutable Sampler name;', insertText: new vscode.SnippetString('mutable ')
	},
	{
		label: 'std430', kind: vscode.CompletionItemKind.Keyword,
		documentation: 'Indicates that buffer layout becomes std430 instead of scalar.',
		signature: 'buffer std430 set0 BufferName\n{\n\t...\n} name;', insertText: new vscode.SnippetString('std430 ')
	},
	{
		label: 'reference', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares that the type can be used as a buffer reference.', 
		signature: 'buffer reference BufferName\n{\n\t...\n};', insertText: new vscode.SnippetString('reference ')
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
		label: 'earlyFragmentTests', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Perform early depth and stencil tests before executing the fragment shader.', 
		signature: 'earlyFragmentTests in;', insertText: new vscode.SnippetString('earlyFragmentTests in;')
	},
	{
		label: 'localSize', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Specifies the size of a compute shader workgroup in the X, Y, and Z dimensions. <br>The values determine how many shader invocations will be launched per workgroup.', 
		signature: 'localSize = x, y, z;', insertText: new vscode.SnippetString('localSize = $1, $2, $3;')
	},
	{
		label: 'spec', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares a specialization constant that can be overridden at pipeline creation time. <br>Specialization constants behave like regular constants in shaders but allow changing their values without recompiling the shader.', 
		signature: 'spec const Type NAME = ...;', insertText: new vscode.SnippetString('spec const $1 $2 = $3;')
	},
	{
		label: 'pipelineState', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares a [pipeline state](https://github.com/cfnptr/garden/blob/main/docs/GSL.md#pipeline-state) that configures fixed-function pipeline behavior.', 
		signature: 'pipelineState\n{\n\t...\n}', insertText: new vscode.SnippetString('pipelineState\n{\n\t$1\n}')
	},
	{
		label: 'pushConstants', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Defines a block of fast, read-only memory that can be updated <br>frequently from the host without rebinding descriptor sets.', 
		signature: 'uniform pushConstants\n{\n\t...\n} pc;', insertText: new vscode.SnippetString('pushConstants ')
	},

	{
		label: '#variantCount', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Total variant count of the shader.', 
		signature: '#variantCount x', insertText: new vscode.SnippetString('#variantCount $1')
	},
	{
		label: '#attributeOffset', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Adds offset to the input vertex attributes in bytes.', 
		signature: '#attributeOffset x', insertText: new vscode.SnippetString('#attributeOffset $1')
	},
	{
		label: '#attachmentOffset', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Adds offset to the `subpassInput` index.', 
		signature: '#attachmentOffset x', insertText: new vscode.SnippetString('#attachmentOffset $1')
	},
	{
		label: '#rayRecursionDepth', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Maximum number of levels of ray recursion allowed in a trace command.', 
		signature: '#rayRecursionDepth x', insertText: new vscode.SnippetString('#rayRecursionDepth $1')
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
		documentation: 'A 1D texture sampler used to sample unsigned integer data in shaders.', 
		signature: 'uniform set0 usampler1D\n{\n\t...\n} name;', insertText: new vscode.SnippetString('usampler1D ')
	},
	{
		label: 'usampler2D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D texture sampler used to sample unsigned integer data in shaders.', 
		signature: 'uniform set0 usampler2D\n{\n\t...\n} name;', insertText: new vscode.SnippetString('usampler2D ')
	},
	{
		label: 'usampler3D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3D texture sampler used to sample unsigned integer data in shaders.', 
		signature: 'uniform set0 usampler3D\n{\n\t...\n} name;', insertText: new vscode.SnippetString('usampler3D ')
	},
	{
		label: 'usamplerCube', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A cubemap texture sampler used to sample unsigned integer data in shaders.', 
		signature: 'uniform set0 usamplerCube\n{\n\t...\n} name;', insertText: new vscode.SnippetString('usamplerCube ')
	},
	{
		label: 'usampler1DArray', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 1D texture array sampler used to sample unsigned integer data in shaders.', 
		signature: 'uniform set0 usampler1DArray\n{\n\t...\n} name;', insertText: new vscode.SnippetString('usampler1DArray ')
	},
	{
		label: 'usampler2DArray', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D texture array sampler used to sample unsigned integer data in shaders.', 
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
		documentation: 'A 1D image that can be read from or written to in shaders, with floating point data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 image1D name : Format;', insertText: new vscode.SnippetString('image1D ')
	},
	{
		label: 'image2D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D image that can be read from or written to in shaders, with floating point data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 image2D name : Format;', insertText: new vscode.SnippetString('image2D ')
	},
	{
		label: 'image3D', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 3D image that can be read from or written to in shaders, with floating point data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 image3D name : Format;', insertText: new vscode.SnippetString('image3D ')
	},
	{
		label: 'imageCube', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A cubemap image that can be read from or written to in shaders, with floating point data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 imageCube name : Format;', insertText: new vscode.SnippetString('imageCube ')
	},
	{
		label: 'image1DArray', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 1D image array that can be read from or written to in shaders, with floating point data.', 
		signature: 'uniform <readonly|writeonly|coherent|volatile|restrict> set0 image1DArray name : Format;', insertText: new vscode.SnippetString('image1DArray ')
	},
	{
		label: 'image2DArray', kind: vscode.CompletionItemKind.Class, 
		documentation: 'A 2D image array that can be read from or written to in shaders, with floating point data.', 
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
		signature: 'uniform set0 subpassInput name;', insertText: new vscode.SnippetString('subpassInput ')
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
		signature: 'break;', insertText: new vscode.SnippetString('break')
	},
	{
		label: 'continue', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Skips the current loop iteration and continues with the next.', 
		signature: 'continue;', insertText: new vscode.SnippetString('continue')
	},
	{
		label: 'return', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Returns a value and exits the current function.', 
		signature: 'return value;', insertText: new vscode.SnippetString('return')
	},

	{
		label: 'printf', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Writes debug messages directly from shaders.', 
		signature: 'void printf(String fmt, ...);', insertText: new vscode.SnippetString('printf($1)')
	},

	{
		label: 'abs', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return the absolute value of the `x`.',
		signature: 'Type abs(Type x);', insertText: new vscode.SnippetString('abs($1)')
	},
	{
		label: 'sign', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Extract the sign of the `x`.',
		signature: 'Type sign(Type x);', insertText: new vscode.SnippetString('sign($1)')
	},
	{
		label: 'exp', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return the natural exponentiation of the `x`.',
		signature: 'FloatX exp(FloatX x);', insertText: new vscode.SnippetString('exp($1)')
	},
	{
		label: 'exp2', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return 2 raised to the power of the `x`.',
		signature: 'FloatX exp2(FloatX x);', insertText: new vscode.SnippetString('exp2($1)')
	},
	{
		label: 'log', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return the natural logarithm of the `x`.',
		signature: 'FloatX log(FloatX x);', insertText: new vscode.SnippetString('log($1)')
	},
	{
		label: 'log2', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return the base 2 logarithm of the `x`.',
		signature: 'FloatX log2(FloatX x);', insertText: new vscode.SnippetString('log2($1)')
	},
	{
		label: 'pow', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return the value of the `x` raised to the power of the `y`. [r = x ^ y]',
		signature: 'FloatX pow(FloatX x, FloatX y);', insertText: new vscode.SnippetString('pow($1, $2)')
	},
	{
		label: 'sqrt', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return the square root of the `x`.',
		signature: 'FloatX sqrt(FloatX x);', insertText: new vscode.SnippetString('sqrt($1)')
	},
	{
		label: 'inversesqrt', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return the inverse of the square root of the `x`. [r = 1.0 / sqrt(x)]',
		signature: 'FloatX inversesqrt(FloatX x);', insertText: new vscode.SnippetString('inversesqrt($1)')
	},
	{
		label: 'fma', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a fused multiply-add operation. [r = a * b + c]',
		signature: 'FloatX fma(FloatX a, FloatX b, FloatX c);', insertText: new vscode.SnippetString('fma($1, $2, $3)')
	},
	{
		label: 'mod', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Compute value of the `x` modulo `y`. [r = x - y * floor(x / y)]',
		signature: 'FloatX mod(FloatX x, FloatX y);', insertText: new vscode.SnippetString('mod($1, $2, $3)')
	},
	{
		label: 'modf', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Separate the `x` into its `integer` and fractional components.',
		signature: 'FloatX modf(FloatX x, out FloatX integer);', insertText: new vscode.SnippetString('modf($1, $2, $3)')
	},

	{
		label: 'floor', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Find the nearest integer less than or equal to the `x`.',
		signature: 'FloatX floor(FloatX x);', insertText: new vscode.SnippetString('floor($1)')
	},
	{
		label: 'ceil', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Find the nearest integer that is greater than or equal to the `x`.',
		signature: 'FloatX ceil(FloatX x);', insertText: new vscode.SnippetString('ceil($1)')
	},
	{
		label: 'round', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Find the nearest integer to the `x`.',
		signature: 'FloatX round(FloatX x);', insertText: new vscode.SnippetString('round($1)')
	},
	{
		label: 'roundEven', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Find the nearest even integer to the `x`.',
		signature: 'FloatX roundEven(FloatX x);', insertText: new vscode.SnippetString('roundEven($1)')
	},
	{
		label: 'fract', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Compute the fractional part of the `x`. [r = x - floor(x)]',
		signature: 'FloatX fract(FloatX x);', insertText: new vscode.SnippetString('fract($1)')
	},
	{
		label: 'trunc', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Find the truncated value of the `x`.',
		signature: 'FloatX trunc(FloatX x);', insertText: new vscode.SnippetString('trunc($1)')
	},
	{
		label: 'min', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return the lesser of two values.',
		signature: 'Type min(Type x, Type y);', insertText: new vscode.SnippetString('min($1, $2)')
	},
	{
		label: 'max', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return the greater of two values.',
		signature: 'Type max(Type x, Type y);', insertText: new vscode.SnippetString('max($1, $2)')
	},
	{
		label: 'clamp', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Constrain the `x` value to lie between two further values. (Inclusive range)',
		signature: 'Type clamp(Type x, Type min, Type max);', insertText: new vscode.SnippetString('clamp($1, $2, $3)')
	},
	{
		label: 'step', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Generate a step function by comparing two values. [r = x < edge ? 0 : 1]',
		signature: 'FloatX step(FloatX edge, FloatX x);', insertText: new vscode.SnippetString('step($1, $2)')
	},
	{
		label: 'smoothstep', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform Hermite interpolation between two values. (edge0 < x < edge1)',
		signature: 'FloatX smoothstep(FloatX edge0, FloatX edge1, FloatX x);', insertText: new vscode.SnippetString('smoothstep($1, $2, $3)')
	},

	{
		label: 'sin', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the sine of the `angle` in radians.',
		signature: 'FloatX sin(FloatX angle);', insertText: new vscode.SnippetString('sin($1)')
	},
	{
		label: 'cos', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the cosine of the `angle` in radians.',
		signature: 'FloatX cos(FloatX angle);', insertText: new vscode.SnippetString('cos($1)')
	},
	{
		label: 'tan', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the tangent of the `angle` in radians.',
		signature: 'FloatX tan(FloatX angle);', insertText: new vscode.SnippetString('tan($1)')
	},
	{
		label: 'asin', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the arcsine of the `angle` in radians.',
		signature: 'FloatX asin(FloatX angle);', insertText: new vscode.SnippetString('asin($1)')
	},
	{
		label: 'acos', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the arccosine of the `angle` in radians.',
		signature: 'FloatX acos(FloatX angle);', insertText: new vscode.SnippetString('acos($1)')
	},
	{
		label: 'atan', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the arc-tangent of the `angle` in radians.',
		signature: 'FloatX atan(FloatX angle);', insertText: new vscode.SnippetString('atan($1)')
	},
	{
		label: 'sinh', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the hyperbolic sine of the `angle` in radians.',
		signature: 'FloatX sinh(FloatX angle);', insertText: new vscode.SnippetString('sinh($1)')
	},
	{
		label: 'cosh', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the hyperbolic cosine of the `angle` in radians.',
		signature: 'FloatX cosh(FloatX angle);', insertText: new vscode.SnippetString('cosh($1)')
	},
	{
		label: 'tanh', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the hyperbolic tangent of the `angle` in radians.',
		signature: 'FloatX tanh(FloatX angle);', insertText: new vscode.SnippetString('tanh($1)')
	},
	{
		label: 'asinh', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the arc hyperbolic sine of the `angle` in radians.',
		signature: 'FloatX asinh(FloatX angle);', insertText: new vscode.SnippetString('asinh($1)')
	},
	{
		label: 'acosh', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the arc hyperbolic cosine of the `angle` in radians.',
		signature: 'FloatX acosh(FloatX angle);', insertText: new vscode.SnippetString('acosh($1)')
	},
	{
		label: 'atanh', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the arc hyperbolic tangent of the `angle` in radians.',
		signature: 'FloatX atanh(FloatX angle);', insertText: new vscode.SnippetString('atanh($1)')
	},
	{
		label: 'degrees', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Convert a quantity in `radians` to degrees.',
		signature: 'FloatX degrees(FloatX radians);', insertText: new vscode.SnippetString('degrees($1)')
	},
	{
		label: 'radians', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Convert a quantity in `degrees` to radians.',
		signature: 'FloatX radians(FloatX degrees);', insertText: new vscode.SnippetString('radians($1)')
	},

	{
		label: 'mix', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Linearly interpolate between two values.',
		signature: 'Type mix(Type x, Type y, Type a);', insertText: new vscode.SnippetString('mix($1, $2, $3)')
	},
	{
		label: 'dot', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Calculate the dot product of two vectors.',
		signature: 'float dot(FloatX x, FloatX y);', insertText: new vscode.SnippetString('dot($1, $2)')
	},
	{
		label: 'cross', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Calculate the cross product of two vectors.',
		signature: 'float3 cross(float3 x, float3 y);', insertText: new vscode.SnippetString('cross($1, $2)')
	},
	{
		label: 'outerProduct', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Calculate the outer product of a pair of vectors.',
		signature: 'FloatXxX outerProduct(FloatX column, FloatX row);', insertText: new vscode.SnippetString('outerProduct($1, $2)')
	},
	{
		label: 'distance', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Calculate the distance between two points.',
		signature: 'float distance(FloatX x, FloatX y);', insertText: new vscode.SnippetString('distance($1, $2)')
	},
	{
		label: 'length', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Calculate the length of the `vector`.',
		signature: 'float length(FloatX vector);', insertText: new vscode.SnippetString('length($1)')
	},
	{
		label: 'normalize', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Calculates the unit vector in the same direction as the original vector.',
		signature: 'FloatX normalize(FloatX vector);', insertText: new vscode.SnippetString('normalize($1)')
	},
	{
		label: 'reflect', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Calculate the reflection direction for an `incident` vector.',
		signature: 'FloatX reflect(FloatX incident, FloatX normal);', insertText: new vscode.SnippetString('reflect($1, $2)')
	},
	{
		label: 'refract', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Calculate the refraction direction for an `incident` vector.',
		signature: 'FloatX refract(FloatX incident, FloatX normal, float eta);', insertText: new vscode.SnippetString('reflect($1, $2, $3)')
	},
	{
		label: 'faceforward', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return a vector pointing in the same direction as another.',
		signature: 'FloatX faceforward(FloatX orient, FloatX incident, FloatX reference);', insertText: new vscode.SnippetString('faceforward($1, $2, $3)')
	},

	{
		label: 'matrixCompMult', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a component-wise multiplication of two matrices.',
		signature: 'FloatXxX matrixCompMult(FloatXxX x, FloatXxX y);', insertText: new vscode.SnippetString('matrixCompMult($1, $2)')
	},
	{
		label: 'inverse', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Calculate the inverse of the `matrix`.',
		signature: 'FloatXxX inverse(FloatXxX matrix);', insertText: new vscode.SnippetString('inverse($1)')
	},
	{
		label: 'transpose', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Calculate the transpose of the `matrix`.',
		signature: 'FloatXxX transpose(FloatXxX matrix);', insertText: new vscode.SnippetString('transpose($1)')
	},
	{
		label: 'determinant', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Calculate the determinant of the `matrix`.',
		signature: 'float determinant(FloatXxX matrix);', insertText: new vscode.SnippetString('determinant($1)')
	},

	{
		label: 'all', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Check whether all elements of a boolean `vector` are true.',
		signature: 'bool all(BoolX vector);', insertText: new vscode.SnippetString('all($1)')
	},
	{
		label: 'any', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Check whether any elements of a boolean `vector` is true.',
		signature: 'bool any(BoolX vector);', insertText: new vscode.SnippetString('any($1)')
	},
	{
		label: 'not', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Logically invert the boolean `vector`.',
		signature: 'bool not(BoolX vector);', insertText: new vscode.SnippetString('not($1)')
	},
	{
		label: 'equal', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a component-wise equal-to comparison of two vectors. [r = x == y]',
		signature: 'BoolX equal(TypeX x, TypeX y);', insertText: new vscode.SnippetString('equal($1, $2)')
	},
	{
		label: 'notEqual', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a component-wise not-equal-to comparison of two vectors. [r = x != y]',
		signature: 'BoolX notEqual(TypeX x, TypeX y);', insertText: new vscode.SnippetString('notEqual($1, $2)')
	},
	{
		label: 'lessThan', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a component-wise less-than comparison of two vectors. [r = x < y]',
		signature: 'BoolX lessThan(TypeX x, TypeX y);', insertText: new vscode.SnippetString('lessThan($1, $2)')
	},
	{
		label: 'lessThanEqual', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a component-wise less-than-or-equal comparison of two vectors. [r = x <= y]',
		signature: 'BoolX lessThanEqual(TypeX x, TypeX y);', insertText: new vscode.SnippetString('lessThanEqual($1, $2)')
	},
	{
		label: 'greaterThan', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a component-wise greater-than comparison of two vectors. [r = x > y]',
		signature: 'BoolX greaterThan(TypeX x, TypeX y);', insertText: new vscode.SnippetString('greaterThan($1, $2)')
	},
	{
		label: 'greaterThanEqual', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a component-wise greater-than-or-equal comparison of two vectors. [r = x >= y]',
		signature: 'BoolX greaterThanEqual(TypeX x, TypeX y);', insertText: new vscode.SnippetString('greaterThanEqual($1, $2)')
	},

	{
		label: 'floatBitsToInt', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Produce the encoding of the `x` floating point value as an signed integer.',
		signature: 'IntX floatBitsToInt(FloatX x);', insertText: new vscode.SnippetString('floatBitsToInt($1)')
	},
	{
		label: 'floatBitsToUint', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Produce the encoding of the `x` floating point value as an unsigned integer.',
		signature: 'UintX floatBitsToUint(FloatX x);', insertText: new vscode.SnippetString('floatBitsToUint($1)')
	},
	{
		label: 'intBitsToFloat', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Produce a floating point using an encoding supplied as the `x` signed integer.',
		signature: 'FloatX intBitsToFloat(IntX x);', insertText: new vscode.SnippetString('intBitsToFloat($1)')
	},
	{
		label: 'uintBitsToFloat', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Produce a floating point using an encoding supplied as the `x` unsigned integer.',
		signature: 'FloatX uintBitsToFloat(UintX x);', insertText: new vscode.SnippetString('uintBitsToFloat($1)')
	},
	{
		label: 'frexp', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Split the `x` floating point number. [x = significand * (2 ^ exponent)]',
		signature: 'FloatX frexp(FloatX x, out FloatX exponent);', insertText: new vscode.SnippetString('frexp($1, $2)')
	},
	{
		label: 'ldexp', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Assemble a floating point number from the `x` and `exponent`',
		signature: 'FloatX ldexp(FloatX x, FloatX exponent);', insertText: new vscode.SnippetString('ldexp($1, $2)')
	},
	{
		label: 'isinf', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Determine whether the `x` is positive or negative infinity',
		signature: 'BoolX isinf(FloatX x);', insertText: new vscode.SnippetString('isinf($1)')
	},
	{
		label: 'isnan', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Determine whether the `x` is a number',
		signature: 'BoolX isnan(FloatX x);', insertText: new vscode.SnippetString('isnan($1)')
	},

	{
		label: 'bitCount', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Counts the number of 1 bits in the `x` integer.',
		signature: 'UIntX bitCount(UIntX x);', insertText: new vscode.SnippetString('bitCount($1)')
	},
	{
		label: 'bitfieldExtract', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Extract a range of bits from the `x` integer.',
		signature: 'UIntX bitfieldExtract(UIntX x, int32 offset, int32 bits);', insertText: new vscode.SnippetString('bitfieldExtract($1, $2, $3)')
	},
	{
		label: 'bitfieldInsert', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Insert a range of bits into the `x` integer.',
		signature: 'UIntX bitfieldInsert(UIntX x, UIntX insert, int32 offset, int32 bits);', insertText: new vscode.SnippetString('bitfieldInsert($1, $2, $3, $4)')
	},
	{
		label: 'bitfieldReverse', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Reverse the order of bits in the `x` integer.',
		signature: 'UIntX bitfieldReverse(UIntX x);', insertText: new vscode.SnippetString('bitfieldReverse($1)')
	},
	{
		label: 'findLSB', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Find the index of the least significant bit set to 1 in the `x` integer.',
		signature: 'IntX findLSB(UIntX x);', insertText: new vscode.SnippetString('findLSB($1)')
	},
	{
		label: 'findMSB', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Find the index of the most significant bit set to 1 in the `x` integer.',
		signature: 'IntX findMSB(UIntX x);', insertText: new vscode.SnippetString('findMSB($1)')
	},

	{
		label: 'packHalf2x16', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Convert two 32-bit floating-point quantities to 16-bit quantities and pack them into a single 32-bit integer.',
		signature: 'uint32 packHalf2x16(float2 x);', insertText: new vscode.SnippetString('packHalf2x16($1)')
	},
	{
		label: 'packDouble2x32', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Create a double-precision value from a pair of unsigned integers.',
		signature: 'double packDouble2x32(uint2 x);', insertText: new vscode.SnippetString('packDouble2x32($1)')
	},
	{
		label: 'packSnorm2x16', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Pack floating-point values into an unsigned integer. [r = round(clamp(x, -1.0, 1.0) * 32767.0)]',
		signature: 'uint32 packSnorm2x16(float2 x);', insertText: new vscode.SnippetString('packSnorm2x16($1)')
	},
	{
		label: 'packSnorm4x8', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Pack floating-point values into an unsigned integer. [r = round(clamp(x, -1.0, 1.0) * 127.0)]',
		signature: 'uint32 packSnorm4x8(float4 x);', insertText: new vscode.SnippetString('packSnorm4x8($1)')
	},
	{
		label: 'packUnorm2x16', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Pack floating-point values into an unsigned integer. [r = round(clamp(x, 0.0, 1.0) * 65535.0)]',
		signature: 'uint32 packUnorm2x16(float2 x);', insertText: new vscode.SnippetString('packUnorm2x16($1)')
	},
	{
		label: 'packUnorm4x8', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Pack floating-point values into an unsigned integer. [r = round(clamp(x, 0.0, 1.0) * 255.0)]',
		signature: 'uint32 packUnorm4x8(float4 x);', insertText: new vscode.SnippetString('packUnorm4x8($1)')
	},

	{
		label: 'unpackHalf2x16', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Convert two 16-bit floating-point values packed into a single 32-bit integer into a vector of two 32-bit floating-point quantities.',
		signature: 'float2 unpackHalf2x16(uint32 x);', insertText: new vscode.SnippetString('unpackHalf2x16($1)')
	},
	{
		label: 'unpackDouble2x32', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Produce two unsigned integers containing the bit encoding of a double precision floating point value.',
		signature: 'uint2 unpackDouble2x32(double x);', insertText: new vscode.SnippetString('unpackDouble2x32($1)')
	},
	{
		label: 'unpackSnorm2x16', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Unpack floating-point values from an unsigned integer. [r = clamp(x / 32727.0, -1.0, 1.0)]',
		signature: 'float2 unpackSnorm2x16(uint32 x);', insertText: new vscode.SnippetString('unpackSnorm2x16($1)')
	},
	{
		label: 'unpackSnorm4x8', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Unpack floating-point values from an unsigned integer. [r = clamp(x / 127.0, -1.0, 1.0)]',
		signature: 'float4 unpackSnorm4x8(uint32 x);', insertText: new vscode.SnippetString('unpackSnorm4x8($1)')
	},
	{
		label: 'unpackUnorm2x16', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Unpack floating-point values from an unsigned integer. [r = x / 65535.0]',
		signature: 'float2 unpackUnorm2x16(uint32 x);', insertText: new vscode.SnippetString('unpackUnorm2x16($1)')
	},
	{
		label: 'unpackUnorm4x8', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Unpack floating-point values from an unsigned integer. [r = x / 255.0]',
		signature: 'float4 unpackUnorm4x8(uint32 x);', insertText: new vscode.SnippetString('unpackUnorm4x8($1)')
	},

	{
		label: 'uaddCarry', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Add unsigned integers and generate carry.',
		signature: 'UintX uaddCarry(UintX x, UintX y, out UintX carry);', insertText: new vscode.SnippetString('uaddCarry($1, $2, $3)')
	},
	{
		label: 'usubBorrow', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Subtract unsigned integers and generate borrow.',
		signature: 'UintX usubBorrow(UintX x, UintX y, out UintX borrow);', insertText: new vscode.SnippetString('usubBorrow($1, $2, $3)')
	},
	{
		label: 'imulExtended', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a 32 by 32 bit signed integer multiply to produce a 64-bit result.',
		signature: 'void imulExtended(IntX x, IntX y, out IntX msb, out IntX lsb);', insertText: new vscode.SnippetString('imulExtended($1, $2, $3, $4)')
	},
	{
		label: 'umulExtended', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a 32 by 32 bit unsigned integer multiply to produce a 64-bit result.',
		signature: 'void umulExtended(UintX x, UintX y, out UintX msb, out UintX lsb);', insertText: new vscode.SnippetString('umulExtended($1, $2, $3, $4)')
	},

	{
		label: 'dFdx', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return the partial derivative of an argument with respect to x. (Fragment Shader)',
		signature: 'FloatX dFdx(FloatX p);', insertText: new vscode.SnippetString('dFdx($1)')
	},
	{
		label: 'dFdy', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return the partial derivative of an argument with respect to y. (Fragment Shader)',
		signature: 'FloatX dFdy(FloatX p);', insertText: new vscode.SnippetString('dFdy($1)')
	},
	{
		label: 'dFdxFine', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Calculate derivatives using local differencing based on the value of `p` for the <br>current fragment and its immediate neighbor(s). (Fragment Shader)',
		signature: 'FloatX dFdxFine(FloatX p);', insertText: new vscode.SnippetString('dFdxFine($1)')
	},
	{
		label: 'dFdyFine', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Calculate derivatives using local differencing based on the value of `p` for the <br>current fragment and its immediate neighbor(s). (Fragment Shader)',
		signature: 'FloatX dFdyFine(FloatX p);', insertText: new vscode.SnippetString('dFdyFine($1)')
	},
	{
		label: 'dFdxCoarse', kind: vscode.CompletionItemKind.Function, 
		documentation: "Calculate derivatives using local differencing based on the value of `p` for the current fragment's <br>neighbors, and will possibly, but not necessarily, include the value for the current fragment. (Fragment Shader)",
		signature: 'FloatX dFdxCoarse(FloatX p);', insertText: new vscode.SnippetString('dFdxCoarse($1)')
	},
	{
		label: 'dFdyCoarse', kind: vscode.CompletionItemKind.Function, 
		documentation: "Calculate derivatives using local differencing based on the value of `p` for the current fragment's <br>neighbors, and will possibly, but not necessarily, include the value for the current fragment. (Fragment Shader)",
		signature: 'FloatX dFdyCoarse(FloatX p);', insertText: new vscode.SnippetString('dFdyCoarse($1)')
	},
	{
		label: 'fwidth', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return the sum of the absolute value of derivatives in x and y. (Fragment Shader)',
		signature: 'FloatX fwidth(FloatX p);', insertText: new vscode.SnippetString('fwidth($1)')
	},
	{
		label: 'fwidthFine', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return the sum of the absolute value of derivatives in x and y. (Fragment Shader)',
		signature: 'FloatX fwidthFine(FloatX p);', insertText: new vscode.SnippetString('fwidthFine($1)')
	},
	{
		label: 'fwidthCoarse', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return the sum of the absolute value of derivatives in x and y. (Fragment Shader)',
		signature: 'FloatX fwidthCoarse(FloatX p);', insertText: new vscode.SnippetString('fwidthCoarse($1)')
	},
	{
		label: 'interpolateAtCentroid', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Sample a varying at the centroid of a pixel. (Fragment Shader)',
		signature: 'FloatX interpolateAtCentroid(FloatX interpolant);', insertText: new vscode.SnippetString('interpolateAtCentroid($1)')
	},
	{
		label: 'interpolateAtOffset', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Sample a varying at specified offset from the center of a pixel. (Fragment Shader)',
		signature: 'FloatX interpolateAtOffset(FloatX interpolant, float2 offset);', insertText: new vscode.SnippetString('interpolateAtOffset($1, $2)')
	},
	{
		label: 'interpolateAtSample', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Sample a varying at the location of a specified sample. (Fragment Shader)',
		signature: 'FloatX interpolateAtSample(FloatX interpolant, int32 sample);', insertText: new vscode.SnippetString('interpolateAtSample($1, $2)')
	},

	{
		label: 'atomicAdd', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform an atomic addition to the `memory` variable.',
		signature: 'UInt atomicAdd(inout UInt memory, UInt value);', insertText: new vscode.SnippetString('atomicAdd($1, $2)')
	},
	{
		label: 'atomicAnd', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform an atomic logical AND operation to the `memory` variable.',
		signature: 'UInt atomicAnd(inout UInt memory, UInt value);', insertText: new vscode.SnippetString('atomicAnd($1, $2)')
	},
	{
		label: 'atomicOr', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform an atomic logical OR operation to the `memory` variable.',
		signature: 'UInt atomicOr(inout UInt memory, UInt value);', insertText: new vscode.SnippetString('atomicOr($1, $2)')
	},
	{
		label: 'atomicXor', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform an atomic logical XOR operation to the `memory` variable.',
		signature: 'UInt atomicXor(inout UInt memory, UInt value);', insertText: new vscode.SnippetString('atomicXor($1, $2)')
	},
	{
		label: 'atomicMin', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform an atomic min operation to the `memory` variable.',
		signature: 'UInt atomicMin(inout UInt memory, UInt value);', insertText: new vscode.SnippetString('atomicMin($1, $2)')
	},
	{
		label: 'atomicMax', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform an atomic max operation to the `memory` variable.',
		signature: 'UInt atomicMax(inout UInt memory, UInt value);', insertText: new vscode.SnippetString('atomicMax($1, $2)')
	},
	{
		label: 'atomicExchange', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform an atomic exchange operation to the `memory` variable.',
		signature: 'UInt atomicExchange(inout UInt memory, UInt value);', insertText: new vscode.SnippetString('atomicExchange($1, $2)')
	},
	{
		label: 'atomicCompSwap', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform an atomic compare-exchange operation to the `memory` variable.',
		signature: 'UInt atomicCompSwap(inout UInt memory, UInt compare, UInt value);', insertText: new vscode.SnippetString('atomicCompSwap($1, $2, $3)')
	},

	{
		label: 'imageAtomicAdd', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Atomically add a value to an existing value in memory and return the original value.',
		signature: 'UInt imageAtomicAdd(Image image, IntX position, UInt value);', insertText: new vscode.SnippetString('imageAtomicAdd($1, $2, $3)')
	},
	{
		label: 'imageAtomicAnd', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Atomically compute the logical AND of a value with an existing value in memory, store that value and return the original value.',
		signature: 'UInt imageAtomicAnd(Image image, IntX position, UInt value);', insertText: new vscode.SnippetString('imageAtomicAnd($1, $2, $3)')
	},
	{
		label: 'imageAtomicOr', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Atomically compute the logical OR of a value with an existing value in memory, store that value and return the original value.',
		signature: 'UInt imageAtomicOr(Image image, IntX position, UInt value);', insertText: new vscode.SnippetString('imageAtomicOr($1, $2, $3)')
	},
	{
		label: 'imageAtomicXor', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Atomically compute the logical XOR of a value with an existing value in memory, store that value and return the original value.',
		signature: 'UInt imageAtomicXor(Image image, IntX position, UInt value);', insertText: new vscode.SnippetString('imageAtomicXor($1, $2, $3)')
	},
	{
		label: 'imageAtomicMin', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Atomically compute the minimum of a value with an existing value in memory, store that value and return the original value.',
		signature: 'UInt imageAtomicMin(Image image, IntX position, UInt value);', insertText: new vscode.SnippetString('imageAtomicMin($1, $2, $3)')
	},
	{
		label: 'imageAtomicMax', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Atomically compute the maximum of a value with an existing value in memory, store that value and return the original value.',
		signature: 'UInt imageAtomicMax(Image image, IntX position, UInt value);', insertText: new vscode.SnippetString('imageAtomicMax($1, $2, $3)')
	},
	{
		label: 'imageAtomicExchange', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Atomically store supplied data into memory and return the original value from memory.',
		signature: 'UInt imageAtomicExchange(Image image, IntX position, UInt value);', insertText: new vscode.SnippetString('imageAtomicExchange($1, $2, $3)')
	},
	{
		label: 'imageAtomicCompSwap', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Atomically compares supplied data with that in memory and conditionally stores it to memory.',
		signature: 'UInt imageAtomicCompSwap(Image image, IntX position, UInt value);', insertText: new vscode.SnippetString('imageAtomicCompSwap($1, $2, $3, $4)')
	},

	{
		label: 'barrier', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Synchronize execution of multiple shader invocations. (Compute Shader)',
		signature: 'void barrier();', insertText: new vscode.SnippetString('barrier()')
	},
	{
		label: 'memoryBarrier', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Controls the ordering of memory transactions issued by a single shader invocation. (Compute Shader)',
		signature: 'void memoryBarrier();', insertText: new vscode.SnippetString('memoryBarrier()')
	},
	{
		label: 'groupMemoryBarrier', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Controls the ordering of memory transaction issued shader invocation relative to a work group. (Compute Shader)',
		signature: 'void groupMemoryBarrier();', insertText: new vscode.SnippetString('groupMemoryBarrier()')
	},
	{
		label: 'memoryBarrierBuffer', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Controls the ordering of operations on buffer variables issued by a single shader invocation. (Compute Shader)',
		signature: 'void memoryBarrierBuffer();', insertText: new vscode.SnippetString('memoryBarrierBuffer()')
	},
	{
		label: 'memoryBarrierImage', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Controls the ordering of operations on image variables issued by a single shader invocation. (Compute Shader)',
		signature: 'void memoryBarrierImage();', insertText: new vscode.SnippetString('memoryBarrierImage()')
	},
	{
		label: 'memoryBarrierShared', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Controls the ordering of operations on shared variables issued by a single shader invocation. (Compute Shader)',
		signature: 'void memoryBarrierShared();', insertText: new vscode.SnippetString('memoryBarrierShared()')
	},

	{
		label: 'texelFetch', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a lookup of a single texel within a texture. (No out of bounds checks!)',
		signature: 'Type4 texelFetch(Sampler sampler, IntX position, int32 lod);', insertText: new vscode.SnippetString('texelFetch($1, $2, $3)')
	},
	{
		label: 'texelFetchOffset', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a lookup of a single texel within a texture with an offset.',
		signature: 'Type4 texelFetchOffset(Sampler sampler, IntX position, int32 lod, Type offset);', insertText: new vscode.SnippetString('texelFetchOffset($1, $2, $3, $4)')
	},
	{
		label: 'texture', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Retrieves texels from a texture.',
		signature: 'Type4 texture(Sampler sampler, FloatX texCoords, [float bias]);', insertText: new vscode.SnippetString('texture($1, $2)')
	},
	{
		label: 'textureOffset', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a texture lookup with offset.',
		signature: 'Type4 textureOffset(Sampler sampler, FloatX texCoords, IntX offset, [float bias]);', insertText: new vscode.SnippetString('textureOffset($1, $2, $3)')
	},
	{
		label: 'textureGather', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Gathers four texels from a texture. (xy: r01, g11, b10, a00)',
		signature: 'Type4 textureGather(Sampler sampler, FloatX texCoords, [int32 component]);', insertText: new vscode.SnippetString('textureGather($1, $2)')
	},
	{
		label: 'textureGatherOffset', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Gathers four texels from a texture with offset. (xy: r01, g11, b10, a00)',
		signature: 'Type4 textureGatherOffset(Sampler sampler, FloatX texCoords, int2 offset, [int32 component]);', insertText: new vscode.SnippetString('textureGatherOffset($1, $2, $3)')
	},
	{
		label: 'textureGatherOffsets', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Gathers four texels from a texture with an array of offsets. (xy: r01, g11, b10, a00)',
		signature: 'Type4 textureGatherOffsets(Sampler sampler, FloatX texCoords, int2 offsets[4], [int32 component]);', insertText: new vscode.SnippetString('textureGatherOffsets($1, $2, $3)')
	},
	{
		label: 'textureGrad', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a texture lookup with explicit gradients.',
		signature: 'Type4 textureGrad(Sampler sampler, FloatX texCoords, FloatX dPdx, FloatX dPdy);', insertText: new vscode.SnippetString('textureGrad($1, $2, $3, $4)')
	},
	{
		label: 'textureGradOffset', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a texture lookup with explicit gradients and offset.',
		signature: 'Type4 textureGradOffset(Sampler sampler, FloatX texCoords, FloatX dPdx, FloatX dPdy, IntX offset);', insertText: new vscode.SnippetString('textureGradOffset($1, $2, $3, $4, $5)')
	},
	{
		label: 'textureLod', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a texture lookup with explicit level-of-detail.',
		signature: 'Type4 textureLod(Sampler sampler, FloatX texCoords, float lod);', insertText: new vscode.SnippetString('textureLod($1, $2, $3)')
	},
	{
		label: 'textureLodOffset', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a texture lookup with explicit level-of-detail and offset.',
		signature: 'Type4 textureLodOffset(Sampler sampler, FloatX texCoords, float lod, IntX offset);', insertText: new vscode.SnippetString('textureLodOffset($1, $2, $3, $4)')
	},
	{
		label: 'textureProj', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a texture lookup with projection.',
		signature: 'Type4 textureProj(Sampler sampler, FloatX texCoords, [float bias]);', insertText: new vscode.SnippetString('textureProj($1, $2)')
	},
	{
		label: 'textureProjOffset', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a texture lookup with projection and offset.',
		signature: 'Type4 textureProjOffset(Sampler sampler, FloatX texCoords, IntX offset, [float bias]);', insertText: new vscode.SnippetString('textureProjOffset($1, $2, $3)')
	},
	{
		label: 'textureProjGrad', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a texture lookup with projection and explicit gradients.',
		signature: 'Type4 textureProjGrad(Sampler sampler, FloatX texCoords, FloatX dPdx, FloatX dPdy);', insertText: new vscode.SnippetString('textureProjGrad($1, $2, $3, $4)')
	},
	{
		label: 'textureProjGradOffset', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a texture lookup with projection, explicit gradients and offset.',
		signature: 'Type4 textureProjGradOffset(Sampler sampler, FloatX texCoords, FloatX dPdx, FloatX dPdy, IntX offset);', insertText: new vscode.SnippetString('textureProjGradOffset($1, $2, $3, $4, $5)')
	},
	{
		label: 'textureProjLod', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a texture lookup with projection and explicit level-of-detail.',
		signature: 'Type4 textureProjLod(Sampler sampler, FloatX texCoords, float lod);', insertText: new vscode.SnippetString('textureProjLod($1, $2, $3)')
	},
	{
		label: 'textureProjLodOffset', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Perform a texture lookup with projection, explicit level-of-detail and offset.',
		signature: 'Type4 textureProjLodOffset(Sampler sampler, FloatX texCoords, float lod, IntX offset);', insertText: new vscode.SnippetString('textureProjLodOffset($1, $2, $3, $4)')
	},
	{
		label: 'textureSize', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Retrieve the dimensions of a level of a texture.',
		signature: 'IntX textureSize(Sampler sampler, int32 lod);', insertText: new vscode.SnippetString('textureSize($1, $2)')
	},
	{
		label: 'textureSamples', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return the number of samples of a texture.',
		signature: 'int32 textureSamples(Sampler sampler);', insertText: new vscode.SnippetString('textureSamples($1)')
	},
	{
		label: 'textureQueryLevels', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Compute the number of accessible mipmap levels of a texture.',
		signature: 'int32 textureQueryLevels(Sampler sampler);', insertText: new vscode.SnippetString('textureQueryLevels($1)')
	},
	{
		label: 'textureQueryLod', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Compute the level-of-detail that would be used to sample from a texture.',
		signature: 'float2 textureQueryLod(Sampler sampler, FloatX texCoords);', insertText: new vscode.SnippetString('textureQueryLod($1, $2)')
	},


	{
		label: 'imageLoad', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Load a single texel from the `image`. (No out of bounds checks!)',
		signature: 'Type4 imageLoad(Image image, IntX position);', insertText: new vscode.SnippetString('imageLoad($1, $2)')
	},
	{
		label: 'imageStore', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Write a single texel into the `image`. (No out of bounds checks!)',
		signature: 'void imageStore(Image image, IntX position, Type4 value);', insertText: new vscode.SnippetString('imageStore($1, $2, $3)')
	},
	{
		label: 'imageSize', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Retrieve the dimensions of the `image`.',
		signature: 'IntX imageSize(Image image);', insertText: new vscode.SnippetString('imageSize($1)')
	},
	{
		label: 'imageSamples', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Return the number of samples of the `image`.',
		signature: 'int32 imageSamples(Image image);', insertText: new vscode.SnippetString('imageSamples($1)')
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
		documentation: 'Contains the index of the current primitive. (Fragment, Ray Tracing and Mesh Shaders)', 
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

	{
		label: 'gsl.variant', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'Compile-time index of the current shader variant.', 
		signature: 'spec const uint32 gsl.variant;', insertText: new vscode.SnippetString('gsl.variant')
	},
	{
		label: 'gsl.rayRecursionDepth', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'Compile-time maximum ray recursion depth.', 
		signature: 'const uint32 gsl.rayRecursionDepth;', insertText: new vscode.SnippetString('gsl.rayRecursionDepth')
	},

	{
		label: 'accelerationStructure', kind: vscode.CompletionItemKind.Class, 
		documentation: "A structure that holds the scene's spatial hierarchy for ray tracing.", 
		signature: 'uniform set0 accelerationStructure name;', insertText: new vscode.SnippetString('accelerationStructure ')
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
		documentation: 'Declares data for callable shaders, which are invoked from ray tracing shaders.', 
		signature: 'callableData Type name;', insertText: new vscode.SnippetString('callableData $1 $2;')
	},
	{
		label: 'callableDataIn', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Declares an input data for callable shaders.', 
		signature: 'callableDataIn Type name;', insertText: new vscode.SnippetString('callableDataIn $1 $2;')
	},
	{
		label: '#rayPayloadOffset', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Adds offset to the `rayPayload` or `rayPayloadIn` index.', 
		signature: '#rayPayloadOffset x', insertText: new vscode.SnippetString('#rayPayloadOffset $1')
	},
	{
		label: '#callableDataOffset', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Adds offset to the `callableData` or `callableDataIn` index.', 
		signature: '#callableDataOffset x', insertText: new vscode.SnippetString('#callableDataOffset $1')
	},
	{
		label: 'ignoreIntersection', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Ignore this intersection and continue traversal, from an any-hit shader. (Ray Tracing Shader)', 
		signature: 'ignoreIntersection;', insertText: new vscode.SnippetString('ignoreIntersection')
	},
	{
		label: 'terminateRay', kind: vscode.CompletionItemKind.Keyword, 
		documentation: 'Terminate traversal immediately, from an any-hit shader. (Ray Tracing Shader)', 
		signature: 'terminateRay;', insertText: new vscode.SnippetString('terminateRay')
	},
	{
		label: 'traceRay', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Trace a ray through the TLAS and invoke the hit/miss shaders. (Ray Tracing Shader)', 
		signature: 'void traceRay(accelerationStructure tlas, uint32 rayFlags, uint32 cullMask, uint32 sbtRecordOffset, \n\tuint32 sbtRecordStride, uint32 missIndex, float3 origin, float tMin, float3 direction, float tMax, int32 payload);', 
		insertText: new vscode.SnippetString('traceRay($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)')
	},
	{
		label: 'reportIntersection', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Report a hit at distance `t` with `hitKind`, from an intersection shader. (Ray Tracing Shader)', 
		signature: 'bool reportIntersection(float t, uint32 hitKind);', insertText: new vscode.SnippetString('reportIntersection($1, $2)')
	},
	{
		label: 'executeCallable', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Invoke a callable shader. (Ray Tracing Shader)', 
		signature: 'void executeCallable(uint32 sbtRecordOffset, uint32 dataIndex);', insertText: new vscode.SnippetString('executeCallable($1, $2)')
	},
	{
		label: 'gl.launchID', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the 3D index of the current ray within the launch grid. (Ray Tracing Shader)', 
		signature: 'in uint3 gl.launchID;', insertText: new vscode.SnippetString('gl.launchID')
	},
	{
		label: 'gl.launchSize', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the 3D size of the ray launch grid. (Ray Tracing Shader)', 
		signature: 'in uint3 gl.launchSize;', insertText: new vscode.SnippetString('gl.launchSize')
	},
	{
		label: 'gl.launchSize', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the 3D size of the ray launch grid. (Ray Tracing Shader)', 
		signature: 'in uint3 gl.launchSize;', insertText: new vscode.SnippetString('gl.launchSize')
	},
	{
		label: 'gl.instanceID', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the instance ID of the hit geometry. (Ray Tracing Shader)', 
		signature: 'in int32 gl.instanceID;', insertText: new vscode.SnippetString('gl.instanceID')
	},
	{
		label: 'gl.instanceCustomIndex', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains user-assigned custom index of the hit instance. (Ray Tracing Shader)', 
		signature: 'in int32 gl.instanceCustomIndex;', insertText: new vscode.SnippetString('gl.instanceCustomIndex')
	},
	{
		label: 'gl.geometryIndex', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the hit geometry index within the acceleration structure. (Ray Tracing Shader)', 
		signature: 'in int32 gl.geometryIndex;', insertText: new vscode.SnippetString('gl.geometryIndex')
	},
	{
		label: 'gl.worldRayOrigin', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the ray origin in world (global) coordinates. (Ray Tracing Shader)', 
		signature: 'in float3 gl.worldRayOrigin;', insertText: new vscode.SnippetString('gl.worldRayOrigin')
	},
	{
		label: 'gl.worldRayDirection', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the ray direction in world (global) coordinates. (Ray Tracing Shader)', 
		signature: 'in float3 gl.worldRayDirection;', insertText: new vscode.SnippetString('gl.worldRayDirection')
	},
	{
		label: 'gl.objectRayOrigin', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the ray origin in object (local) coordinates. (Ray Tracing Shader)', 
		signature: 'in float3 gl.objectRayOrigin;', insertText: new vscode.SnippetString('gl.objectRayOrigin')
	},
	{
		label: 'gl.objectRayDirection', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the ray direction in object (local) coordinates. (Ray Tracing Shader)', 
		signature: 'in float3 gl.objectRayDirection;', insertText: new vscode.SnippetString('gl.objectRayDirection')
	},
	{
		label: 'gl.rayTmin', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the minimum ray-parameter `t` to consider. (Ray Tracing Shader)', 
		signature: 'in float gl.rayTmin;', insertText: new vscode.SnippetString('gl.rayTmin')
	},
	{
		label: 'gl.rayTmax', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the maximum ray-parameter `t` to consider. <br>It may be volatile in an intersection shader! (Ray Tracing Shader)', 
		signature: 'in <volatile> float gl.rayTmax;', insertText: new vscode.SnippetString('gl.rayTmax')
	},
	{
		label: 'gl.incomingRayFlags', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the bitmask of the ray flags. (Ray Tracing Shader)', 
		signature: 'in uint32 gl.incomingRayFlags;', insertText: new vscode.SnippetString('gl.incomingRayFlags')
	},
	{
		label: 'gl.hitT', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'An alias of the `gl.rayTmax` variable. (Ray Tracing Shader)', 
		signature: 'in <volatile> float gl.hitT;', insertText: new vscode.SnippetString('gl.hitT')
	},
	{
		label: 'gl.hitKind', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Indicates front or back facing triangle hit, or a custom 0-255 hit kind. (Ray Tracing Shader)', 
		signature: 'in uint32 gl.hitKind;', insertText: new vscode.SnippetString('gl.hitKind')
	},
	{
		label: 'gl.objectToWorld', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the object to world space transformation matrix. (Ray Tracing Shader)', 
		signature: 'in float4x3 gl.objectToWorld;', insertText: new vscode.SnippetString('gl.objectToWorld')
	},
	{
		label: 'gl.worldToObject', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the world to object space transformation matrix. (Ray Tracing Shader)', 
		signature: 'in float4x3 gl.worldToObject;', insertText: new vscode.SnippetString('gl.worldToObject')
	},
	{
		label: 'gl.worldToObject3x4', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the 3x4 portion of the world to objecttransformation matrix. (Ray Tracing Shader)', 
		signature: 'in float3x4 gl.worldToObject3x4;', insertText: new vscode.SnippetString('gl.worldToObject3x4')
	},
	{
		label: 'gl.objectToWorld3x4', kind: vscode.CompletionItemKind.Variable, 
		documentation: 'Contains the 3x4 portion of the object to world transformation matrix. (Ray Tracing Shader)', 
		signature: 'in float3x4 gl.objectToWorld3x4;', insertText: new vscode.SnippetString('gl.objectToWorld3x4')
	},
	{
		label: 'gl.rayFlagsNone', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'No special behavior. (Ray Tracing Shader)', 
		signature: 'const uint32 gl.rayFlagsNone = 0x00;', insertText: new vscode.SnippetString('gl.rayFlagsNone')
	},
	{
		label: 'gl.rayFlagsOpaque', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'Treat all geometry as opaque, ignore "noOpaque" any-hit logic. (Ray Tracing Shader)', 
		signature: 'const uint32 gl.rayFlagsOpaque = 0x01;', insertText: new vscode.SnippetString('gl.rayFlagsOpaque')
	},
	{
		label: 'gl.rayFlagsNoOpaque', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'Treat all geometry as non-opaque, ignore "opaque" any-hit logic. (Ray Tracing Shader)', 
		signature: 'const uint32 gl.rayFlagsNoOpaque = 0x02;', insertText: new vscode.SnippetString('gl.rayFlagsNoOpaque')
	},
	{
		label: 'gl.rayFlagsTerminateOnFirstHit', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'Terminate traversal on first hit. (Ray Tracing Shader)', 
		signature: 'const uint32 gl.rayFlagsTerminateOnFirstHit = 0x04;', insertText: new vscode.SnippetString('gl.rayFlagsTerminateOnFirstHit')
	},
	{
		label: 'gl.rayFlagsSkipClosestHitShader', kind: vscode.CompletionItemKind.Constant, 
		documentation: "Don't invoke closest-hit shaders. (Ray Tracing Shader)", 
		signature: 'const uint32 gl.rayFlagsSkipClosestHitShader = 0x08;', insertText: new vscode.SnippetString('gl.rayFlagsSkipClosestHitShader')
	},
	{
		label: 'gl.rayFlagsCullBackFacingTriangles', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'Cull back-facing triangles. (Ray Tracing Shader)', 
		signature: 'const uint32 gl.rayFlagsCullBackFacingTriangles = 0x10;', insertText: new vscode.SnippetString('gl.rayFlagsCullBackFacingTriangles')
	},
	{
		label: 'gl.rayFlagsCullFrontFacingTriangles', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'Cull front-facing triangles. (Ray Tracing Shader)', 
		signature: 'const uint32 gl.rayFlagsCullFrontFacingTriangles = 0x20;', insertText: new vscode.SnippetString('gl.rayFlagsCullFrontFacingTriangles')
	},
	{
		label: 'gl.rayFlagsCullOpaque', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'Cull opaque geometry. (Ray Tracing Shader)', 
		signature: 'const uint32 gl.rayFlagsCullOpaque = 0x40;', insertText: new vscode.SnippetString('gl.rayFlagsCullOpaque')
	},
	{
		label: 'gl.rayFlagsCullNoOpaque', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'Cull non-opaque geometry. (Ray Tracing Shader)', 
		signature: 'const uint32 gl.rayFlagsCullNoOpaque = 0x80;', insertText: new vscode.SnippetString('gl.rayFlagsCullNoOpaque')
	},
	{
		label: 'gl.hitKindFrontFacingTriangle', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'Front-facing triangle hit. (Ray Tracing Shader)', 
		signature: 'const uint32 gl.hitKindFrontFacingTriangle = 0xFE;', insertText: new vscode.SnippetString('gl.hitKindFrontFacingTriangle')
	},
	{
		label: 'gl.hitKindBackFacingTriangle', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'Front-facing triangle hit. (Ray Tracing Shader)', 
		signature: 'const uint32 gl.hitKindBackFacingTriangle = 0xFF;', insertText: new vscode.SnippetString('gl.hitKindBackFacingTriangle')
	},

	{
		label: 'rayQuery', kind: vscode.CompletionItemKind.Class, 
		documentation: "A structure that holds the query traversal information for ray tracing.", 
		signature: 'rayQuery name;', insertText: new vscode.SnippetString('rayQuery ')
	},
	{
		label: 'rayQueryInitialize', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Initializes a ray query object for future TLAS traversal.', 
		signature: 'void rayQueryInitialize(rayQuery query, accelerationStructure tlas, uint32 rayFlags, \n\tuint32 cullMask, float3 origin, float tMin, float3 direction, float tMax);', 
		insertText: new vscode.SnippetString('rayQueryInitialize($1, $2, $3, $4, $5, $6, $7, $8)')
	},
	{
		label: 'rayQueryProceed', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Proceed TLAS traversal when ray traversal is incomplete.', 
		signature: 'bool rayQueryProceed(rayQuery query);', 
		insertText: new vscode.SnippetString('rayQueryProceed($1)')
	},
	{
		label: 'rayQueryTerminate', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Terminates execution of ray query when traversal is incomplete.', 
		signature: 'void rayQueryTerminate(rayQuery query);', 
		insertText: new vscode.SnippetString('rayQueryTerminate($1)')
	},
	{
		label: 'rayQueryGenerateIntersection', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Generates and commits an AABB intersection at `tHit`.', 
		signature: 'void rayQueryGenerateIntersection(rayQuery query, float tHit);', 
		insertText: new vscode.SnippetString('rayQueryGenerateIntersection($1, $2)')
	},
	{
		label: 'rayQueryConfirmIntersection', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Commits current candidate triangle intersection.', 
		signature: 'void rayQueryConfirmIntersection(rayQuery query);', 
		insertText: new vscode.SnippetString('rayQueryConfirmIntersection($1)')
	},
	{
		label: 'rayQueryGetIntersectionType', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns type of committed or candidate intersection.', 
		signature: 'uint32 rayQueryGetIntersectionType(rayQuery query, bool committed);', 
		insertText: new vscode.SnippetString('rayQueryGetIntersectionType($1, $2)')
	},
	{
		label: 'rayQueryGetRayTMin', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the parametric `tMin` value for the ray query.', 
		signature: 'float rayQueryGetRayTMin(rayQuery query);', 
		insertText: new vscode.SnippetString('rayQueryGetRayTMin($1)')
	},
	{
		label: 'rayQueryGetRayFlags', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the ray flags for the ray query.', 
		signature: 'uint32 rayQueryGetRayFlags(rayQuery query);', 
		insertText: new vscode.SnippetString('rayQueryGetRayFlags($1)')
	},
	{
		label: 'rayQueryGetWorldRayOrigin', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the world-space origin of ray for the ray query.', 
		signature: 'float3 rayQueryGetWorldRayOrigin(rayQuery query);', 
		insertText: new vscode.SnippetString('rayQueryGetWorldRayOrigin($1)')
	},
	{
		label: 'rayQueryGetWorldRayDirection', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the world-space direction of ray for the ray query.', 
		signature: 'float3 rayQueryGetWorldRayDirection(rayQuery query);', 
		insertText: new vscode.SnippetString('rayQueryGetWorldRayDirection($1)')
	},
	{
		label: 'rayQueryGetWorldRayDirection', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the world-space direction of ray for the ray query.', 
		signature: 'float3 rayQueryGetWorldRayDirection(rayQuery query);', 
		insertText: new vscode.SnippetString('rayQueryGetWorldRayDirection($1)')
	},
	{
		label: 'rayQueryGetIntersectionT', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the parametric `t` value for current intersection.', 
		signature: 'float rayQueryGetIntersectionT(rayQuery query, bool committed);', 
		insertText: new vscode.SnippetString('rayQueryGetIntersectionT($1, $2)')
	},
	{
		label: 'rayQueryGetIntersectionInstanceCustomIndex', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the custom index of the instance for current intersection.', 
		signature: 'int32 rayQueryGetIntersectionInstanceCustomIndex(rayQuery query, bool committed);', 
		insertText: new vscode.SnippetString('rayQueryGetIntersectionInstanceCustomIndex($1, $2)')
	},
	{
		label: 'rayQueryGetIntersectionInstanceId', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the index of the instance for current intersection.', 
		signature: 'int32 rayQueryGetIntersectionInstanceId(rayQuery query, bool committed);', 
		insertText: new vscode.SnippetString('rayQueryGetIntersectionInstanceId($1, $2)')
	},
	{
		label: 'rayQueryGetIntersectionInstanceShaderBindingTableRecordOffset', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the application-specified BLAS SBT record offset value.', 
		signature: 'uint32 rayQueryGetIntersectionInstanceShaderBindingTableRecordOffset(rayQuery query, bool committed);', 
		insertText: new vscode.SnippetString('rayQueryGetIntersectionInstanceShaderBindingTableRecordOffset($1, $2)')
	},
	{
		label: 'rayQueryGetIntersectionGeometryIndex', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns implementation defined index of geometry for current intersection.', 
		signature: 'int32 rayQueryGetIntersectionGeometryIndex(rayQuery query, bool committed);', 
		insertText: new vscode.SnippetString('rayQueryGetIntersectionGeometryIndex($1, $2)')
	},
	{
		label: 'rayQueryGetIntersectionPrimitiveIndex', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns the index of the primitive within the geometry of the BLAS being processed.', 
		signature: 'int32 rayQueryGetIntersectionPrimitiveIndex(rayQuery query, bool committed);', 
		insertText: new vscode.SnippetString('rayQueryGetIntersectionPrimitiveIndex($1, $2)')
	},
	{
		label: 'rayQueryGetIntersectionBarycentrics', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns floating point barycentric coordinates of current intersection of ray.', 
		signature: 'float2 rayQueryGetIntersectionBarycentrics(rayQuery query, bool committed);', 
		insertText: new vscode.SnippetString('rayQueryGetIntersectionBarycentrics($1, $2)')
	},
	{
		label: 'rayQueryGetIntersectionFrontFace', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns true if the current intersection is a front facing triangle.', 
		signature: 'bool rayQueryGetIntersectionFrontFace(rayQuery query, bool committed);', 
		insertText: new vscode.SnippetString('rayQueryGetIntersectionFrontFace($1, $2)')
	},
	{
		label: 'rayQueryGetIntersectionCandidateAABBOpaque', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns true if the current candidate intersection is an opaque AABB.', 
		signature: 'bool rayQueryGetIntersectionCandidateAABBOpaque(rayQuery query);', 
		insertText: new vscode.SnippetString('rayQueryGetIntersectionCandidateAABBOpaque($1)')
	},
	{
		label: 'rayQueryGetIntersectionObjectRayDirection', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns object-space direction of ray for current intersection.', 
		signature: 'float3 rayQueryGetIntersectionObjectRayDirection(rayQuery query, bool committed);', 
		insertText: new vscode.SnippetString('rayQueryGetIntersectionObjectRayDirection($1, $2)')
	},
	{
		label: 'rayQueryGetIntersectionObjectRayOrigin', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns object-space origin of ray for current intersection.', 
		signature: 'float3 rayQueryGetIntersectionObjectRayOrigin(rayQuery query, bool committed);', 
		insertText: new vscode.SnippetString('rayQueryGetIntersectionObjectRayOrigin($1, $2)')
	},
	{
		label: 'rayQueryGetIntersectionObjectToWorld', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns object to world transformation matrix for current intersection.', 
		signature: 'float4x3 rayQueryGetIntersectionObjectToWorld(rayQuery query, bool committed);', 
		insertText: new vscode.SnippetString('rayQueryGetIntersectionObjectToWorld($1, $2)')
	},
	{
		label: 'rayQueryGetIntersectionWorldToObject', kind: vscode.CompletionItemKind.Function, 
		documentation: 'Returns world to object transformation matrix for current intersection.', 
		signature: 'float4x3 rayQueryGetIntersectionWorldToObject(rayQuery query, bool committed);', 
		insertText: new vscode.SnippetString('rayQueryGetIntersectionWorldToObject($1, $2)')
	},
	{
		label: 'gl.rayQueryCommittedIntersectionNone', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'No intersection was committed.', 
		signature: 'const uint32 gl.rayQueryCommittedIntersectionNone = 0;', insertText: new vscode.SnippetString('gl.rayQueryCommittedIntersectionNone')
	},
	{
		label: 'gl.rayQueryCommittedIntersectionTriangle', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'The committed hit was a triangle.', 
		signature: 'const uint32 gl.rayQueryCommittedIntersectionTriangle = 1;', insertText: new vscode.SnippetString('gl.rayQueryCommittedIntersectionTriangle')
	},
	{
		label: 'gl.rayQueryCommittedIntersectionGenerated', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'The committed hit was a generated geometry.', 
		signature: 'const uint32 gl.rayQueryCommittedIntersectionGenerated = 2;', insertText: new vscode.SnippetString('gl.rayQueryCommittedIntersectionGenerated')
	},
	{
		label: 'gl.rayQueryCandidateIntersectionTriangle', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'The candidate intersection is with a triangle.', 
		signature: 'const uint32 gl.rayQueryCandidateIntersectionTriangle = 0;', insertText: new vscode.SnippetString('gl.rayQueryCandidateIntersectionTriangle')
	},
	{
		label: 'gl.rayQueryCandidateIntersectionAABB', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'The candidate intersection is with an AABB.', 
		signature: 'const uint32 gl.rayQueryCandidateIntersectionAABB = 1;', insertText: new vscode.SnippetString('gl.rayQueryCandidateIntersectionAABB')
	},

	{
		label: 'ext.debugPrintf', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'Adds a printf(fmt, ...) function which writes to the debug output log. (Use vkconfig-gui)', 
		signature: '#feature ext.debugPrintf', insertText: new vscode.SnippetString('ext.debugPrintf')
	},
	{
		label: 'ext.rayQuery', kind: vscode.CompletionItemKind.Constant, 
		documentation: 'Allows existing shader stages to execute ray intersection queries.', 
		signature: '#feature ext.rayQuery', insertText: new vscode.SnippetString('ext.rayQuery')
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

	let completionProvider = vscode.languages.registerCompletionItemProvider("*",
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
	});

	context.subscriptions.push(hoverProvider, completionProvider);
}

module.exports =
{
	activate
};
