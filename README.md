# Garden Shading Language (GSL)

GSL is a custom shader language based on [GLSL](https://en.wikipedia.org/wiki/OpenGL_Shading_Language). It was created for the [Garden](https://github.com/cfnptr/garden) game engine
to simplify and standardize shader development. In the engine's repository, you can also find a compiler. Documentation with language changes is [here](https://github.com/cfnptr/garden/blob/main/docs/GSL.md).

#### Vertex Shader (.vert)

```
#include "common/tone-mapping.gsl"

in float2 vs.position : f32;
in float2 vs.texCoords : f32;
in float4 vs.color : f8;

out float4 fs.color;
out float2 fs.texCoords;

uniform pushConstants
{
    float2 scale;
    float2 translate;
} pc;

void main()
{
    gl.position = float4(vs.position * pc.scale + pc.translate, 0.0f, 1.0f);
    float3 color = gammaCorrection(vs.color.rgb, DEFAULT_GAMMA);
    fs.color = float4(color, vs.color.a);
    fs.texCoords = vs.texCoords;
}
```

#### Fragment Shader (.frag)

```
pipelineState
{
    faceCulling = off;
    blending0 = on;
}

in float4 fs.color;
in float2 fs.texCoords;

out float4 fb.color;

uniform sampler2D
{
    filter = linear;
} tex;

void main()
{
    fb.color = fs.color * texture(tex, fs.texCoords);
}
```
