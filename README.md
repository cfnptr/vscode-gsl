<p align="center"><img src="logo.png" alt="GSL logo" width="128"/></p>

# Garden Shading Language (GSL)

GSL is a custom shader language based on [GLSL](https://en.wikipedia.org/wiki/OpenGL_Shading_Language). It was created for the [Garden](https://github.com/cfnptr/garden) game engine
to simplify and standardize shader development. In the engine's repository, you can also find a compiler. Documentation with language changes is [here](https://github.com/cfnptr/garden/blob/main/docs/GSL.md).

#### Vertex Shader (.vert)

```
#include "common/tone-mapping.gsl"

vertexBuffer
{
    float2 position : f32;
    float2 texCoords : f32;
    float4 color : unorm8;
}

uniform pushConstants
{
    float2 scale;
    float2 translate;
} pc;

out float4 fs.color;
out float2 fs.texCoords;

void main()
{
    gl.position = float4(vs.position * pc.scale + pc.translate, 0.0f, 1.0f);
    fs.color = float4(gammaCorrection(vs.color.rgb, DEFAULT_GAMMA), vs.color.a);
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

# Install GSL extension

1. Launch VS Code Quick Open **(Ctrl+P)**
2. And enter: ```ext install cfnptr.gsl-linter```

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for details.
