struct VSInput {
    float3 inPosition;
    float3 inColor;
    float2 inTextCoord;
}
struct UniformBufferObject {
  float4x4 model;
  float4x4 view;
  float4x4 proj;
};
ConstantBuffer<UniformBufferObject> ubo;
struct VSOutput {
    float4 pos : SV_Position;
    float3 color;
    float2 fragTextCoord;
};

[shader("vertex")]
VSOutput vertMain(VSInput input) {
    VSOutput output;
    output.pos = mul(ubo.proj,mul(ubo.view,mul(ubo.model,float4(input.inPosition,1.0))));
    output.color = input.inColor;
    output.fragTextCoord = input.inTextCoord;
    return output;
}

Sampler2D texture;

[shader("fragment")]
float4 fragMain(VSOutput vertIn) : SV_Target
{
    return texture.Sample(vertIn.fragTextCoord);
}
