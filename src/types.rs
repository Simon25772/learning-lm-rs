use half::{f16, bf16};
pub trait F32{
    fn from_f32(value: f32) -> Self;
    fn as_f32(&self) -> f32;
}
impl F32 for f32{
    fn from_f32(value: f32) -> Self{
        value
    }
    fn as_f32(&self)->f32{
        *self
    }
}
impl F32 for f16{
    fn from_f32(value: f32) -> Self{
        f16::from_f32(value)
    }
    fn as_f32(&self)->f32{
        self.to_f32()
    }
}
impl F32 for bf16{
    fn from_f32(value: f32) -> Self{
        bf16::from_f32(value)
    }
    fn as_f32(&self)->f32{
        self.to_f32()
    }
}