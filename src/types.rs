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
use std::sync::mpsc;
use bytes::Bytes;
use futures::stream::Stream;
use std::pin::Pin;
pub enum FactoryMessage {
    Request(String, mpsc::Sender<FactoryMessage>), // HTTP 请求 (id, 返回流的 Sender)
    Response(Pin<Box<dyn Stream<Item = Result<Bytes, std::io::Error>> + Send>>)
}
