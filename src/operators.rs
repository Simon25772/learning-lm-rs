use crate::tensor::Tensor;
use std::ops::{Add, Div, Mul, Neg, Sub};
use num_traits::float::Float; // 使用 num_traits 库中的 Float trait 来支持浮点数操作
use cust::{launch, module::Module, stream::Stream, memory::DeviceBuffer};
use cust::memory::DeviceCopy;
use cust::memory::CopyDestination;

// get (row) vectors from a 2D table given a list of indices
pub fn gather<T>(y: &mut Tensor<T>, indices: &Tensor<u32>, table: &Tensor<T>)
where
    T: Float + Copy + Clone + Default,
{
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// #[cfg(not(feature="gpu"))]
// RoPE: Rotary Positional Embedding
pub fn rope<T>(y: &mut Tensor<T>, start_pos: usize, theta: T)
where
    T: Float + Sub<Output = T> + Mul<Output = T> + Copy + Clone + Default,
{
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = T::from(start_pos + tok).unwrap();
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let idx1 = tok * n_heads * d + head * d + i;
                let idx2 = idx1 + d / 2;
                let a = data[idx1];
                let b = data[idx2];
                let freq = pos / theta.powf(T::from(i * 2).unwrap() / T::from(d).unwrap());
                let (sin, cos) = freq.sin_cos();
                data[idx1] = a * cos - b * sin;
                data[idx2] = b * cos + a * sin;
            }
        }
    }
}

#[cfg(not(feature="gpu"))]
// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax<T>(y: &mut Tensor<T>)
where
    T: Float + Sub<Output = T> + Div<Output = T> + Copy + Clone + Default + std::iter::Sum + std::ops::DivAssign,
{
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, &b| a.max(b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<T>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = T::zero());
        }
    }
}

#[cfg(not(feature="gpu"))]
pub fn rms_norm<T>(y: &mut Tensor<T>, x: &Tensor<T>, w: &Tensor<T>, epsilon: T)
where
    T: Float + Mul<Output = T> + Add<Output = T> + Div<Output = T> + Copy + Clone + Default + std::iter::Sum,
{
    let dim = x.shape()[x.shape().len() - 1];
    let batch = x.size() / dim;

    for i in 0..batch {
        let start = i * dim;
        let x_i = &x.data()[start..][..dim];
        let y_i = &mut unsafe { y.data_mut() }[start..][..dim];

        let f = (x_i.iter().map(|&x_ii| x_ii * x_ii).sum::<T>() / T::from(dim).unwrap() + epsilon).sqrt();

        y_i.iter_mut()
            .zip(x_i.iter().zip(w.data().iter()).map(|(&x_ii, &w_i)| x_ii * w_i / f))
            .for_each(|(y_ii, x_ii)| *y_ii = x_ii);
    }
}

#[cfg(not(feature="gpu"))]
// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu<T>(y: &mut Tensor<T>, x: &Tensor<T>)
where
    T: Float + Mul<Output = T> + Add<Output = T> + Neg<Output = T> + Copy + Clone + Default,
{
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    unsafe {
        y.data_mut()
            .iter_mut()
            .zip(
                x.data()
                    .iter()
                    .map(|x_i| T::one() / (T::one() + (-*x_i).exp()))
                    .zip(x.data().iter())
                    .map(|(s_x, x_i)| s_x * *x_i),
            )
            .for_each(|(y_i, s_x)| *y_i = s_x * (*y_i));
    }
}

#[cfg(not(feature="gpu"))]
// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb<T>(
    c: &mut Tensor<T>,
    beta: T,
    a: &Tensor<T>,
    b: &Tensor<T>,
    alpha: T,
) where
    T: Float + Mul<Output = T> + Add<Output = T> + Copy + Clone + Default,
{
    // 获取矩阵的维度
    let dim1 = a.shape()[0];
    let dim2 = b.shape()[0];
    let dim3 = a.shape()[1];

    // 检查维度是否匹配
    assert_eq!(a.shape()[1], b.shape()[1], "矩阵 a 和 b 的列数不匹配");
    assert_eq!(c.shape()[0], dim1, "矩阵 c 的行数与矩阵 a 的行数不匹配");
    assert_eq!(c.shape()[1], dim2, "矩阵 c 的列数与矩阵 b 的行数不匹配");

    // 遍历矩阵进行计算
    for i in 0..dim1 {
        for j in 0..dim2 {
            let mut sum = T::zero(); // 使用泛型的零值
            for k in 0..dim3 {
                sum = sum + a.data()[i * dim3 + k] * b.data()[j * dim3 + k];
            }
            unsafe {
                c.data_mut()[i * dim2 + j] = sum * alpha + c.data()[i * dim2 + j] * beta;
            }
        }
    }
}


// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot<T>(x: &Tensor<T>, y: &Tensor<T>) -> T
where
    T: Float + Mul<Output = T> + Add<Output = T> + Copy + Default,
{
    let len = x.size();
    assert!(len == y.size());
    x.data()
        .iter()
        .zip(y.data().iter())
        .fold(T::zero(), |sum, (&xi, &yi)| sum + xi * yi)
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample<T>(x: &Tensor<T>, top_p: T, top_k: u32, temperature: T) -> u32
where
    T: Float + PartialOrd + Clone + Copy + Default,
{
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= T::zero() || top_k < 2 || top_p <= T::zero() {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }

    impl<T: Float> From<(usize, &T)> for Probability {
        #[inline]
        fn from((i, p): (usize, &T)) -> Self {
            Self {
                val: p.to_f32().unwrap(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.0);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature.to_f32().unwrap()).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p.to_f32().unwrap();
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

#[test]
fn test_rope() {
    // 创建一个输入张量 y，大小为 [3, 2, 4]，即 seq_len=3, n_heads=2, d=4
    let mut y = Tensor::<f32>::new(
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0
        ], 
        &vec![3, 2, 4]
    );
    
    // 设置起始位置和 theta
    let start_pos = 0;
    let theta = 10000.0_f32;

    // 调用 rope 函数
    rope(&mut y, start_pos, theta);

    // 创建期望的输出张量，假设我们计算出以下值（这仅为示例，实际值需要根据公式计算）
    let expected = Tensor::<f32>::new(vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        7.0, 8.0, -4.3934603, 9.879502, 13.516563, 12.099399, 
        -5.598135, 13.839303, 19.043655, 16.139198, -24.351147, 17.596428, 
        7.5512652, 20.355976, -29.652924, 21.515633, 9.523868, 24.435171], &vec![3, 2, 4]); // 根据实际计算修改这里的值
    // 检查结果是否接近预期值
    assert!(y.close_to(&expected, 1e-3)); // 允许精度误差为 1e-3
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    swiglu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}
#[test]
fn test_masked_softmax() {
    // 初始化输入张量 y
    let mut y = Tensor::<f32>::new(
        vec![
            1.0, 2.0, 3.0, 4.0, // 第一个序列
            5.0, 6.0, 7.0, 8.0, // 第二个序列
        ],
        &vec![2, 4], // 形状为 [batch=2, seq_len=4]
    );

    // 调用 masked_softmax 函数
    masked_softmax(&mut y);

    // 验证结果是否接近预期值
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![
                0.09003057, 0.24472847, 0.66524095, 0.0, // 第一个序列
                0.0320586, 0.08714432, 0.23688282, 0.6439142, // 第二个序列
            ],
            &vec![2, 4],
        ),
        1e-3, // 允许的误差范围
    ));
}
use cust::prelude::Context;
use std::thread;
#[cfg(feature = "gpu")]
static mut _CTX:Option<Context>=None;
#[cfg(feature = "gpu")]
static mut STREAM:Option<Stream>=None;
#[cfg(feature = "gpu")]
static mut MODULE:Option<Module>=None;
#[cfg(feature = "gpu")]
static PTX:&str=include_str!("../cuda/operator.ptx");
#[cfg(feature = "gpu")]
static mut CURRENT_THREADID:Option<thread::ThreadId>=None;
#[cfg(feature = "gpu")]
fn get_threadid()->&'static thread::ThreadId{
    unsafe {
        let id=match CURRENT_THREADID.as_mut() {
            Some(x)=>x,
            None=>{
                let x =thread::current().id();
                CURRENT_THREADID=Some(x);
                CURRENT_THREADID.as_ref().unwrap()
            }
        };
        id
    }
}
#[cfg(feature = "gpu")]
fn get_ctx()->&'static Context{
    unsafe {
        let _ctx=match _CTX.as_mut() {
            Some(x)=>{
                if get_threadid().eq(&thread::current().id()){
                    x
                }else{
                    let x: cust::prelude::Context = cust::quick_init().unwrap();
                    _CTX=Some(x);
                    _CTX.as_ref().unwrap()
                }
            },
            None=>{
                let x: cust::prelude::Context = cust::quick_init().unwrap();
                _CTX=Some(x);
                _CTX.as_ref().unwrap()
            }
        };
        _ctx
    }
}
#[cfg(feature = "gpu")]
fn get_stream()->&'static Stream{
    unsafe {
        get_ctx();
        let stream=match STREAM.as_mut() {
            Some(x)=>{
                if get_threadid().eq(&thread::current().id()){
                    x
                }else{
                    let x=Stream::new(cust::stream::StreamFlags::DEFAULT, None).unwrap();
                    STREAM=Some(x);
                    STREAM.as_ref().unwrap()
                }
            },
            None=>{
                let x=Stream::new(cust::stream::StreamFlags::DEFAULT, None).unwrap();
                STREAM=Some(x);
                STREAM.as_ref().unwrap()
            }
        };
        stream
    }
}
#[cfg(feature = "gpu")]
fn get_module()->&'static Module{
    unsafe {
        let module=match MODULE.as_mut() {
            Some(x)=>{
                if get_threadid().eq(&thread::current().id()){
                    x
                }else{  
                    let x: Module = Module::from_ptx(PTX, &[]).unwrap();
                    MODULE=Some(x);
                    MODULE.as_ref().unwrap()
                }
            },
            None=>{
                let x: Module = Module::from_ptx(PTX, &[]).unwrap();
                MODULE=Some(x);
                MODULE.as_ref().unwrap()
            }
        };
        module
    }
}
// #[cfg(feature="gpu")]
// pub fn rope<T>(
//     y: &mut Tensor<T>,
//     start_pos: usize,
//     theta: T,
// ) where
//     T: Float + Sub<Output = T> + Mul<Output = T> + Copy + Clone + Default + DeviceCopy,
// {
//     let shape = y.shape();
//     assert!(shape.len() == 3);
//     let seq_len = shape[0];
//     let n_heads = shape[1];
//     let d = shape[2];

//     // 初始化 CUDA 环境
//     let _ctx = cust::quick_init().unwrap();
//     let stream = Stream::new(cust::stream::StreamFlags::DEFAULT, None).unwrap();

//     // 加载 PTX 模块
//     let ptx = include_str!("../cuda/rope_kernel.ptx");
//     let module = Module::from_ptx(ptx, &[]).unwrap();

//     // 获取内核函数
//     let kernel = module.get_function("rope_kernel").unwrap();

//     // 将数据复制到 GPU
//     let mut y_gpu = DeviceBuffer::from_slice(y.data()).unwrap();

//     // 定义线程块和网格大小
//     let threads_per_block = 256; // 每个线程块处理 256 个元素
//     let blocks_per_grid = ((seq_len * n_heads * d) as u32 + threads_per_block - 1) / threads_per_block;

//     // 启动内核
//     unsafe {
//         launch!(
//             kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
//                 y_gpu.as_device_ptr(),
//                 seq_len as i32,
//                 n_heads as i32,
//                 d as i32,
//                 start_pos as i32,
//                 theta
//             )
//         )
//         .unwrap();
//     }

//     // 同步流以确保计算完成
//     stream.synchronize().unwrap();

//     // 将结果从 GPU 复制回主机
//     unsafe {
//         y_gpu.copy_to(y.data_mut()).unwrap();
//     }
// }
use crate::SuperTrait;
use crate::types::F32;
#[cfg(feature="gpu")]
pub fn masked_softmax<T:SuperTrait>(
    y: &mut Tensor<T>  // 确保使用f32类型
){
    use crate::types::F32;

    

    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);

    let stream = get_stream();
    let module = get_module();

    // 获取内核函数
    let kernel = module.get_function("masked_softmax_kernel").unwrap();

     // 将数据复制到 GPU
     let y_gpu = DeviceBuffer::from_slice(y.data().iter().map(|&x|x.as_f32()).collect::<Vec<f32>>().as_slice()).unwrap();

    // 计算执行配置
    let threads_per_block:u32 = 256;
    let num_rows = batch * seq_len;
    let blocks_per_grid = num_rows as u32;
    let shared_mem_bytes:u32 = 2 * threads_per_block * std::mem::size_of::<f32>() as u32;

    // 启动内核
    unsafe {
        launch!(
            kernel<<<blocks_per_grid, threads_per_block, shared_mem_bytes, stream>>>(
                y_gpu.as_device_ptr(),
                batch as i32,
                seq_len as i32,
                total_seq_len as i32
            )
        ).unwrap();
    }

    // 同步并传回数据
    stream.synchronize().unwrap();
    unsafe {
        let mut temp_f32 = vec![0.0f32; y_gpu.len()];
        y_gpu.copy_to(&mut temp_f32).unwrap();
        let converted: Vec<T> = temp_f32.into_iter().map(|x| <T as F32>::from_f32(x)).collect();
        y.data_mut().copy_from_slice(&converted);
    }

}

#[cfg(feature="gpu")]
pub fn rms_norm<T>(
    y: &mut Tensor<T>,
    x: &Tensor<T>,
    w: &Tensor<T>,
    epsilon: T,
) where
    T: SuperTrait
{
    let device_id = 0; 
    let dim = x.shape()[x.shape().len() - 1];
    let batch = x.size() / dim;
    let stream = get_stream();
    let module = get_module();
    let kernel = module.get_function("rms_norm_kernel").unwrap();

    // 将数据复制到 GPU
    let x_gpu = DeviceBuffer::from_slice(x.data().iter().map(|&x|x.as_f32()).collect::<Vec<f32>>().as_slice()).unwrap();
    let w_gpu = DeviceBuffer::from_slice(w.data().iter().map(|&x|x.as_f32()).collect::<Vec<f32>>().as_slice()).unwrap();
    let mut y_gpu = DeviceBuffer::from_slice(y.data().iter().map(|&x|x.as_f32()).collect::<Vec<f32>>().as_slice()).unwrap();

    // 定义线程块和网格大小
    let threads_per_block = 256; // 每个线程块处理 256 个元素
    let blocks_per_grid = ((batch * dim) as u32 + threads_per_block - 1) / threads_per_block;

    // 启动内核
    unsafe {
        launch!(
            kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>( 
                x_gpu.as_device_ptr(),
                w_gpu.as_device_ptr(),
                y_gpu.as_device_ptr(),
                dim as i32,
                batch as i32,
                epsilon.as_f32()
            )
        )
        .unwrap();
    }

    // 同步流以确保计算完成
    stream.synchronize().unwrap();
    
    // 将结果从 GPU 复制回主机
    unsafe {
        let mut temp_f32 = vec![0.0f32; y_gpu.len()];
        y_gpu.copy_to(&mut temp_f32).unwrap();
        let converted: Vec<T> = temp_f32.into_iter().map(|x| <T as F32>::from_f32(x)).collect();
        y.data_mut().copy_from_slice(&converted);
    }
}

#[cfg(feature="gpu")]
pub fn swiglu<T>(
    y: &mut Tensor<T>,
    x: &Tensor<T>,
) where
    T: SuperTrait,
{
    let len = y.size();
    assert_eq!(len, x.size(), "输入和输出张量的大小必须相同");

    let stream = get_stream();
    let module = get_module();

    // 获取内核函数
    let kernel = module.get_function("swiglu_kernel").unwrap();

    // 将数据复制到 GPU
    let x_gpu = DeviceBuffer::from_slice(x.data().iter().map(|&x|x.as_f32()).collect::<Vec<f32>>().as_slice()).unwrap();
    let y_gpu = DeviceBuffer::from_slice(y.data().iter().map(|&x|x.as_f32()).collect::<Vec<f32>>().as_slice()).unwrap();

    // 定义线程块和网格大小
    let threads_per_block = 256; // 每个线程块处理 256 个元素
    let blocks_per_grid = (len as u32 + threads_per_block - 1) / threads_per_block;

    // 启动内核
    unsafe {
        launch!(
            kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
                x_gpu.as_device_ptr(),
                y_gpu.as_device_ptr(),
                len as i32
            )
        )
        .unwrap();
    }

    // 同步流以确保计算完成
    stream.synchronize().unwrap();

    // 将结果从 GPU 复制回主机
    unsafe {
        let mut temp_f32 = vec![0.0f32; y_gpu.len()];
        y_gpu.copy_to(&mut temp_f32).unwrap();
        let converted: Vec<T> = temp_f32.into_iter().map(|x| <T as F32>::from_f32(x)).collect();
        y.data_mut().copy_from_slice(&converted);
    }
}

#[cfg(feature="gpu")]
pub fn matmul_transb<T>(
    c: &mut Tensor<T>,
    beta: T,
    a: &Tensor<T>,
    b: &Tensor<T>,
    alpha: T,
) where
    T: Float + Mul<Output = T> + Add<Output = T> + Copy + Clone + Default + F32,
{

    // 获取矩阵维度

    use crate::types::F32;
    let dim1 = a.shape()[0];
    let dim2 = b.shape()[0];
    let dim3 = a.shape()[1];

    // 检查维度是否匹配
    assert_eq!(a.shape()[1], b.shape()[1], "矩阵 a 和 b 的列数不匹配");
    assert_eq!(c.shape()[0], dim1, "矩阵 c 的行数与矩阵 a 的行数不匹配");
    assert_eq!(c.shape()[1], dim2, "矩阵 c 的列数与矩阵 b 的行数不匹配");

    let stream = get_stream();
    let module = get_module();

    // 获取内核函数
    let kernel = module.get_function("matmul_transb_kernel").unwrap();

    // 将数据复制到 GPU
    // 转换为f32数据
    let a_gpu = DeviceBuffer::from_slice(a.data().iter().map(|&x|x.as_f32()).collect::<Vec<f32>>().as_slice()).unwrap();
    let b_gpu = DeviceBuffer::from_slice(b.data().iter().map(|&x|x.as_f32()).collect::<Vec<f32>>().as_slice()).unwrap();
    let c_gpu = DeviceBuffer::from_slice(c.data().iter().map(|&x|x.as_f32()).collect::<Vec<f32>>().as_slice()).unwrap();

    // 定义线程块和网格大小
    let threads_per_block = (16, 16); // 每个线程块处理 16x16 的子矩阵
    let blocks_per_grid = (
        (dim2 as u32 + threads_per_block.0 - 1) / threads_per_block.0,
        (dim1 as u32 + threads_per_block.1 - 1) / threads_per_block.1,
    );

    // 启动内核
    unsafe {
        launch!(
            kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
                a_gpu.as_device_ptr(),
                b_gpu.as_device_ptr(),
                c_gpu.as_device_ptr(),
                dim1 as i32,
                dim2 as i32,
                dim3 as i32,
                alpha.as_f32(),
                beta.as_f32()
            )
        )
        .unwrap();
    }

    // 同步流以确保计算完成
    stream.synchronize().unwrap();
    unsafe {
        let mut temp_f32 = vec![0.0f32; c_gpu.len()];
        c_gpu.copy_to(&mut temp_f32).unwrap();
        let converted: Vec<T> = temp_f32.into_iter().map(|x| <T as F32>::from_f32(x)).collect();
        c.data_mut().copy_from_slice(&converted);
    }
}