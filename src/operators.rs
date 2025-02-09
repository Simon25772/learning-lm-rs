use crate::tensor::Tensor;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::cmp::Ordering;
use num_traits::float::Float; // 使用 num_traits 库中的 Float trait 来支持浮点数操作

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
