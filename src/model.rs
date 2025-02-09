use std::fs::File;
use std::ops::{Add, AddAssign, DivAssign, Mul};
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators::{self as OP, matmul_transb, rms_norm, swiglu};
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use num_traits::Float;
use rand::distributions::uniform::SampleUniform;
use safetensors::SafeTensors;
use serde::de;
use std::path::Path;
pub struct Llama<T> {
    vocab: usize,           // vocab size
    n_layers: usize,        // number of layers
    n_q_h: usize,           // number of heads for q
    n_kv_h: usize,          // number of heads for k and v
    d: usize,               // dimension of hidden states
    dqkv: usize,            // length of a single q, k, or v vector
    di: usize,              // dimension of intermediate states
    eps: f32,               // epsilon for RMS normalization
    rope_theta: f32,        // rope theta for rope initialization
    max_seq_len: usize,     // maximum sequence length
    params: LLamaParams<T>, // trained weights of this model
    bos_token_id: u32,      // start token id
    pub eos_token_id: u32,      // end token id
}

impl<T> Llama<T> 
where 
    T: Float + std::ops::AddAssign + std::ops::Mul<Output = T> + std::ops::DivAssign + Copy + Add<Output = T> + Clone + Default + std::iter::Sum
{
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();
        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params: params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<T> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<T>) -> Tensor<T> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        // Some pre-allocated buffers that will be reused
        let mut residual = Tensor::<T>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<T>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<T>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores = Tensor::<T>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<T>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<T>::default(&vec![seq_len, self.di]);

        // Computation Starts Here
        OP::gather(&mut residual, input, &self.params.embedding_table);
        
        for layer in 0..self.n_layers {
            OP::rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                T::from(self.eps).unwrap(),
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            OP::matmul_transb(q, T::zero(), &hidden_states, &self.params.wq[layer], T::one());
            OP::matmul_transb(k, T::zero(), &hidden_states, &self.params.wk[layer], T::one());
            OP::matmul_transb(v, T::zero(), &hidden_states, &self.params.wv[layer], T::one());
            OP::rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                T::from(self.rope_theta).unwrap(),
            );
            OP::rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                T::from(self.rope_theta).unwrap(),
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            self_attention(
                &mut hidden_states,              // 存储注意力机制的输出张量
                &mut att_scores,      // 存储注意力得分的张量
                q,                   // Query 张量，表示查询向量
                &full_k,             // Key 张量，表示键向量
                &full_v,             // Value 张量，表示值向量
                self.n_kv_h,         // Key 和 Value 的头数量
                n_groups,            // 注意力头的分组数量
                seq_len,             // 输入序列的长度
                total_seq_len,       // 总序列长度（包括当前序列和缓存的过往序列）
                self.dqkv,           // 单个 Query、Key 或 Value 向量的维度
            );

            matmul_transb(&mut residual, T::one(), &hidden_states, &self.params.wo[layer], T::one());

            mlp(
                &mut residual,                                  // residual 被传入 MLP 层处理
                &mut hidden_states,                             // 处理结果存储在 hidden_states 中
                &mut gate_buf,                                  // gate_buf 是 MLP 中的中间结果
                &mut up_buf,                                    // up_buf 是另一个中间结果
                &self.params.w_up[layer],                       // w_up 是 MLP 层的上投影矩阵
                &self.params.w_down[layer],                     // w_down 是 MLP 层的下投影矩阵
                &self.params.w_gate[layer],                     // w_gate 是 MLP 层的门控权重
                &self.params.rms_ffn_w[layer],                  // RMS 归一化的权重
                T::from(self.eps).unwrap(),                                       // 归一化的 epsilon
            );
        }

        let mut logits = Tensor::<T>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        OP::rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            T::from(self.eps).unwrap(),
        );

        OP::matmul_transb(&mut logits, T::zero(), &hidden_states, &self.params.lm_head, T::one());

        logits
    }
}


impl<T> Llama<T>
where
    T: Float + AddAssign + Mul<Output = T> + DivAssign + Copy + Clone + Default + std::iter::Sum
{
    pub fn generate(
        &self,
        token_ids: &[u32],
        max_len: usize,
        top_p: T,
        top_k: u32,
        temperature: T
    ) -> Vec<u32> {
        // 用于存储最终生成的 token 序列
        let mut generated_tokens = Vec::<u32>::new();
        // 初始化 key - value 缓存，缓存可在每一层计算中复用
        let mut kv_cache = self.new_cache();
        // 把输入的 token 序列复制到一个新的 Vec 中
        let input_tokens: Vec<u32> = token_ids.to_vec();
        // 将 token 序列转换为张量，输入张量是二维的，形状为 (1, token_ids 的长度)
        let mut input_tensor = Tensor::<u32>::new(input_tokens, &vec![1, token_ids.len()]);
        // 开始生成循环，持续生成直到达到最大长度限制
        while generated_tokens.len() < max_len {
            // 执行前向传播操作，得到每个词的未归一化概率分布（logits）
            let raw_prob_distribution = self.forward(&input_tensor, &mut kv_cache);
            // 根据 top_p、top_k 和 temperature 策略，从 logits 中采样得到下一个 token
            let next_generated_token = OP::random_sample(
                &raw_prob_distribution,
                top_p,
                top_k,
                temperature,
            );
            // 将新生成的 token 添加到最终结果列表中
            generated_tokens.push(next_generated_token);
            // 检查是否生成了结束标记（EOS），如果是则停止生成过程
            if next_generated_token == self.eos_token_id {
                break;
            }
            // 更新输入张量，将新生成的 token 作为下一次生成的输入
            input_tensor = Tensor::<u32>::new(vec![next_generated_token], &vec![1, 1]);
        }
        // 返回最终生成的 token 序列
        generated_tokens
    }

    // 返回一个迭代器
    pub fn generate_iter<'a>(
        &'a self,
        token_ids: &[u32],
        max_len: usize,
        top_p: T,
        top_k: u32,
        temperature: T,
        mut cache: &'a mut KVCache<T>,
    ) -> impl Iterator<Item = u32> + 'a {
        // 用于存储生成的 token 序列
        let mut generated_token_sequence = Vec::<u32>::new();
        // 将输入的 token 序列复制到一个新的 Vec 中
        let input_token_vec: Vec<u32> = token_ids.to_vec();
        // 把 token 序列转换为二维张量，形状为 (1, token_ids 的长度)
        let mut input_tensor = Tensor::<u32>::new(input_token_vec, &vec![1, token_ids.len()]);
        // 创建一个迭代器，通过闭包逻辑来生成 token
        std::iter::from_fn(move || {
            // 检查是否达到最大生成长度，如果达到则停止迭代
            if generated_token_sequence.len() >= max_len {
                return None;
            }
            // 执行前向传播，得到每个词的未归一化概率分布
            let probability_distribution = self.forward(&input_tensor, &mut cache);
            // 根据 top_p、top_k 和 temperature 策略从概率分布中采样下一个 token
            let next_generated_token = OP::random_sample(
                &probability_distribution,
                top_p,
                top_k,
                temperature,
            );
            // 将新生成的 token 添加到生成序列中
            generated_token_sequence.push(next_generated_token);
            // 检查是否生成了结束标记（EOS），如果是则停止迭代
            if next_generated_token == self.eos_token_id {
                return None;
            }
            // 更新输入张量，将新生成的 token 作为下一次的输入
            input_tensor = Tensor::<u32>::new(vec![next_generated_token], &vec![1, 1]);
            // 返回新生成的 token
            Some(next_generated_token)
        })
    }
}


fn self_attention<T>(
    hidden_states: &mut Tensor<T>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<T>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<T>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<T>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<T>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,
    n_groups: usize,
    seq_len: usize,
    total_seq_len: usize,
    dqkv: usize,
) where T: Float + Mul<Output = T> + Add<Output = T> + Copy + Clone + Default + std::iter::Sum + std::ops::AddAssign + std::ops::DivAssign{
   // 获取各张量的数据指针
   let att_scores_data = unsafe { att_scores.data_mut() };
   let q_data = q.data();
   let k_data = k.data();
   let v_data = v.data();
   // 计算归一化因子，即 dqkv 的平方根
   let norm_factor = T::from(dqkv as f32).unwrap().sqrt();
   // 第一步：计算注意力分数，公式为 score = Q @ K.T / sqrt(dim)
   for head_group in 0..n_kv_h * n_groups {
       for query_pos in 0..seq_len {
           for key_pos in 0..total_seq_len {
               // 初始化点积的累加和
               let mut dot_prod = T::zero();
               // 对 Query 和 Key 向量的每个维度进行点积计算
               for dim in 0..dqkv {
                   let q_index = query_pos * n_kv_h * n_groups * dqkv + head_group * dqkv + dim;
                   let k_index = key_pos * n_kv_h * dqkv + (head_group / n_groups) * dqkv + dim;
                   dot_prod += q_data[q_index] * k_data[k_index];
               }
               // 计算注意力分数的存储位置
               let score_index = head_group * seq_len * total_seq_len + query_pos * total_seq_len + key_pos;
               // 存储归一化后的点积结果
               att_scores_data[score_index] = dot_prod / norm_factor;
           }
       }
   }
   // 第二步：对注意力分数应用 Softmax 函数，将其转换为概率分布
   OP::masked_softmax(att_scores);
   // 第三步：计算最终的隐藏状态，公式为 x = attn @ V
   let att_scores = att_scores.data();
   let hidden_states_data = unsafe { hidden_states.data_mut() };
   for head_group in 0..n_kv_h * n_groups {
       for query_pos in 0..seq_len {
           for dim in 0..dqkv {
               // 初始化加权和
               let mut weighted_sum = T::zero();
               // 对所有 Key 位置进行加权求和
               for key_pos in 0..total_seq_len {
                   let att_score_index = head_group * seq_len * total_seq_len + query_pos * total_seq_len + key_pos;
                   let v_index = dim + (head_group / n_groups) * dqkv + key_pos * n_kv_h * dqkv;
                   weighted_sum += att_scores[att_score_index] * v_data[v_index];
               }
               // 计算隐藏状态的存储位置
               let hidden_index = query_pos * n_kv_h * n_groups * dqkv + head_group * dqkv + dim;
               // 存储加权和结果
               hidden_states_data[hidden_index] = weighted_sum;
           }
       }
   }
}

fn mlp<T>(
    residual: &mut Tensor<T>,
    hidden_states: &mut Tensor<T>,
    gate: &mut Tensor<T>,
    up: &mut Tensor<T>,
    w_up: &Tensor<T>,
    w_down: &Tensor<T>,
    w_gate: &Tensor<T>,
    rms_w: &Tensor<T>,
    eps: T,
) where T:Float + Mul<Output = T> + Add<Output = T> + Copy + Clone + Default + std::iter::Sum + std::ops::AddAssign{
    // 对 residual 进行 RMS 归一化，结果存储在 hidden_states 中
    rms_norm(hidden_states, residual, rms_w, eps);

    // 进行矩阵乘法
    matmul_transb(gate, T::zero(), hidden_states, w_gate, T::one());
    matmul_transb(up, T::zero(), hidden_states, w_up, T::one());

    // 计算 SwiGLU 激活函数
    swiglu(up, gate);

    // 进行矩阵乘法
    matmul_transb(hidden_states, T::zero(), up, w_down, T::one());

    // 残差连接
    unsafe {
        residual.data_mut().iter_mut()
        .zip(hidden_states.data().iter())
        .for_each(|(r, h)| *r += *h);
    }
}


#[test]
pub fn test_mlp() {
    let seq_len = 4;
    let d = 2;
    let di = 3;
    let mut residual = Tensor::<f32>::new(vec![1., 1., 1., 1., 1., 1., 1., 1.], &vec![seq_len, d]);
    let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, d]);
    let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let mut up_buf = Tensor::<f32>::default(&vec![seq_len, di]);
    let w_up = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let w_down = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![d, di]);
    let w_gate = Tensor::<f32>::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], &vec![di, d]);
    let rms_w = Tensor::<f32>::new(vec![1., 1.], &vec![d]);
    let eps = 1e-6;
    mlp(
        &mut residual,
        &mut hidden_states,
        &mut gate_buf,
        &mut up_buf,
        &w_up,
        &w_down,
        &w_gate,
        &rms_w,
        eps,
    );

    assert!(residual.close_to(
        &Tensor::<f32>::new(
            vec![
                1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964, 1.7290739, 1.3429964,
                1.7290739
            ],
            &vec![seq_len, d]
        ),
        1e-3
    ))
}

#[test]
pub fn test_load_safetensors() {
    use std::path::PathBuf;
    use crate::tensor::float_eq;
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let model = Llama::from_safetensors(model_dir);
    assert_eq!(model.vocab, 2048);
    assert_eq!(model.n_layers, 2);
    assert_eq!(model.n_q_h, 8);
    assert_eq!(model.n_kv_h, 4);
    assert_eq!(model.d, 128);
    assert_eq!(model.dqkv, 16);
    assert_eq!(model.di, 384);

    assert!(float_eq(&model.params.embedding_table.data()[50], &0.14453125, 1e-6));
    assert_eq!(model.params.lm_head.data()[10], model.params.embedding_table.data()[10]);
    assert!(float_eq(&model.params.rms_att_w[0].data()[10], &0.18652344, 1e-6));
    assert!(float_eq(&model.params.rms_ffn_w[1].data()[10], &0.32421875, 1e-6));
    assert!(float_eq(&model.params.rms_out_w.data()[100], &0.73046875, 1e-6));
    assert!(float_eq(&model.params.w_down[0].data()[100], &-0.0625, 1e-6));
    assert!(float_eq(&model.params.w_up[0].data()[100], &1.46875, 1e-6));
    assert!(float_eq(&model.params.w_gate[1].data()[100], &0.296875, 1e-6));
    assert!(float_eq(&model.params.wq[1].data()[100], &0.032226563, 1e-6));
    assert!(float_eq(&model.params.wk[1].data()[100], &-0.21386719, 1e-6));
    assert!(float_eq(&model.params.wv[0].data()[100], &0.041015625, 1e-6));
    assert!(float_eq(&model.params.wo[0].data()[100], &0.01965332, 1e-6));

}
