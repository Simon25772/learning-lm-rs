mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
mod api;
use config::LlamaConfigJson;
use cust::memory::DeviceCopy;
use half::{f16, bf16};
use num_traits::Float;
use std::{fs::File, io::{Error, Write}, iter::Sum, ops::{AddAssign, DivAssign, Mul}, path::PathBuf, time::Instant};
use tokenizers::Tokenizer;
use api::start_api;



// SuperTrait
pub trait SuperTrait: 'static + Send + Sync
    + Float
    + AddAssign
    + Mul<Output = Self>
    + DivAssign
    + Copy
    + Clone
    + Default
    + DeviceCopy
    + Sum<Self>{}
    

impl<T> SuperTrait for T where 
T:'static + Send + Sync
    + Float
    + AddAssign
    + Mul<Output = T>
    + DivAssign
    + Copy
    + Clone
    + Default
    + DeviceCopy
    + Sum<T>{}

fn get_model_config(name:&str)->Result<LlamaConfigJson, Error>{
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join(name);
    
    // 使用 as_path() 转换为 &Path
    let config_file_path = model_dir.as_path().join("config.json");
    let config_file = File::open(&config_file_path)?; // 打开文件
    
    // 从文件读取 JSON 数据并反序列化
    let config: LlamaConfigJson = serde_json::from_reader(config_file)?;
    Ok(config)
}

fn chat_start<T>()
where T:SuperTrait
{
    // 加载引擎
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<T>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    // 创建kv cache
    let mut cache = llama.new_cache();
    loop{
        println!("\nYou:");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        input = input.trim().to_string();
        // 使用Jinja2模板引擎
        input = format!("<|im_start|>{}\n{}<|im_end|>\n<|im_start|>assistant\n", "user", input);
        // println("DEBUG!input:{:?}", input);
        let binding = tokenizer.encode(input, true).unwrap();
        let input_ids = binding.get_ids();
        // println!("DEBUG!input_ids:{:?}", input_ids);
        println!("Assistant:");
        let start_time = Instant::now();
        // 使用推荐参数do_sample
        // 推理太慢了，使用迭代器提高交互速度
        let output_iter = llama.generate_iter(
            input_ids,
            256,
            T::from(0.9).unwrap(),
            1,
            T::from(1.0).unwrap(),
            &mut cache
        );
        // 使用迭代器输出
        for output_id in output_iter{
            // 适当添加空格然后输出
            let word = tokenizer.decode(&vec![output_id], true).unwrap();
            let word = if word.chars().all(|c| c.is_alphabetic()){
                " ".to_string() + &word
            }else{
                word
            };
            print!("{}", word);
            std::io::stdout().flush().unwrap();
        }
        let duration = start_time.elapsed();
        println!("Time taken: {:?}", duration);
    }
}

fn story_start<T>()
where T:SuperTrait
{
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<T>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = "Once upon a time";
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    print!("\n{}", input);
    let output_ids = llama.generate(
        input_ids,
        200,
        T::from(0.8).unwrap(),
        1,
        T::from(1.).unwrap(),
    );
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}

fn run_chat_start(){
    match get_model_config("chat").unwrap().torch_dtype.as_str() {
        "float32" => chat_start::<f32>(),
        // "float16" => chat_start::<f16>(),
        // "bfloat16" => chat_start::<bf16>(),
        _=> panic!("Unsupported dtype!"),
    }
}

fn run_story_start(){
    let start_time = Instant::now();
    match get_model_config("story").unwrap().torch_dtype.as_str() {
        "float32" => story_start::<f32>(),
        // "float16" => story_start::<f16>(),
        // "bfloat16" => story_start::<bf16>(),
        _=> panic!("Unsupported dtype!"),
    }
    let duration = start_time.elapsed();
    println!("Time taken: {:?}", duration);
}

fn start(){
    println!("\nWelcome to Llama Chatbot!");
    println!("Please select a mode:");
    println!("1. Chat mode");
    println!("2. Story mode");
    println!("3. API service");
    println!("4. Exit");
    let mut mode = String::new();
    std::io::stdin().read_line(&mut mode).unwrap();
    let mode = mode.trim();
    match mode{
        "1" => run_chat_start(),
        "2" => run_story_start(),
        "3" => start_api().unwrap(),
        "4" => std::process::exit(0),
        _ => println!("Invalid mode!"),
    }
}

// CONSTANT
const NUM_DEVICE:usize=4;

fn main() {
    start();
}
