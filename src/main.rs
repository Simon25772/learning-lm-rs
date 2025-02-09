mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
use config::LlamaConfigJson;
use half::{f16, bf16};
use num_traits::Float;
use rand::distributions::uniform::SampleUniform;
use std::{fs::File, io::Write, ops::Add, path::PathBuf, time::Instant};
use model::Llama;
use tokenizers::Tokenizer;
use actix_web::{get, web, App, HttpServer, Responder};

fn chat_start<T>()
where T: Float
+ std::ops::AddAssign
+ std::ops::Mul<Output = T>
+ std::ops::DivAssign
+ Copy
+ Clone
+ Default
+ std::iter::Sum<T>
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
        // 使用推荐参数do_sample
        // 推理太慢了，使用迭代器提高交互速度
        let output_iter = llama.generate_iter(
            input_ids,
            256,
            T::from(0.9).unwrap(),
            4,
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
    }
}

fn chat_start_for_api<T>(prompt:String)->String
where T: Float
+ std::ops::AddAssign
+ std::ops::Mul<Output = T>
+ std::ops::DivAssign
+ Copy
+ Clone
+ Default
+ std::iter::Sum<T>
{
    // 加载引擎
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<T>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    // 创建kv cache
    let mut cache = llama.new_cache();
    let mut chat_result = "".to_string();
    println!("\nYou:");
    let mut input = prompt.clone();
    // 使用Jinja2模板引擎
    input = format!("<|im_start|>{}\n{}<|im_end|>\n<|im_start|>assistant\n", "user", input);
    // println("DEBUG!input:{:?}", input);
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    // println!("DEBUG!input_ids:{:?}", input_ids);
    println!("Assistant:");
    // 使用推荐参数do_sample
    // 推理太慢了，使用迭代器提高交互速度
    let output_iter = llama.generate_iter(
        input_ids,
        256,
        T::from(0.9).unwrap(),
        4,
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
        chat_result.push_str(&word);
        print!("{}", word);
        std::io::stdout().flush().unwrap();
    }
    chat_result
}



fn story_start<T>()
where T: Float
+ std::ops::AddAssign
+ std::ops::Mul<Output = T>
+ std::ops::DivAssign
+ Copy
+ Clone
+ Default
+ std::iter::Sum<T>
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
        30,
        T::from(1.).unwrap(),
    );
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}

fn story_start_for_api<T>(prompt:String)->String
where T: Float
+ std::ops::AddAssign
+ std::ops::Mul<Output = T>
+ std::ops::DivAssign
+ Copy
+ Clone
+ Default
+ std::iter::Sum<T>
{
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    let llama = model::Llama::<T>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    let input = prompt.as_str();
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    print!("\n{}", input);
    let output_ids = llama.generate(
        input_ids,
        200,
        T::from(0.8).unwrap(),
        30,
        T::from(1.).unwrap(),
    );
    let mut story = prompt.clone();
    story = story + &(tokenizer.decode(&output_ids, true).unwrap());
    println!("{}", story);
    story
}

fn run_chat_start(){
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    
    // 使用 as_path() 转换为 &Path
    let config_file_path = model_dir.as_path().join("config.json");
    let config_file = File::open(&config_file_path).unwrap(); // 打开文件
    
    // 从文件读取 JSON 数据并反序列化
    let config: LlamaConfigJson = serde_json::from_reader(config_file).unwrap();
    match config.torch_dtype.as_str() {
        "float32" => chat_start::<f32>(),
        "float16" => chat_start::<f16>(),
        "bfloat16" => chat_start::<bf16>(),
        _=> panic!("Unsupported dtype!"),
    }
}
fn run_story_start(){
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    
    // 使用 as_path() 转换为 &Path
    let config_file_path = model_dir.as_path().join("config.json");
    let config_file = File::open(&config_file_path).unwrap(); // 打开文件
    
    // 从文件读取 JSON 数据并反序列化
    let config: LlamaConfigJson = serde_json::from_reader(config_file).unwrap();
    match config.torch_dtype.as_str() {
        "float32" => story_start::<f32>(),
        "float16" => story_start::<f16>(),
        "bfloat16" => story_start::<bf16>(),
        _=> panic!("Unsupported dtype!"),
    }
}

fn run_story_start_for_api(prompt:String)->String{
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    
    // 使用 as_path() 转换为 &Path
    let config_file_path = model_dir.as_path().join("config.json");
    let config_file = File::open(&config_file_path).unwrap(); // 打开文件
    
    // 从文件读取 JSON 数据并反序列化
    let config: LlamaConfigJson = serde_json::from_reader(config_file).unwrap();
    match config.torch_dtype.as_str() {
        "float32" => story_start_for_api::<f32>(prompt),
        "float16" => story_start_for_api::<f16>(prompt),
        "bfloat16" => story_start_for_api::<bf16>(prompt),
        _=> "INNER ERROR".to_string(),
    }
}

fn run_chat_start_for_api(prompt:String)->String{
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    
    // 使用 as_path() 转换为 &Path
    let config_file_path = model_dir.as_path().join("config.json");
    let config_file = File::open(&config_file_path).unwrap(); // 打开文件
    
    // 从文件读取 JSON 数据并反序列化
    let config: LlamaConfigJson = serde_json::from_reader(config_file).unwrap();
    match config.torch_dtype.as_str() {
        "float32" => chat_start_for_api::<f32>(prompt),
        "float16" => chat_start_for_api::<f16>(prompt),
        "bfloat16" => chat_start_for_api::<bf16>(prompt),
        _=> "ERROR".to_string(),
    }
}

#[get("/story")]
async fn story_api_without_prompt() -> impl Responder {
    run_story_start_for_api("Once upon a time".to_string())
}
#[get("/story/{prompt}")]
async fn story_api(prompt: web::Path<String>) -> impl Responder {
    run_story_start_for_api(prompt.to_string())
}
#[get("/chat/{prompt}")]
async fn chat_api(prompt: web::Path<String>) -> impl Responder {
    run_chat_start_for_api(prompt.to_string())
}
#[actix_web::main] // or #[tokio::main]
async fn start_api() -> std::io::Result<()> {
    println!("Server running at http://127.0.0.1:8080");
    HttpServer::new(|| {
        App::new().service(story_api).service(chat_api).service(story_api_without_prompt)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
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

fn main() {
    // 记录开始时间
    let _start = Instant::now();

    start();

    // 记录结束时间并计算耗时
    let duration = _start.elapsed();

    // 打印运行时间
    println!("total time:{:?}", duration);
}
