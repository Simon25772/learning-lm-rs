mod config;
mod kvcache;
mod model;
mod operators;
mod params;
mod tensor;
use config::LlamaConfigJson;
use half::{f16, bf16};
use kvcache::KVCache;
use num_traits::Float;
use rand::Rng;
use std::{collections::HashMap, f64::consts::E, fs::File, io::Write, path::PathBuf, sync::Mutex, time::Instant};
use tokenizers::Tokenizer;
use actix_web::{get, web::{self, Data}, App, HttpServer, Responder};
use actix_session::Session;
use actix_session::CookieSession;

fn chat_start<T>()
where T:'static+Send+Sync+ Float
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
where T: 'static+Send+Sync+Float
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
        1,
        T::from(1.).unwrap(),
    );
    println!("{}", tokenizer.decode(&output_ids, true).unwrap());
}

fn story_start_for_api<T>(prompt:String)->String
where T: 'static+Send+Sync+Float
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
    let start_time = Instant::now();
    match config.torch_dtype.as_str() {
        "float32" => story_start::<f32>(),
        "float16" => story_start::<f16>(),
        "bfloat16" => story_start::<bf16>(),
        _=> panic!("Unsupported dtype!"),
    }
    let duration = start_time.elapsed();

    // 打印运行时间（秒和毫秒）
    println!("Time taken: {:?}", duration);
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

#[get("/story")]
async fn story_api_without_prompt() -> impl Responder {
    run_story_start_for_api("Once upon a time".to_string())
}
#[get("/story/{prompt}")]
async fn story_api(prompt: web::Path<String>) -> impl Responder {
    run_story_start_for_api(prompt.to_string())
}

fn chat_start_for_api<T>(user_id:i32, data: &web::Data<Mutex<Status<T>>>, prompt:String)->String
where T: 'static+Send+Sync+Float
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
    // 获取or创建kv cache
    let mut data = data.lock().unwrap(); // 锁住 Mutex
    let tmp=data.memory.entry(user_id).or_insert(llama.new_cache());
    // 构造模板
    let mut chat_result = "".to_string();
    println!("\nYou:");
    let mut input = prompt.clone();
    // 使用Jinja2模板引擎
    input = format!("<|im_start|>{}\n{}<|im_end|>\n<|im_start|>assistant\n", "user", input);
    // inputembedding
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    println!("Assistant:");
    // 使用推荐参数do_sample
    // 推理太慢了，使用迭代器提高交互速度
    let output_iter = llama.generate_iter(
        input_ids,
        256,
        T::from(0.9).unwrap(),
        4,
        T::from(1.0).unwrap(),
        tmp
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

//session_id
static mut COUNTER:i32=0;
async fn chat_api<T>(data: web::Data<Mutex<Status<T>>>, prompt:web::Path<String>, session: Session)->impl Responder
where T: 'static+Send+Sync+Float
+ std::ops::AddAssign
+ std::ops::Mul<Output = T>
+ std::ops::DivAssign
+ Copy
+ Clone
+ Default
+ std::iter::Sum<T>
{
    // let prompt = String::from("My name is Lijie!");
    println!("CHAT!");
    if let Some(user_id) = session.get::<i32>("user_id").unwrap() {
        chat_start_for_api(user_id, &data, prompt.to_string())
    } else {
        unsafe {
            COUNTER+=1;
            session.insert("user_id", COUNTER).unwrap();
            chat_start_for_api(COUNTER, &data, prompt.to_string())
        }
    }
}
pub struct Status<T>
where T:Default{
    memory:HashMap<i32,KVCache<T>>,
}
impl<T> Status<T>
where T:Default
{
    fn new()->Self{
        Self{
            memory:HashMap::new(),
        }
    }
}
#[actix_web::main]
async fn start_api() -> std::io::Result<()> {
    println!("Server running at http://127.0.0.1:8080");
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("story");
    
    // 使用 as_path() 转换为 &Path
    let config_file_path = model_dir.as_path().join("config.json");
    let config_file = File::open(&config_file_path)?; // 打开文件
    
    // 从文件读取 JSON 数据并反序列化
    let config: LlamaConfigJson = serde_json::from_reader(config_file)?;
    match config.torch_dtype.as_str() {
        "float32" => {
            HttpServer::new(move || {
                App::new()
                    .wrap(CookieSession::signed(&[0; 32]).secure(false))
                    .app_data(web::Data::new(Mutex::new(Status::<f32>::new())))
                    .route("/chat/{prompt}", web::get().to(chat_api::<f32>))
                    .service(story_api)
                    .service(story_api_without_prompt)
            })
            .bind(("127.0.0.1", 8080))?
            .run()
            .await
        },
        "float16" => {
            HttpServer::new(move || {
                App::new()
                    .wrap(CookieSession::signed(&[0; 32]).secure(false))
                    .app_data(web::Data::new(Mutex::new(Status::<f16>::new())))
                    .route("/chat/{prompt}", web::get().to(chat_api::<f16>))
                    .service(story_api)
                    .service(story_api_without_prompt)
            })
            .bind(("127.0.0.1", 8080))?
            .run()
            .await
        },
        "bfloat16" => {
            HttpServer::new(move || {
                App::new()
                    .wrap(CookieSession::signed(&[0; 32]).secure(false))
                    .app_data(web::Data::new(Mutex::new(Status::<bf16>::new())))
                    .route("/chat/{prompt}", web::get().to(chat_api::<bf16>))
                    .service(story_api)
                    .service(story_api_without_prompt)
            })
            .bind(("127.0.0.1", 8080))?
            .run()
            .await
        },
        _ => panic!("Unsupported torch_dtype"),
    }
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
const NUM_DEVICE:usize=4;
fn main() {
    start();
}
