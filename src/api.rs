use actix_cors::Cors;
use half::{f16, bf16};
use crate::kvcache::KVCache;
use std::{collections::HashMap, fs::File, path::PathBuf, sync::{Arc, Mutex}};
use tokenizers::Tokenizer;
use actix_web::{get, http, web::{self}, App, HttpResponse, HttpServer, Responder};
use actix_session::Session;
use actix_session::CookieSession;
use crate::model;
use crate::config::LlamaConfigJson;
use crate::get_model_config;
use crate::SuperTrait;

pub struct Status<T>
where T:Default{
    pub memory:HashMap<i32,KVCache<T>>,
}

impl<T> Status<T>
where T:Default
{
    pub fn new()->Self{
        Self{
            memory:HashMap::new(),
        }
    }
}

fn story_start_for_api<T>(prompt:String)->String
where T: SuperTrait
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

fn chat_start_for_api<T:SuperTrait>(user_id:i32, data: web::Data<Arc<Mutex<Status<T>>>>, prompt:String)->HttpResponse
{
    // 加载引擎
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<T>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    
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
    llama.generate_stream(
        input_ids,
        128,
        T::from(0.9).unwrap(),
        4,
        T::from(1.0).unwrap(),
        data,
        user_id,
        tokenizer
    )
}

//session_id
static mut COUNTER:i32=0;
async fn chat_api<T>(data: web::Data<Arc<Mutex<Status<T>>>>, prompt:web::Path<String>, session: Session)->HttpResponse
where T: SuperTrait
{   
    // let prompt = String::from("My name is Lijie!");
    println!("CHAT!");
    if let Some(user_id) = session.get::<i32>("user_id").unwrap() {
        chat_start_for_api(user_id, data, prompt.to_string())
    } else {
        unsafe {
            COUNTER+=1;
            session.insert("user_id", COUNTER).unwrap();
            chat_start_for_api(COUNTER, data, prompt.to_string())
        }
    }
}


async fn run_serve<T:SuperTrait>()->std::io::Result<()>{
    let data=Arc::new(Mutex::new(Status::<T>::new()));
    HttpServer::new(move || {
        let cors = Cors::default()
            .allowed_origin("http://localhost:8080") 
            .allowed_methods(vec!["GET", "POST", "PUT", "DELETE"]) 
            .allowed_headers(vec![http::header::AUTHORIZATION, http::header::ACCEPT])
            .allowed_header(http::header::CONTENT_TYPE)
            .max_age(3600); 
        App::new()
            .wrap(cors)
            .wrap(CookieSession::signed(&[0; 32]).secure(false))
            .app_data(web::Data::new(data.clone()))
            .route("/chat/{prompt}", web::get().to(chat_api::<T>))
            .service(story_api)
            .service(story_api_without_prompt)
    })
    .bind(("127.0.0.1", 8081))?
    .run()
    .await
}



#[actix_web::main]
pub async fn start_api() -> std::io::Result<()> {
    println!("Server running at http://127.0.0.1:8081");
    match get_model_config("chat").unwrap().torch_dtype.as_str() {
        "float32" => run_serve::<f32>().await,
        "float16" => run_serve::<f16>().await,
        "bfloat16" => run_serve::<bf16>().await,
        _ => panic!("Unsupported torch_dtype"),
    }
}