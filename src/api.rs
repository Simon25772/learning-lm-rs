use actix_cors::Cors;
use chrono::{DateTime, Utc};
use half::{f16, bf16};
use serde::Serialize;
use uuid::Uuid;
use crate::{kvcache::KVCache, types::F32};
use std::{collections::HashMap, default, fs::File, path::PathBuf, sync::{Arc, Mutex}};
use tokenizers::Tokenizer;
use actix_web::{get, http, web::{self}, App, HttpResponse, HttpServer, Responder};
use actix_session::CookieSession;
use crate::model;
use crate::get_model_config;
use crate::SuperTrait;

pub struct Status<T>
where T:Default{
    pub memory:HashMap<String,KVCache<T>>,
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

#[derive(Clone)]
#[derive(Debug)]
#[derive(Serialize)]
pub enum Role {
    AI,
    User,
    System
}

#[derive(Clone)]
#[derive(Debug)]
#[derive(Serialize)]
pub struct Message{
    pub role:Role,
    pub content:String,
    pub token_count:usize,
}

impl Message{
    pub fn new_with_role(role: Role) -> Self {
        Self {
            role,
            content: "".to_string(),
            token_count:0
        }
    }
    pub fn new(role: Role, content: String) -> Self {
        Self {
            role,
            content,
            token_count:0,
        }
    }
    pub fn add_count(&mut self, n:usize){
        self.token_count += n;
    }
}

pub struct MySession<T>{
    pub id:String,
    pub title:String,
    pub created_at: DateTime<Utc>,
    pub history:Vec<Message>,
    pub cache:Option<KVCache<T>>
}

impl<T:SuperTrait> MySession<T>{
    pub fn new()->Self{
        Self{
            id:Uuid::new_v4().to_string(),
            title:"New Chat".to_string(),
            created_at: Utc::now(),
            history:Vec::new(),
            cache:None
        }
    }
    pub fn callback(&mut self, index:usize){
        while self.history.len() > index{
            if let Some(msg) = self.history.pop() {
                if let Some(cache) = self.cache.as_mut() {
                    cache.decrement(msg.token_count);
                }
            }
        }
    }
}

#[derive(Serialize)]
pub struct MySessionData{
    pub id:String,
    pub title:String,
    pub created_at: DateTime<Utc>,
    pub history:Vec<Message>,
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
        <T as F32>::from_f32(0.8),
        30,
        <T as F32>::from_f32(1.),
    );
    let mut story = prompt.clone();
    story = story + &(tokenizer.decode(&output_ids, true).unwrap());
    println!("{}", story);
    story
}

fn run_story_start_for_api(prompt:String)->String{
    match get_model_config("story").unwrap().torch_dtype.as_str() {
        "float32" => story_start_for_api::<f32>(prompt),
        // "float16" => story_start_for_api::<f16>(prompt),
        // "bfloat16" => story_start_for_api::<bf16>(prompt),
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

fn chat_start_for_api<T:SuperTrait>(
    session:Arc<Mutex<MySession<T>>>,
    prompt:String)->HttpResponse
{
    // 加载引擎
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    let llama = model::Llama::<T>::from_safetensors(&model_dir);
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).unwrap();
    
    // 构造模板
    let mut input = prompt.clone();
    // 使用Jinja2模板引擎
    input = format!("<|im_start|>{}\n{}<|im_end|>\n<|im_start|>assistant\n", "user", input);
    // inputembedding
    let binding = tokenizer.encode(input, true).unwrap();
    let input_ids = binding.get_ids();
    // 使用推荐参数do_sample
    // 推理太慢了，使用迭代器提高交互速度
    llama.generate_stream(
        input_ids,
        128,
        <T as F32>::from_f32(0.9),
        4,
        <T as F32>::from_f32(1.0),
        session,
        tokenizer
    )
}

async fn get_all_sessions<T:SuperTrait>(data: web::Data<Arc<Mutex<Vec<Arc<Mutex<MySession<T>>>>>>>)-> impl Responder{
    let sessions = data.lock().unwrap();
    let session_data: Vec<MySessionData> = sessions.iter()
        .map(|session| {
            let session = session.lock().unwrap(); // 锁住每个 session
            MySessionData {
                id: session.id.clone(),
                title: session.title.clone(),
                created_at: session.created_at.clone(),
                history: session.history.clone(),
            }
        })
        .collect();
    HttpResponse::Ok().json(session_data)
}

async fn create_session<T:SuperTrait>(data: web::Data<Arc<Mutex<Vec<Arc<Mutex<MySession<T>>>>>>>)-> impl Responder{
    {
        let mut sessions = data.lock().unwrap();
        sessions.push(Arc::new(Mutex::new(MySession::new())));
    }
    get_all_sessions(data).await
}

async fn chat_api<T>(data: web::Data<Arc<Mutex<Vec<Arc<Mutex<MySession<T>>>>>>>,
    query: web::Query<HashMap<String, String>>)->HttpResponse
where T: SuperTrait
{   
    let session_id=query.get("id").unwrap();
    let content=query.get("content").unwrap();
    let chat_sessions=data.lock().unwrap();
    match chat_sessions.iter().find(|&x|x.lock().unwrap().id==*session_id){
        Some(item)=>{
                {item.lock().unwrap().history.push(Message::new(Role::User, content.clone()));};
                chat_start_for_api(Arc::clone(item), content.clone())
        },
        None=>HttpResponse::BadRequest().json("Invalid Session Id")
    }
}

async fn run_serve<T:SuperTrait>()->std::io::Result<()>{
    // let data=Arc::new(Mutex::new(Status::<T>::new()));
    // let chat_sessions=Arc::new(Mutex::new(Vec::<MySession>::new()));
    let data=Arc::new(Mutex::new(Vec::<Arc<Mutex<MySession<T>>>>::new()));
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
            // .app_data(web::Data::new(chat_sessions.clone()))
            .route("/chat", web::get().to(chat_api::<T>))
            .route("/getAllSessions", web::get().to(get_all_sessions::<T>))
            .route("/createSession", web::get().to(create_session::<T>))
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