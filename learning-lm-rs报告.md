# learning-lm-rs报告

[项目地址]: https://github.com/Simon25772/learning-lm-rs

一，项目概述
本项目在基础语言模型功能之上实现两大核心扩展：     
1.混合精度推理系统 - 支持FP32/FP16/BF16计算模式     
2.网络服务化接口 - 提供网络API服务      
开发环境：WSL2 (Ubuntu 22.04 LTS)      
技术栈：Rust 1.83，Actix-Web 4.0      

二，核心功能实现

1.混合精度推理：

技术方案：                 
a.采用泛型编程范式抽象计算逻辑     
b.集成half库实现FP16/BF16数据类型支持

2.网络服务API:

技术方案：               
a.基于Actix-Web框架构建异步REST API    
b.使用session技术识别用户    
c.在APP层面为不同用户缓存KVCache    

三，功能演示

1.长文本生成能力

![img](story_show.png)

2.对话交互系统

![img](chat_show.png)

3.API服务接口

![img](webapi_show.png)

四，当前局限与改进方向

1.精度支持扩展，目标添加TF32（TensorFloat-32）支持

2.对话交互系统网络API添加流式相应支持

3.功能继续扩展     
a.多会话管理以及历史会话回滚；    
b.多线程分布式推理优化，附加性能对比；     
c.适配一种加速软件栈后端，Nvidia、AMD、国产芯片或 OpenCL、九源后端均可；     
d.有其他想法可和导师讨论是否可以作为拓展；     




