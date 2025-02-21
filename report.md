# learning-lm-rs报告

[仓库]: https://github.com/Simon25772/learning-lm-rs

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

![img](file:///C:\Users\Simon\Documents\Tencent Files\447604201\nt_qq\nt_data\Pic\2025-02\Ori\a705747f7fe0cabdf5ff23aa6a11bc4a.png)

[^]: 

2.对话交互系统

![img](file:///C:\Users\Simon\Documents\Tencent Files\447604201\nt_qq\nt_data\Pic\2025-02\Ori\8d46eede54e9e5cae683f397cebdfe69.png)

3.API服务接口

![img](file:///C:\Users\Simon\Documents\Tencent Files\447604201\nt_qq\nt_data\Pic\2025-02\Ori\91fd7bda6a39f89065cc85e48f590046.png)

4.多线程推理

将top_k设置成1后，使用chat模型，在用户输入为hi的条件下，单线程推理用时为231秒，多线程推理用时为167秒，提速37%。

![image-20250218184246579](C:\Users\Simon\AppData\Roaming\Typora\typora-user-images\image-20250218184246579.png)

![image-20250218184057034](C:\Users\Simon\AppData\Roaming\Typora\typora-user-images\image-20250218184057034.png)

四，有待提高

1.混合精度不支持TF32

2.chat模型的网络服务API等待时间太长

3.功能还可以继续扩展





