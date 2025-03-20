# Week5 Report
## 本周进展
1. 阅读PowerInfer论文
   1. High sparsity: hot/cold neurons, 其中neuron指矩阵的一行/列, 比较容易让ReLU不为0的称为hot(对于其他激活函数比如SwiGLU指的是影响较小)
   2. GPU-CPU Hybrid: low batch size下, 直接在CPU中计算要优于load到GPU中计算
   3. Workflow: offline统计出hot/cold分配到GPU/CPU中, online predictor(是一个MLP)预测哪些被activated, 取出计算
   4. Balance: online predictor的MLP占用GPU, 大小需要和模型大小做平衡
2. 阅读PowerInfer2论文
   1. High sparsity: neuron cluster, 不同于PowerInfer, 此处的neuron cluster是动态的, cluster的大小取决于I/O速度
   2. Offline Planner: 将neuron cluster分配到CPU, NPU和Flash中
   3. Prefill: NPU-Centric, 直接合成一个大cluster进行计算, CPU在这里起一个preload的作用来缓解NPU容量问题
   4. Decoding: CPU-NPU Hybrid, 类似PowerInfer, 但这个过程中hot/cold是动态的, 发生CPU和NPU之间的neuron交换
   5. Cache: KV-Cache, hot in NPU, cold in CPU, 策略LRU, NPU以cluster为单位, CPU以neuron为单位
   6. Pipeline: 每I/O一个cluster计算一个
3. 阅读LLMFlash论文
   1. Memory: Attention参数放在DRAM中, 预测得到activated neuron加载到DRAM中, 采用Sliding Window
   2. I/O: 全连接层同时activated的部分拼在一起存
   3. DRAM: 预先开出一片空间自己管理, 降低删除时间(swap到尾部再减指针)
4. 在本地clone了PowerInfer跑了一下

## 下周计划
1. 看一下KV Cache相关
2. 看llama.cpp和PowerInfer源码, 是怎么在其基础上进行修改的