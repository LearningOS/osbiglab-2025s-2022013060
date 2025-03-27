# Week6 Report
## 本周进展
部署llama.cpp, 并读llama.cpp代码
1. llama_model_loader.cpp
   1. 下面列出struct中与内存管理优化有关的变量, 比较重要的是files和mappings两种读入模式
   ```cpp
    struct llama_model_loader {
        int n_tensors = 0; // tensor的数量
        bool use_mmap = false; // 用不用mmap
        llama_files files; // 文件
        llama_mmaps mappings; // mmap
        std::map<std::string, struct llama_tensor_weight, weight_name_comparer> weights_map; // 张量的权重
        gguf_context_ptr meta; // 上下文指针
        std::vector<ggml_context_ptr> contexts; // 上下文表
        std::vector<std::pair<size_t, size_t>> mmaps_used;
    }
   ```
   其中mappings和files的区别在于是否产生映射, mmap的速度更快

   2. 这个过程中也会伴随mlock
2. llama_model_mmap.cpp
   1. llama_files: 实现逻辑比较简单, 就是简单的`ggml_fopen`和`fread/fwrite`, 从file中加载出数据
   2. llama_mmap: 逻辑也很简单, `mmap`到一个地址并存到`mapped_fragments`里
   3. llama_mlock: 强行保证处于RAM中. 使用逻辑, 需要锁定时调用`grow_to`方法, 后者检查是否还有`lock`的余量, 并进行pagesize的对齐后产生一个lock
3. llama_model.cpp 中相关的部分
   ```cpp
    struct llama_model{
        llama_mmaps mappings;

        // objects representing data potentially being locked in memory
        llama_mlocks mlock_bufs;
        llama_mlocks mlock_mmaps;

        // contexts where the model tensors metadata is stored
        std::vector<ggml_context_ptr> ctxs;

        // the model memory buffers for the tensor data
        std::vector<ggml_backend_buffer_ptr> bufs;

        buft_list_t cpu_buft_list;
        std::map<ggml_backend_dev_t, buft_list_t> gpu_buft_list;
    }
   ```
   具体实现时, `load_tensors`会在加载buf和tensor的时候尝试lock住对应的内存, 并加入到对应的vector中
4. 目前思路/下周计划
   `llama.cpp`中的lock方法现在是静态的, 计划做以下两个方法比较效率
   1. 通过统计来确定哪些tensor被lock
   2. 通过一种类似cache的做法来确定lock
   然后FFN层计算时即时抛掉小于一定阈值的项, 这样就不用每次全部fetch一遍了
