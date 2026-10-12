import torch
import transformers
from cache import KVCache, NaiveKVCache, PagedKVCache, BlockManager
import sampler
from scheduler import RequestStatus, Scheduler, Request
from kv_transfer import KVTransferBase, SyncKVTransfer, NoOpKVTransfer, AsyncKVTransfer
import time

class Engine:
    """推理引擎：协调模型、调度器、Block 管理、Prefix Cache 与 PD 分离传输。

    对外只暴露 submit / step / mark_finished / is_done 四个方法；step 内部负责
    一次完整的 prefill + decode 调度并回收已完成请求的资源。
    """
    def __init__(self, 
                 model, 
                 tokenizer, 
                 block_manager: BlockManager, 
                 cache_cls : type[PagedKVCache], 
                 cache_kwargs: dict|None = None, device='cuda',
                 use_cuda_graph=True,   # 是否启用 CUDA Graph，启用后要求 batch size 固定为 max_batch_size
                 max_batch_size=8,      # ← CUDA Graph 需要固定 batch size,
                 rank=0,
                 peer_ranks:list[int]=[0],
                 local_tp_size: int = 1,      # 本侧并行度
                 remote_tp_size: int = 1,     # 对侧并行度
                 is_primary: bool = True,     # 负责元数据meta传输的主要实例
                 role:str="pd",
                 enable_prefix_cache=True,
                 manual_seed=42,
                 detokenize_outputs: bool = True,
                 ignore_eos: bool = False,
                 max_num_seqs: int = 4,
                 ):
        device = torch.device(device)
        if device.type != "cuda":
            use_cuda_graph = False
        model_ops = getattr(model, "ops", None)
        if model_ops is not None:
            model_ops_device_type = getattr(model_ops, "device_type", None)
            if model_ops_device_type is not None and model_ops_device_type != device.type:
                raise ValueError(
                    f"Model ops backend '{model_ops.name}' targets {model_ops_device_type}, "
                    f"but Engine device is {device.type}. Build ModelConfig with a matching "
                    "use_cuda/ops_backend setting."
                )
            if not model_ops.supports_cuda_graph:
                use_cuda_graph = False

        self.model = model.to(device)
        # self.sampler = sampler
        self.tokenizer = tokenizer
        self.device = device
        self.kv_cache_cls = cache_cls
        # KV cache 的配置参数，后续可以改成动态的
        self.kv_cache_kwargs = cache_kwargs if cache_kwargs is not None else dict(
            block_manager=block_manager,
            num_layers=model.cfg.num_hidden_layers,
            max_seq_len=4096,
            num_kv_heads=model.cfg.local_num_key_value_heads,
            head_dim=model.cfg.head_dim,
            device=device,
            dtype=next(model.parameters()).dtype,
        )
        self.block_manager = block_manager
        self.eos_token_id = tokenizer.eos_token_id
        self.detokenize_outputs = detokenize_outputs
        self.ignore_eos = ignore_eos

        ### tp 并行相关设置 ###
        self.local_tp_size = local_tp_size
        self.rank = rank
        if local_tp_size > 1:
            torch.manual_seed(manual_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(manual_seed)
        
        ### PD 分离相关设置 ###
        self.role = role
        self.peer_ranks = peer_ranks
        #peer_rank是需要自身rank向其传递KV cache的相应rank
        if self.role == "p":
            self.transfer = AsyncKVTransfer(local_rank=rank, peer_ranks=peer_ranks, device=device,
                                            local_tp_size=local_tp_size,remote_tp_size=remote_tp_size,is_primary=is_primary,
                                            block_manager=block_manager, model_cfg=model.cfg,cache_kwargs=self.kv_cache_kwargs,role=role,)
        elif self.role == "d":
            self.transfer = AsyncKVTransfer(local_rank=rank, peer_ranks=peer_ranks, device=device,
                                            local_tp_size=local_tp_size,remote_tp_size=remote_tp_size,is_primary=is_primary,
                                            block_manager=block_manager, model_cfg=model.cfg,cache_kwargs=self.kv_cache_kwargs,role=role,)
        else:
            # self.transfer = None  # role="pd" 不需要传输层
            self.transfer = NoOpKVTransfer()  # 类型安全，poll() 和 try_recv 都是 no-op

        self.scheduler = Scheduler(kv_cache_cls=cache_cls, kv_cache_kwargs=self.kv_cache_kwargs,
                                   max_num_seqs=max_num_seqs)
        self.no_more_requests = False # 不会再有更多请求进入
        self._done_sent = False # 已经全部发送完

        self.use_cuda_graph = use_cuda_graph
        self.max_batch_size = max_batch_size
        
        # 创建 PrefixCache（所有 role 都可以用，PD 分离下 P 和 D 各有自己的）
        if enable_prefix_cache:
            from radix_tree import KVCacheRadixTree
            from prefix_cache import PrefixCache
            self.radix_tree = KVCacheRadixTree(block_manager.block_size)
            self.prefix_cache = PrefixCache(self.radix_tree, block_manager)
            block_manager.set_evict_callback(
                lambda n: len(self.prefix_cache.try_evict(n)) # type: ignore
            )
        else:
            self.prefix_cache = None

        # 需要 plan 的 attention 后端（如 FlashInfer）：每个 forward 前由 engine 显式 plan 一次，
        # 28 层共享；Triton/Torch 后端 needs_attention_plan=False，这里全部跳过，行为不变。
        self.attn_ops = model_ops if getattr(model_ops, "needs_attention_plan", False) else None
        if self.attn_ops is not None:
            self._setup_attention_backend()

        # CUDA Graph 只在有 decode 职责时 build
        if self.use_cuda_graph and role in ("d", "pd"):
            self._build_cuda_graph()

        self.model.eval()
        
    def _setup_attention_backend(self):
        """为需要 plan 的 attention 后端做一次性初始化。

        - 绑定本 rank 的 attention 形状（TP 下用 local head 数）
        - CUDA Graph 模式占用一个 dummy block：ghost 请求指向它，attention 只读它，
          ghost 行的输出直接丢弃。FlashInfer 由 page 数推 kv_len，ghost 不能是 0 页，
          所以需要一个合法 page。它不属于任何请求、不进 radix tree，不会被分配给别人或驱逐。
          fused RoPE 对 context_lens=0 的行不写 KV；eager-only 不预留 dummy block。
        """
        from blockmanager import pagedblocktype

        dummy_page = 0
        self._attn_dummy_logical_block = None
        if self.use_cuda_graph and self.role in ("d", "pd"):
            dummy_logical = self.block_manager.allocate(1)[0]
            block_type, dummy_page = self.block_manager.block_mapping[dummy_logical]
            if block_type != pagedblocktype.GPU:
                self.block_manager.dec_ref([dummy_logical])
                raise RuntimeError("could not reserve a GPU block for attention padding")
            self._attn_dummy_logical_block = dummy_logical

        cfg = self.model.cfg
        self.attn_ops.setup_attention(
            num_qo_heads=cfg.local_num_attention_heads,
            num_kv_heads=cfg.local_num_key_value_heads,
            head_dim=cfg.head_dim,
            page_size=self.block_manager.block_size,
            dtype=self.block_manager.dtype,
            device=self.device,
            dummy_page=dummy_page,
        )

    def _build_cuda_graph(self):
        """预分配静态 buffer 并 capture graph，只做一次"""
        if self.device.type != "cuda":
            raise RuntimeError("CUDA Graph requires a CUDA device.")

        # 静态 buffer
        self.static_input_ids    = torch.zeros(self.max_batch_size, 1, dtype=torch.long, device=self.device)
        self.static_slot_mapping = torch.full((self.max_batch_size,), -1, dtype=torch.int32, device=self.device)
        self.static_position_ids = torch.zeros(self.max_batch_size, 1, dtype=torch.long, device=self.device)
        self.static_block_table  = torch.full(
            (self.max_batch_size, self.model.cfg.MAX_BLOCKS_PER_SEQ), -1,
            dtype=torch.int32, device=self.device
        )
        self.static_context_lens = torch.zeros(self.max_batch_size, dtype=torch.int32, device=self.device)

        if self.attn_ops is not None:
            # graph 模式下 attention 元数据住在固定地址的 buffer 里：每次 replay 前在 graph
            # 外 plan（写这些 buffer），graph 只 capture kernel launch。
            # indices buffer 上界：每个请求最多 cdiv(max_seq_len, block_size) 页
            #（prefix 共享的页会在多个请求里重复出现，所以不能用总 block 数做上界）。
            max_pages_per_seq = -(-self.kv_cache_kwargs["max_seq_len"] // self.block_manager.block_size)
            self.attn_ops.init_cuda_graph(
                max_batch_size=self.max_batch_size,
                max_num_pages=self.max_batch_size * max_pages_per_seq,
            )
            # warmup / capture 用全 ghost 的 batch（与静态 buffer 的初始状态一致）
            self.attn_ops.plan_decode(
                [], [], use_cuda_graph=True,
                block_table=self.static_block_table,
                context_lens=self.static_context_lens,
            )

        # 预热（触发后端 autotune/编译）
        for _ in range(3):
            with torch.no_grad():
                _ = self.model.forward_decode(
                    self.static_input_ids,
                    kv_cache_k=self.block_manager.gpu_kv_cache[0],
                    kv_cache_v=self.block_manager.gpu_kv_cache[1],
                    position_ids=self.static_position_ids,
                    slot_mapping=self.static_slot_mapping,
                    block_table=self.static_block_table,
                    context_lens=self.static_context_lens,
                )
        torch.cuda.synchronize()
        
        # capture
        self.cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph):
            self.static_output = self.model.forward_decode(
                self.static_input_ids,
                kv_cache_k=self.block_manager.gpu_kv_cache[0],
                kv_cache_v=self.block_manager.gpu_kv_cache[1],
                position_ids=self.static_position_ids,
                slot_mapping=self.static_slot_mapping,
                block_table=self.static_block_table,
                context_lens=self.static_context_lens,
            )
        
        print(f"CUDA Graph captured (batch_size={self.max_batch_size})")
    
    def _decode_step_graph(self, decoding, kv_caches) -> torch.Tensor:
        """CUDA Graph replay 路径"""
        B = len(decoding)

        # =================================================================
        # 1. CPU侧
        # =================================================================
        input_ids_cpu = []
        slot_mapping_cpu = []
        position_ids_cpu = []
        context_lens_cpu = []

        for request, c in zip(decoding, kv_caches):
            c.prepare_decode_step()
            input_ids_cpu.append([request.generated_ids[-1]])
            slot_mapping_cpu.append(c.get_decode_slot())
            position_ids_cpu.append([c._seq_len])
            context_lens_cpu.append(c._seq_len + 1)

        # 补齐 Padding (幽灵请求)
        if B < self.max_batch_size:
            pad_len = self.max_batch_size - B
            input_ids_cpu.extend([[0]] * pad_len)
            slot_mapping_cpu.extend([-1] * pad_len)
            position_ids_cpu.extend([[0]] * pad_len)
            context_lens_cpu.extend([0] * pad_len)

        # =================================================================
        # 2. DMA 异步拷贝
        # =================================================================
        self.static_input_ids.copy_(torch.tensor(input_ids_cpu, dtype=torch.long), non_blocking=True)
        self.static_slot_mapping.copy_(torch.tensor(slot_mapping_cpu, dtype=torch.int32), non_blocking=True)
        self.static_position_ids.copy_(torch.tensor(position_ids_cpu, dtype=torch.long), non_blocking=True)
        self.static_context_lens.copy_(torch.tensor(context_lens_cpu, dtype=torch.int32), non_blocking=True)

        # =================================================================
        # 3. Block Table
        # =================================================================
        self.static_block_table.fill_(-1)

        for i, c in enumerate(kv_caches):
            bt = c.get_block_table()
            self.static_block_table[i, :bt.shape[0]].copy_(bt, non_blocking=True)

        # 3.5 需要 plan 的 attention 后端：在 graph 外 plan，写进 capture 时绑定的固定 buffer。
        # 页表直接用各请求 GPU 上的 block table 视图（不经过 CPU），长度用 CPU 侧已知值，无同步。
        # context_lens_cpu[:B] 已经包含本步要写入的新 token（+1）；ghost 由后端补齐到 max_batch_size。
        if self.attn_ops is not None:
            self.attn_ops.plan_decode(
                [c.get_block_table() for c in kv_caches],
                context_lens_cpu[:B],
                use_cuda_graph=True,
                block_table=self.static_block_table,
                context_lens=self.static_context_lens,
            )

        # =================================================================
        # 4. CUDA Graph重放
        # =================================================================
        self.cuda_graph.replay()

        for c in kv_caches:
            c._seq_len += 1

        return self.static_output[:B, -1, :]  # (B, vocab_size)
    
    def _decode_step_eager(self, decoding, kv_caches, B) -> torch.Tensor:
        """普通 eager 路径，用于 B > max_batch_size 或 use_cuda_graph=False"""
        for c in kv_caches:
            c.prepare_decode_step()

        slots = [c.get_decode_slot() for c in kv_caches]
        slot_mapping = torch.tensor(slots, dtype=torch.int32, device=self.device)

        input_ids = torch.tensor(
            [[r.generated_ids[-1]] for r in decoding],
            dtype=torch.long, device=self.device
        )
        position_ids = torch.tensor(
            [[c.seq_len] for c in kv_caches],
            dtype=torch.long, device=self.device
        )
        block_table = torch.full(
            (B, self.model.cfg.MAX_BLOCKS_PER_SEQ), -1,
            dtype=torch.int32, device=self.device
        )
        for i, c in enumerate(kv_caches):
            bt = c.get_block_table()
            block_table[i, :len(bt)] = bt

        seq_lens = [c.seq_len + 1 for c in kv_caches]
        context_lens = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)

        if self.attn_ops is not None:
            self.attn_ops.plan_decode(
                [c.get_block_table() for c in kv_caches],
                seq_lens,
                use_cuda_graph=False,
                block_table=block_table,
                context_lens=context_lens,
            )

        with torch.no_grad():
            logits = self.model(
                input_ids,
                kv_cache_k=self.block_manager.gpu_kv_cache[0],
                kv_cache_v=self.block_manager.gpu_kv_cache[1],
                position_ids=position_ids,
                slot_mapping=slot_mapping,
                is_prefill=False,
                block_table=block_table,
                context_lens=context_lens,
            )

        for c in kv_caches:
            c._seq_len += 1

        return logits[:, -1, :]  # (B, vocab_size)

    ###########################################
    # Batch 版本的接口，包括submit(),step(),is_done()
    ###########################################

    def submit(self, 
            prompt: str, 
            max_new_tokens: int, 
            temperature: float, 
            top_p: float) -> int:
            """提交生成请求，返回 request_id"""
            input_ids = self.tokenizer.encode(prompt)
            return self.submit_token_ids(input_ids, max_new_tokens, temperature, top_p)

    def submit_token_ids(
            self,
            input_ids: list[int],
            max_new_tokens: int,
            temperature: float,
            top_p: float) -> int:
            """提交已经 tokenized 的生成请求，避免 engine benchmark 计入 tokenizer。"""
            # 如果已经完成，返回-1
            if self.no_more_requests:
                return -1

            request = self.scheduler.create_request(list(input_ids), max_new_tokens, temperature, top_p,
                                                     self.kv_cache_cls, self.kv_cache_kwargs)

            # 在submit的时候，进行一次 prefix match 过程，insert 操作则在 Prefill 结束之后对整个 Prefill 输入序列进行
            if self.prefix_cache is not None:
                block_size = self.block_manager.block_size
                # 限制最大匹配长度：至少保留 1 个 token 做 prefill
                # 对齐到 block_size
                max_matchable = ((len(input_ids) - 1) // block_size) * block_size

                matched_blocks, matched_len, last_node = self.prefix_cache.match(
                    input_ids[:max_matchable]
                )

                request.matched_blocks = matched_blocks
                request.matched_len = matched_len
                request.last_node = last_node
            
            self.scheduler.add_request(request)
            return request.request_id

    def submit_ids(self, input_ids: list[int], max_new_tokens: int,
                   temperature: float, top_p: float) -> int:
        """Compatibility alias for the upstream pre-tokenized submission API."""
        return self.submit_token_ids(input_ids, max_new_tokens, temperature, top_p)

    def _completed_text(self, request: Request) -> str:
        if not self.detokenize_outputs:
            return ""
        return self.tokenizer.decode(request.input_ids + request.generated_ids)
    
    def mark_finished(self):
        """通知 Engine 不会再有新请求提交"""
        self.no_more_requests = True
    
    def step(self) -> list[tuple[int, str]]:
        # t0 = time.perf_counter()
        self.transfer.poll()  # 如果不启用则是 NoOpKVTransfer，对应操作是 no-op，不影响 role="pd"

        # === D 侧：检查是否有新到达请求，在 schedule 之前 ===
        if self.role == "d" and not self.transfer.recv_done:
            while True:
                new_request = self.transfer.try_recv_request()
                if new_request is None:
                    break
                self.scheduler.add_request(new_request, RequestStatus.DECODING)

        prefilling, decoding = self.scheduler.schedule()
        completed_requests = []
        # t1 = time.perf_counter()

        # === P 侧：逐个请求处理，动态形状，不使用 cuda graph ===
        for request in prefilling:
            # t_match = time.perf_counter()
            matched_blocks = request.matched_blocks
            matched_len = request.matched_len

            # 计算需要新 prefill 的 token
            new_tokens = request.input_ids[matched_len:]
            new_len = len(new_tokens)
            total_len = matched_len + new_len   # = len(request.input_ids)

            kv_cache = request.kv_cache

            # 挂载 matched block
            if matched_blocks:
                kv_cache.adopt_blocks(matched_blocks, matched_len)
            # adopt 后 kv_cache._seq_len = matched_len

            # t_alloc = time.perf_counter()
            # 为新 token 分配 block（在 matched blocks 之后）
            kv_cache._allocate_for_prefill(new_len)

            # slot_mapping 从 matched_len 开始
            slot_mapping = kv_cache.get_prefill_slot_mapping(new_len)
            # get_prefill_slot_mapping 内部从 _seq_len 开始算，正好是 matched_len

            # position_ids 从 matched_len 开始
            position_ids = torch.arange(
                matched_len, total_len,
                dtype=torch.long, device=self.device
            ).unsqueeze(0)

            # 把输入转化为 Tensor（直接构在 device 上，省一次中间 CPU 张量）
            input_ids_tensor = torch.tensor(new_tokens, dtype=torch.long, device=self.device).unsqueeze(0)

            bt = kv_cache.get_block_table()
            block_table = torch.full(
                (1, self.model.cfg.MAX_BLOCKS_PER_SEQ), -1,
                dtype=torch.int32, device=self.device
            )
            block_table[0, :bt.shape[0]] = bt

            context_lens = torch.tensor([total_len], dtype=torch.int32, device=self.device)
            new_token_lens = torch.tensor([new_len], dtype=torch.int32, device=self.device)
            q_start_loc = torch.tensor([0], dtype=torch.int32, device=self.device)

            if self.attn_ops is not None:
                # Q = 本次新算的 new_len 个 token，KV = prefix(命中) + 新 token；
                # 后端用 bottom-right causal，对齐方式与 Triton prefill kernel 相同。
                self.attn_ops.plan_prefill(
                    [bt], [total_len], [new_len],
                    block_table=block_table,
                    context_lens=context_lens,
                    new_token_lens=new_token_lens,
                    q_start_loc=q_start_loc,
                )

            # t_gpu = time.perf_counter()
            with torch.no_grad():
                logits = self.model(
                    input_ids_tensor,
                    kv_cache_k=self.block_manager.gpu_kv_cache[0],
                    kv_cache_v=self.block_manager.gpu_kv_cache[1],
                    position_ids=position_ids,
                    slot_mapping=slot_mapping,
                    is_prefill=True,
                    block_table=block_table,
                    context_lens=context_lens,
                    new_token_lens=new_token_lens,
                    q_start_loc=q_start_loc,
                    max_new_len=new_len,  # B=1 prefill：直接传 Python int，跳过 wrapper 内每层一次的 .item() 同步
                )

            # t_sample = time.perf_counter()
            # 更新 seq_len 到完整长度
            kv_cache._seq_len = total_len

            # prefill 完成后，插入 prefix cache
            # 这里的设计逻辑目前是只有 Prefill 阶段和实例会和 prefix cache 相关，Decode 是无关的
            if self.prefix_cache is not None:
                # 只插入对齐到 block_size 的部分（完全 block 化版本）
                full_blocks_count = total_len // self.block_manager.block_size
                if full_blocks_count > 0:
                    aligned_tokens = request.input_ids[:full_blocks_count * self.block_manager.block_size]
                    aligned_logical = kv_cache.logical_block_ids[:full_blocks_count]
                    self.prefix_cache.insert(aligned_tokens, aligned_logical)

            # 采样首 token
            next_token_id = sampler.sample(logits[:, -1, :], request.temperature, request.top_p)
            request.generated_ids.append(int(next_token_id.item()))

            if self.role == "p":
                self.transfer.send_request(request)
                request.has_finished_notification = True
            elif ((not self.ignore_eos and next_token_id.item() == self.eos_token_id)
                  or request.is_max_len_finished()):
                request.has_finished_notification = True
                completed_requests.append((
                    request.request_id,
                    self._completed_text(request)
                ))
            # t_done = time.perf_counter()
            # print(f"[prefill] schedule={t1-t0:.5f} "
            #     f"match={t_alloc-t_match:.5f} "
            #     f"alloc+tensor={t_gpu-t_alloc:.5f} "
            #     f"gpu={t_sample-t_gpu:.5f} "
            #     f"sample+insert={t_done-t_sample:.5f}")

        # === P 侧：所有 prefill 做完且用户标记 no_more_requests，则发送终止信号 ===
        if (self.role == "p"
            and self.no_more_requests
            and len(self.scheduler.waiting) == 0
            and len(self.scheduler.prefilling) == 0
            and not self._done_sent):
            self.transfer.send_done()
            self._done_sent = True

        # === D 侧：batch，根据设置走 CUDA Graph 或 eager ===
        if decoding and self.role in ("d", "pd"):
            B = len(decoding)
            kv_caches = [r.kv_cache for r in decoding]

            if self.use_cuda_graph and B <= self.max_batch_size:
                logits_batch = self._decode_step_graph(decoding, kv_caches)
            else:
                logits_batch = self._decode_step_eager(decoding, kv_caches, B)

            temps = [r.temperature for r in decoding]
            top_ps = [r.top_p for r in decoding]
            token_ids = sampler.sample_batch(logits_batch, temps, top_ps)
            for i, request in enumerate(decoding):
                request.generated_ids.append(token_ids[i])
                if ((not self.ignore_eos and token_ids[i] == self.eos_token_id)
                    or request.is_max_len_finished()):
                    request.has_finished_notification = True
                    completed_requests.append((
                        request.request_id,
                        self._completed_text(request)
                    ))

        # === 统一释放 FINISHED 请求的资源 ===
        # 暂时不再使用 Scheduler 的 clear_finished()
        for request in self.scheduler.finished:
            if request.request_status == RequestStatus.FINISHED:
                self._close_request(request)

        return completed_requests
    
    def _close_request(self, request: Request):
        """
        释放请求持有的所有资源，把状态设为 CLOSED。
        对所有 role（p/d/pd）和所有场景（正常结束/prefill 失败/被取消）都一样。
        """
        if request.request_status == RequestStatus.CLOSED:
            return  # 保证幂等性质

        kv_cache = request.kv_cache

        if (self.role == "pd" or self.role == "p") and self.prefix_cache is not None and kv_cache.seq_len > 0:
            # 1. Radix 层：根据 node 记录，dec match 时 inc 过的路径
            # 这个策略并不唯一，可以不 dec，同时 evict 做相应调整，允许 evict 计数不为0的node
            if request.last_node is not self.radix_tree.root:
                self.radix_tree.dec_lock_ref(request.last_node)

            # 2. Block 层：dec 请求持有的所有 block
            held_blocks = list(kv_cache.logical_block_ids)
            if held_blocks:
                self.block_manager.dec_ref(held_blocks)

            # 3. 清理 kv_cache 状态
            kv_cache._seq_len = 0
            kv_cache.allocated_cache_block_num = 0
            kv_cache.physical_block_ids = []
            kv_cache.logical_block_ids = []
            kv_cache.gpu_block_table.fill_(-1)
        else:
            # 没有 prefix cache 或请求还没 prefill 过，则走原始 reset
            kv_cache.reset()

        request.request_status = RequestStatus.CLOSED

    def is_done(self) -> bool:
        if self.role == "p":
            return self._done_sent and len(self.transfer.pending_sends) == 0
        elif self.role == "d":
            return (self.transfer.recv_done
                    and len(self.scheduler.decoding) == 0)
        else:
            return (self.no_more_requests
                    and len(self.scheduler.waiting) == 0
                    and len(self.scheduler.prefilling) == 0
                    and len(self.scheduler.decoding) == 0)
