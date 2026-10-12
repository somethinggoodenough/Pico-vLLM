"""CPU regressions for attention setup and repeated benchmark workloads."""

import random
from types import SimpleNamespace

import pytest
import torch

from benchmarks.bench_decode_e2e import StubTokenizer, run_workload
from blockmanager import BlockManager
from cache import PagedKVCache
from engine import Engine
from model import ModelConfig, Qwen25_15B

pytestmark = [pytest.mark.cpu, pytest.mark.single_card]


def small_block_manager(num_blocks=4):
    return BlockManager(num_gpu_blocks=num_blocks, num_cpu_blocks=0, block_size=16,
                        num_layers=1, num_kv_heads=1, head_dim=16,
                        dtype=torch.float32, device="cpu")


@pytest.mark.parametrize("graph,role,reserves", [(False, "pd", False), (True, "p", False),
                                                (True, "pd", True), (True, "d", True)])
def test_dummy_page_is_reserved_only_for_graph_decode(graph, role, reserves):
    bm = small_block_manager()
    bm.allocate(1)  # page zero already belongs to another owner
    calls = []
    cfg = SimpleNamespace(local_num_attention_heads=2, local_num_key_value_heads=1, head_dim=16)
    engine = object.__new__(Engine)
    engine.block_manager, engine.model = bm, SimpleNamespace(cfg=cfg)
    engine.device, engine.use_cuda_graph, engine.role = torch.device("cpu"), graph, role
    engine.attn_ops = SimpleNamespace(setup_attention=lambda **kw: calls.append(kw))
    engine._setup_attention_backend()
    assert bm.num_free_blocks == (2 if reserves else 3)
    assert calls[0]["dummy_page"] == (1 if reserves else 0)
    assert (engine._attn_dummy_logical_block is not None) == reserves


def test_repeated_workloads_release_kv_before_next_prefill(monkeypatch):
    # The benchmark timer uses CUDA sync; the real CPU Engine lets us test
    # scheduler/block lifetimes without needing CUDA or FlashInfer.
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    cfg = ModelConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, intermediate_size=64,
                      max_position_embeddings=64, use_cuda=False, ops_backend="torch")
    bm = small_block_manager()
    engine = Engine(Qwen25_15B(cfg), StubTokenizer(), bm, PagedKVCache,
                    device="cpu", use_cuda_graph=False, max_num_seqs=2,
                    enable_prefix_cache=False, ignore_eos=True)
    for _ in range(3):
        prefill, decodes = run_workload(engine, 2, 17, 4, 64, random.Random(0))
        assert prefill >= 0 and len(decodes) == 3
        assert bm.num_free_blocks == 4
        assert engine.scheduler.num_in_progress == 0
        assert not engine.scheduler.finished


def test_workload_rejects_missing_decode_phase():
    with pytest.raises(ValueError, match="gen_len must be >= 2"):
        run_workload(None, 1, 16, 1, 64, random.Random(0))


@pytest.mark.parametrize("method", ["submit", "submit_token_ids", "submit_ids"])
@pytest.mark.parametrize("detokenize", [True, False])
def test_upstream_submission_and_detokenize_apis(method, detokenize):
    class Tokenizer(StubTokenizer):
        decode_calls = 0

        def encode(self, text):
            return [1, 2, 3]

        def decode(self, ids):
            self.decode_calls += 1
            return "decoded"

    cfg = ModelConfig(vocab_size=64, hidden_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, intermediate_size=64,
                      max_position_embeddings=64, use_cuda=False, ops_backend="torch")
    tokenizer = Tokenizer()
    engine = Engine(Qwen25_15B(cfg), tokenizer, small_block_manager(), PagedKVCache,
                    device="cpu", use_cuda_graph=False, enable_prefix_cache=False,
                    max_num_seqs=2, ignore_eos=True, detokenize_outputs=detokenize)
    prompt = "prompt" if method == "submit" else [1, 2, 3]
    rid = getattr(engine, method)(prompt, max_new_tokens=1, temperature=0.0, top_p=1.0)
    assert engine.step() == [(rid, "decoded" if detokenize else "")]
    assert tokenizer.decode_calls == int(detokenize)
    engine.step()
    engine.scheduler.clear_finished()
    assert engine.block_manager.num_free_blocks == 4
