"""
nanodurov.py — a telegram client that trains a language model on chat messages.

one file. telethon + notorch. connect to a group, watch bots and humans talk,
learn their patterns, generate text in their style. the chat is the corpus.
the model grows with the conversation.

dedicated to Pavel Durov, who built the platform where bots can't see each other
but we're training on them anyway.

no pytorch. no pip install torch. no 2.7 GB of your soul.
the engine is notorch — pure C, loaded via ctypes.

usage:
    pip install telethon
    python nanodurov.py                         # interactive telegram mode
    python nanodurov.py --generate "hello"      # generate from prompt
    python nanodurov.py --train-only chat.txt   # train on exported chat

env vars:
    TELEGRAM_API_ID      — from my.telegram.org
    TELEGRAM_API_HASH    — from my.telegram.org
"""

import os
import sys
import math
import time
import struct
import hashlib
import asyncio
import argparse
import random
import ctypes
from collections import defaultdict

# --- notorch (ctypes to libnotorch) -------------------------------------------
# the engine. no torch. no numpy. just C.

from ariannamethod.notorch_nn import (
    _lib, _get_tensor_struct, _NtTapeEntry, _NtTensor,
    Tensor, Parameter, Module, Linear, Embedding, RMSNorm,
    softmax, multinomial, seed as nt_seed,
)
from ariannamethod.chuck import ChuckOptimizer

NOTORCH_AVAILABLE = True

# --- optional telegram --------------------------------------------------------

try:
    from telethon import TelegramClient, events
    from telethon.tl.types import User, Channel, Chat
    TELETHON_AVAILABLE = True
except ImportError:
    TELETHON_AVAILABLE = False

# --- hyperparameters -----------------------------------------------------------
# the model grows with the corpus. more data → bigger model.
# like a tree. not like a corporation.

GROWTH_STAGES = [
    # (min_corpus_kb, dim, n_heads, n_layers, ctx_len, max_merges, name)
    (0,    32,  2, 1,  64,  128,  'seed'),
    (5,    48,  3, 2,  64,  256,  'sprout'),
    (20,   64,  4, 3, 128,  512,  'sapling'),
    (50,   96,  4, 4, 128,  768,  'tree'),
    (100, 128,  4, 6, 256, 1024,  'oak'),
    (250, 192,  6, 8, 256, 1536,  'forest'),
    (500, 256,  8, 10, 512, 2048, 'ancient'),
]

LR = 3e-4
TRAIN_STEPS_PER_ROUND = 50
AUTO_TRAIN_INTERVAL = 60  # seconds between auto-train rounds

def get_stage(corpus_bytes):
    """Pick the largest stage that fits the corpus."""
    kb = corpus_bytes / 1024
    stage = GROWTH_STAGES[0]
    for s in GROWTH_STAGES:
        if kb >= s[0]:
            stage = s
    return stage

# --- BPE tokenizer (KARL, from nanoagi) ----------------------------------------
# the tokenizer that eats your chat and asks for seconds.

class BPE:
    """Byte-pair encoding. Learns merges from text, encodes, decodes.
    Append-only — vocab grows, never shrinks. Like regret."""

    def __init__(self, max_merges=256):
        self.max_merges = max_merges
        self.merges = []  # [(a, b, new_id), ...]
        self.vocab_size = 256
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.seen_hashes = set()
        self.corpus = b""

    def _count_pairs(self, ids):
        counts = defaultdict(int)
        for i in range(len(ids) - 1):
            counts[(ids[i], ids[i + 1])] += 1
        return counts

    def _merge(self, ids, a, b, new_id):
        out = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def learn(self, data, num_merges=None):
        """Learn BPE merges from raw bytes."""
        if isinstance(data, str):
            data = data.encode('utf-8', errors='replace')
        num_merges = num_merges or min(self.max_merges, 256)
        ids = list(data)

        for m in range(num_merges):
            counts = self._count_pairs(ids)
            if not counts:
                break
            best = max(counts, key=counts.get)
            if counts[best] < 2:
                break
            new_id = 256 + len(self.merges)
            if new_id >= 256 + self.max_merges:
                break
            ids = self._merge(ids, best[0], best[1], new_id)
            self.merges.append((best[0], best[1], new_id))
            self.vocab[new_id] = self.vocab.get(best[0], b'?') + self.vocab.get(best[1], b'?')
            self.vocab_size = 256 + len(self.merges)

        print(f"[bpe] {len(self.merges)} merges, vocab={self.vocab_size}, tokens={len(ids)}")
        return ids

    def encode(self, text):
        if isinstance(text, str):
            text = text.encode('utf-8', errors='replace')
        ids = list(text)
        for a, b, new_id in self.merges:
            ids = self._merge(ids, a, b, new_id)
        return ids

    def decode(self, ids):
        raw = b''
        for i in ids:
            raw += self.vocab.get(i, b'?')
        return raw.decode('utf-8', errors='replace')

    def ingest(self, text):
        """Add text to corpus with dedup and quality filter."""
        if isinstance(text, str):
            raw = text
            text = text.encode('utf-8', errors='replace')
        else:
            raw = text.decode('utf-8', errors='replace')
        if len(text) < 15:
            return False
        h = hashlib.sha256(text).hexdigest()[:16]
        if h in self.seen_hashes:
            return False
        stripped = raw.strip()
        if stripped.startswith('http://') or stripped.startswith('https://'):
            if ' ' not in stripped:
                return False
        alpha = sum(1 for c in stripped if c.isalpha() or c.isspace())
        if len(stripped) > 0 and alpha / len(stripped) < 0.3:
            return False
        if len(stripped) > 5:
            most_common = max(set(stripped), key=stripped.count)
            if stripped.count(most_common) / len(stripped) > 0.5:
                return False
        self.seen_hashes.add(h)
        self.corpus += text
        return True

    def retokenize(self, max_new=64):
        """Grow vocab with new merges from accumulated corpus."""
        ids = list(self.corpus)
        for a, b, new_id in self.merges:
            ids = self._merge(ids, a, b, new_id)
        found = 0
        for _ in range(min(max_new, self.max_merges - len(self.merges))):
            counts = self._count_pairs(ids)
            if not counts:
                break
            best = max(counts, key=counts.get)
            if counts[best] < 3:
                break
            new_id = 256 + len(self.merges)
            ids = self._merge(ids, best[0], best[1], new_id)
            self.merges.append((best[0], best[1], new_id))
            self.vocab[new_id] = self.vocab.get(best[0], b'?') + self.vocab.get(best[1], b'?')
            found += 1
        self.vocab_size = 256 + len(self.merges)
        if found:
            print(f"[bpe] +{found} merges (vocab={self.vocab_size})")
        return ids

    def save(self, path):
        with open(path, 'wb') as f:
            f.write(b'BPE1')
            f.write(struct.pack('<I', len(self.merges)))
            for a, b, nid in self.merges:
                f.write(struct.pack('<III', a, b, nid))
            f.write(struct.pack('<I', len(self.corpus)))
            f.write(self.corpus)
        print(f"[bpe] saved to {path}")

    def load(self, path):
        if not os.path.exists(path):
            return False
        with open(path, 'rb') as f:
            if f.read(4) != b'BPE1':
                return False
            n = struct.unpack('<I', f.read(4))[0]
            self.merges = []
            for _ in range(n):
                a, b, nid = struct.unpack('<III', f.read(12))
                self.merges.append((a, b, nid))
                self.vocab[nid] = self.vocab.get(a, bytes([a % 256])) + self.vocab.get(b, bytes([b % 256]))
            self.vocab_size = 256 + len(self.merges)
            corpus_len = struct.unpack('<I', f.read(4))[0]
            self.corpus = f.read(corpus_len)
        print(f"[bpe] loaded: {len(self.merges)} merges, {len(self.corpus)} bytes corpus")
        return True

# --- transformer model (notorch) -----------------------------------------------
# RMSNorm, RoPE, SwiGLU, causal attention. backed by libnotorch via ctypes.
# every forward/backward runs through the C tape. no torch. no numpy.

class NanoDurovModel(Module):
    """LLaMA-style transformer backed by notorch."""

    def __init__(self, vocab_size, dim, n_heads, n_layers, ctx_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ctx_len = ctx_len
        self.head_dim = dim // n_heads
        hidden = dim * 4

        # Parameters — all stored as notorch tensors
        self.tok_emb = Embedding(vocab_size, dim)
        self.layers = []
        for l in range(n_layers):
            layer = {
                'rms1': RMSNorm(dim),
                'wq': Linear(dim, dim),
                'wk': Linear(dim, dim),
                'wv': Linear(dim, dim),
                'wo': Linear(dim, dim),
                'rms2': RMSNorm(dim),
                'w_gate': Linear(dim, hidden),
                'w_up': Linear(dim, hidden),
                'w_down': Linear(hidden, dim),
            }
            # register as modules
            for k, v in layer.items():
                setattr(self, f'l{l}_{k}', v)
            self.layers.append(layer)
        self.norm_f = RMSNorm(dim)
        self.head = Linear(dim, vocab_size)

    def param_list(self):
        """Ordered parameter list matching C training scripts."""
        params = [self.tok_emb.weight]
        for l in self.layers:
            params.extend([
                l['rms1'].weight, l['wq'].weight, l['wk'].weight,
                l['wv'].weight, l['wo'].weight, l['rms2'].weight,
                l['w_gate'].weight, l['w_up'].weight, l['w_down'].weight,
            ])
        params.extend([self.norm_f.weight, self.head.weight])
        return params

    def forward_train(self, token_ids, target_ids):
        """
        Full forward+backward+step through notorch tape.
        token_ids: list[int] length ctx_len
        target_ids: list[int] length ctx_len
        Returns loss float.
        """
        CTX = len(token_ids)
        DIM = self.dim
        HD = self.head_dim

        _lib.nt_tape_start()
        _lib.nt_train_mode(1)

        # Register params on tape
        params = self.param_list()
        tape_ids = []
        for p in params:
            idx = _lib.nt_tape_param(p._ptr)
            tape_ids.append(idx)
        # Mark embedding as no-decay
        _lib.nt_tape_no_decay(tape_ids[0])

        # Token/target tensors
        tok_t = Tensor.zeros(CTX)
        tgt_t = Tensor.zeros(CTX)
        tok_t.set_data([float(x) for x in token_ids])
        tgt_t.set_data([float(x) for x in target_ids])
        tok_idx = _lib.nt_tape_record(tok_t._ptr, 0, -1, -1, ctypes.c_float(0))
        tgt_idx = _lib.nt_tape_record(tgt_t._ptr, 0, -1, -1, ctypes.c_float(0))
        tok_t._owns = False
        tgt_t._owns = False

        # Forward
        pi = 0
        h = _lib.nt_seq_embedding(tape_ids[pi], -1, tok_idx, CTX, DIM); pi += 1

        for l in range(self.n_layers):
            rms1 = tape_ids[pi]; pi += 1
            wq = tape_ids[pi]; pi += 1
            wk = tape_ids[pi]; pi += 1
            wv = tape_ids[pi]; pi += 1
            wo = tape_ids[pi]; pi += 1
            rms2 = tape_ids[pi]; pi += 1
            wg = tape_ids[pi]; pi += 1
            wu = tape_ids[pi]; pi += 1
            wd = tape_ids[pi]; pi += 1

            xn = _lib.nt_seq_rmsnorm(h, rms1, CTX, DIM)
            q = _lib.nt_rope(_lib.nt_seq_linear(wq, xn, CTX), CTX, HD)
            k = _lib.nt_rope(_lib.nt_seq_linear(wk, xn, CTX), CTX, HD)
            v = _lib.nt_seq_linear(wv, xn, CTX)
            attn = _lib.nt_mh_causal_attention(q, k, v, CTX, HD)
            h = _lib.nt_add(h, _lib.nt_seq_linear(wo, attn, CTX))

            xn = _lib.nt_seq_rmsnorm(h, rms2, CTX, DIM)
            gate = _lib.nt_silu(_lib.nt_seq_linear(wg, xn, CTX))
            up = _lib.nt_seq_linear(wu, xn, CTX)
            h = _lib.nt_add(h, _lib.nt_seq_linear(wd, _lib.nt_mul(gate, up), CTX))

        rmsf = tape_ids[pi]; pi += 1
        head = tape_ids[pi]; pi += 1
        hf = _lib.nt_seq_rmsnorm(h, rmsf, CTX, DIM)
        logits_idx = _lib.nt_seq_linear(head, hf, CTX)
        loss_idx = _lib.nt_seq_cross_entropy(logits_idx, tgt_idx, CTX, self.vocab_size)

        # Read loss from tape
        tape_ptr = _lib.nt_tape_get()
        entry_size = ctypes.sizeof(_NtTapeEntry)
        tape_addr = ctypes.cast(tape_ptr, ctypes.c_void_p).value
        loss_entry = ctypes.cast(
            tape_addr + loss_idx * entry_size,
            ctypes.POINTER(_NtTapeEntry)
        ).contents
        loss_tensor = ctypes.cast(loss_entry.output, ctypes.POINTER(_NtTensor)).contents
        loss_val = loss_tensor.data[0]

        return loss_idx, loss_val

    def backward_step(self, loss_idx, loss_val, lr):
        """Backward + Chuck step + clear tape."""
        _lib.nt_tape_backward(loss_idx)
        _lib.nt_tape_clip_grads(ctypes.c_float(1.0))
        _lib.nt_tape_chuck_step(ctypes.c_float(lr), ctypes.c_float(loss_val))
        _lib.nt_tape_clear()

    def generate(self, token_ids, max_new=100, temperature=0.8, top_k=40):
        """Generate tokens using forward pass (no tape, inference only)."""
        _lib.nt_train_mode(0)
        ctx = list(token_ids)

        for _ in range(max_new):
            if len(ctx) > self.ctx_len:
                ctx = ctx[-self.ctx_len:]
            CTX = len(ctx)

            _lib.nt_tape_start()
            params = self.param_list()
            tape_ids = [_lib.nt_tape_param(p._ptr) for p in params]

            tok_t = Tensor.zeros(CTX)
            tgt_t = Tensor.zeros(CTX)
            tok_t.set_data([float(x) for x in ctx])
            tok_idx = _lib.nt_tape_record(tok_t._ptr, 0, -1, -1, ctypes.c_float(0))
            tgt_idx = _lib.nt_tape_record(tgt_t._ptr, 0, -1, -1, ctypes.c_float(0))
            tok_t._owns = False
            tgt_t._owns = False

            pi = 0
            h = _lib.nt_seq_embedding(tape_ids[pi], -1, tok_idx, CTX, self.dim); pi += 1
            for l in range(self.n_layers):
                rms1=tape_ids[pi]; pi+=1
                wq=tape_ids[pi]; pi+=1; wk=tape_ids[pi]; pi+=1
                wv=tape_ids[pi]; pi+=1; wo=tape_ids[pi]; pi+=1
                rms2=tape_ids[pi]; pi+=1
                wg=tape_ids[pi]; pi+=1; wu=tape_ids[pi]; pi+=1; wd=tape_ids[pi]; pi+=1
                xn = _lib.nt_seq_rmsnorm(h, rms1, CTX, self.dim)
                q = _lib.nt_rope(_lib.nt_seq_linear(wq, xn, CTX), CTX, self.head_dim)
                k = _lib.nt_rope(_lib.nt_seq_linear(wk, xn, CTX), CTX, self.head_dim)
                v = _lib.nt_seq_linear(wv, xn, CTX)
                attn = _lib.nt_mh_causal_attention(q, k, v, CTX, self.head_dim)
                h = _lib.nt_add(h, _lib.nt_seq_linear(wo, attn, CTX))
                xn = _lib.nt_seq_rmsnorm(h, rms2, CTX, self.dim)
                gate = _lib.nt_silu(_lib.nt_seq_linear(wg, xn, CTX))
                up = _lib.nt_seq_linear(wu, xn, CTX)
                h = _lib.nt_add(h, _lib.nt_seq_linear(wd, _lib.nt_mul(gate, up), CTX))

            rmsf=tape_ids[pi]; pi+=1; head_i=tape_ids[pi]; pi+=1
            hf = _lib.nt_seq_rmsnorm(h, rmsf, CTX, self.dim)
            logits_idx = _lib.nt_seq_linear(head_i, hf, CTX)

            # Get logits for last position
            tape_ptr = _lib.nt_tape_get()
            entry_size = ctypes.sizeof(_NtTapeEntry)
            tape_addr = ctypes.cast(tape_ptr, ctypes.c_void_p).value
            logits_entry = ctypes.cast(
                tape_addr + logits_idx * entry_size,
                ctypes.POINTER(_NtTapeEntry)
            ).contents
            logits_t = ctypes.cast(logits_entry.output, ctypes.POINTER(_NtTensor)).contents
            # Last position logits
            offset = (CTX - 1) * self.vocab_size
            raw_logits = [logits_t.data[offset + i] / temperature for i in range(self.vocab_size)]

            # Top-k
            if top_k > 0 and top_k < self.vocab_size:
                sorted_vals = sorted(raw_logits, reverse=True)
                threshold = sorted_vals[min(top_k - 1, len(sorted_vals) - 1)]
                raw_logits = [v if v >= threshold else -1e30 for v in raw_logits]

            probs = softmax(raw_logits)
            next_id = multinomial(probs)

            _lib.nt_tape_clear()

            ctx.append(next_id)
            if next_id == 10 and len(ctx) > len(token_ids) + 5:  # newline
                break

        return ctx[len(token_ids):]

    def save_weights(self, path):
        """Save weights using nt_save."""
        params = self.param_list()
        n = len(params)
        arr = (ctypes.c_void_p * n)(*[p._ptr for p in params])
        _lib.nt_save(path.encode(), arr, n)

    def load_weights(self, path):
        """Load weights using nt_load."""
        if not os.path.exists(path):
            return False
        n_loaded = ctypes.c_int(0)
        loaded = _lib.nt_load(path.encode(), ctypes.byref(n_loaded))
        if not loaded:
            return False
        params = self.param_list()
        for i in range(min(n_loaded.value, len(params))):
            src = _get_tensor_struct(loaded[i])
            dst = _get_tensor_struct(params[i]._ptr)
            if src.len == dst.len:
                ctypes.memmove(dst.data, src.data, dst.len * 4)
            _lib.nt_tensor_free(loaded[i])
        return True


# --- trainer -------------------------------------------------------------------
# the part where numbers go down and hope goes up.
# or numbers go up and you stare at the ceiling.

class Trainer:
    def __init__(self, bpe):
        self.bpe = bpe
        self.model = None
        self.stage_name = None
        self.total_steps = 0
        self.best_loss = float('inf')
        self._token_ids = None

    def _ensure_model(self):
        """Create or grow model based on corpus size."""
        stage = get_stage(len(self.bpe.corpus))
        _, dim, n_heads, n_layers, ctx_len, max_merges, name = stage

        if self.stage_name == name and self.model is not None:
            return True

        old_name = self.stage_name
        self.bpe.max_merges = max_merges
        vocab_size = 256 + max_merges

        self.model = NanoDurovModel(vocab_size, dim, n_heads, n_layers, ctx_len)
        self.stage_name = name

        n_params = sum(p.numel for p in self.model.param_list())
        print(f"[model] {name}: {n_params:,} params, dim={dim}, "
              f"layers={n_layers}, heads={n_heads}, ctx={ctx_len}")
        if old_name:
            print(f"[model] GREW: {old_name} → {name}")
        return True

    def tokenize(self):
        """Tokenize corpus, learning merges if needed."""
        if not self.bpe.corpus:
            return None
        if not self.bpe.merges:
            ids = self.bpe.learn(self.bpe.corpus)
        else:
            ids = self.bpe.retokenize()
        self._token_ids = ids
        return ids

    def train(self, steps=None, verbose=True):
        """Train for N steps. Returns average loss."""
        if not self._ensure_model():
            return None

        steps = steps or TRAIN_STEPS_PER_ROUND
        stage = get_stage(len(self.bpe.corpus))
        ctx_len = stage[4]

        if self._token_ids is None:
            self.tokenize()
        if self._token_ids is None or len(self._token_ids) < ctx_len + 1:
            if verbose:
                print(f"[train] not enough tokens ({len(self._token_ids) if self._token_ids else 0})")
            return None

        vocab = self.bpe.vocab_size
        ids = [min(t, vocab - 1) for t in self._token_ids]

        losses = []
        t0 = time.time()

        for step in range(steps):
            # random window
            off = random.randint(0, len(ids) - ctx_len - 1)
            tokens = ids[off:off + ctx_len]
            targets = ids[off + 1:off + ctx_len + 1]

            loss_idx, loss_val = self.model.forward_train(tokens, targets)
            self.model.backward_step(loss_idx, loss_val, LR)

            losses.append(loss_val)
            self.total_steps += 1

            if verbose and (step + 1) % 10 == 0:
                avg = sum(losses[-10:]) / len(losses[-10:])
                print(f"  step {self.total_steps} | train {avg:.4f}")

        elapsed = time.time() - t0
        avg_loss = sum(losses) / len(losses) if losses else 0
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss

        if verbose:
            print(f"[train] {len(losses)} steps in {elapsed:.1f}s | "
                  f"train {avg_loss:.4f} | best {self.best_loss:.4f} | "
                  f"stage={self.stage_name}")

        self._ensure_model()
        return avg_loss

    def generate(self, prompt, max_new=100, temperature=0.8):
        """Generate text from prompt."""
        if not self.model:
            return "[no model trained yet]"
        ids = self.bpe.encode(prompt)
        if not ids:
            ids = [0]
        ids = [min(t, self.bpe.vocab_size - 1) for t in ids]
        generated = self.model.generate(ids, max_new=max_new, temperature=temperature)
        return self.bpe.decode(generated)

    def save(self, path):
        if not self.model:
            return
        self.model.save_weights(path)
        # save metadata alongside
        meta_path = path + '.meta'
        with open(meta_path, 'w') as f:
            f.write(f"{self.stage_name}\n{self.total_steps}\n{self.best_loss}\n")
            f.write(f"{self.model.vocab_size}\n{self.model.dim}\n")
            f.write(f"{self.model.n_heads}\n{self.model.n_layers}\n{self.model.ctx_len}\n")
        print(f"[train] saved to {path}")

    def load(self, path):
        if not os.path.exists(path):
            return False
        meta_path = path + '.meta'
        if not os.path.exists(meta_path):
            return False
        with open(meta_path) as f:
            lines = f.read().strip().split('\n')
        if len(lines) < 8:
            return False
        self.stage_name = lines[0]
        self.total_steps = int(lines[1])
        self.best_loss = float(lines[2])
        vocab_size = int(lines[3])
        dim = int(lines[4])
        n_heads = int(lines[5])
        n_layers = int(lines[6])
        ctx_len = int(lines[7])
        self.model = NanoDurovModel(vocab_size, dim, n_heads, n_layers, ctx_len)
        if self.model.load_weights(path):
            n_params = sum(p.numel for p in self.model.param_list())
            print(f"[train] loaded: step={self.total_steps}, loss={self.best_loss:.4f}, "
                  f"params={n_params:,}")
            return True
        return False

# --- telegram client -----------------------------------------------------------
# MTProto observer. sees all messages including bot-to-bot.
# does NOT relay. does NOT forward. watches. learns. generates when asked.

async def run_telegram(trainer):
    if not TELETHON_AVAILABLE:
        print("[telegram] install telethon: pip install telethon")
        return

    api_id = int(os.environ.get('TELEGRAM_API_ID', 0))
    api_hash = os.environ.get('TELEGRAM_API_HASH', '')
    if not api_id or not api_hash:
        print("[telegram] set TELEGRAM_API_ID and TELEGRAM_API_HASH")
        print("[telegram] get them at https://my.telegram.org/apps")
        return

    client = TelegramClient('nanodurov_session', api_id, api_hash)
    await client.start()
    print("[telegram] connected")

    group_input = input("\nGroup @username or ID: ").strip()
    try:
        entity = await client.get_entity(group_input)
        title = getattr(entity, 'title', group_input)
        print(f"[telegram] watching: {title}\n")
    except Exception as e:
        print(f"[telegram] can't find group: {e}")
        return

    print("[telegram] loading history...")
    messages = await client.get_messages(entity, limit=500)
    for msg in reversed(messages):
        if msg.message:
            sender = await msg.get_sender()
            name = _sender_name(sender)
            line = f"[{name}]: {msg.message}"
            trainer.bpe.ingest(line)
    print(f"[telegram] ingested {len(trainer.bpe.corpus)} bytes from history")

    if len(trainer.bpe.corpus) > 500:
        trainer.tokenize()
        trainer.train(steps=TRAIN_STEPS_PER_ROUND)

    @client.on(events.NewMessage(chats=entity))
    async def handler(event):
        msg = event.message
        if not msg.message:
            return
        sender = await msg.get_sender()
        name = _sender_name(sender)
        bot = _is_bot(sender)
        tag = " [BOT]" if bot else ""
        ts = msg.date.strftime("%H:%M:%S") if msg.date else "??:??:??"
        print(f"[{ts}] {name}{tag}: {msg.message}")
        line = f"[{name}]: {msg.message}"
        trainer.bpe.ingest(line)

    last_corpus_size = len(trainer.bpe.corpus)
    async def auto_train():
        nonlocal last_corpus_size
        while True:
            await asyncio.sleep(AUTO_TRAIN_INTERVAL)
            corpus_size = len(trainer.bpe.corpus)
            new_bytes = corpus_size - last_corpus_size
            if corpus_size > 500 and new_bytes > 1024:
                print(f"\n[auto-train] +{new_bytes/1024:.1f}KB new data, training...")
                last_corpus_size = corpus_size
                trainer.tokenize()
                trainer.train(steps=TRAIN_STEPS_PER_ROUND)
                trainer.bpe.save('nanodurov_bpe.bin')
                trainer.save('nanodurov_weights.bin')
                print("[auto-train] done. watching...\n")

    async def input_loop():
        loop = asyncio.get_event_loop()
        while True:
            try:
                line = await loop.run_in_executor(None, lambda: input(""))
            except EOFError:
                break
            line = line.strip()
            if not line:
                continue

            if line == '/quit':
                print("saving...")
                trainer.bpe.save('nanodurov_bpe.bin')
                trainer.save('nanodurov_weights.bin')
                await client.disconnect()
                break
            elif line == '/train':
                trainer.tokenize()
                trainer.train(steps=TRAIN_STEPS_PER_ROUND)
            elif line.startswith('/generate') or line.startswith('/ai'):
                prompt = line.split(' ', 1)[1] if ' ' in line else '[User]: '
                text = trainer.generate(prompt)
                print(f"  > {text}")
            elif line == '/status':
                n = sum(p.numel for p in trainer.model.param_list()) if trainer.model else 0
                print(f"  stage={trainer.stage_name} params={n:,} "
                      f"steps={trainer.total_steps} loss={trainer.best_loss:.4f} "
                      f"corpus={len(trainer.bpe.corpus)/1024:.1f}KB "
                      f"vocab={trainer.bpe.vocab_size}")
            elif line == '/save':
                trainer.bpe.save('nanodurov_bpe.bin')
                trainer.save('nanodurov_weights.bin')
            elif line == '/history':
                msgs = await client.get_messages(entity, limit=20)
                for m in reversed(msgs):
                    if m.message:
                        s = await m.get_sender()
                        n = _sender_name(s)
                        b = " [BOT]" if _is_bot(s) else ""
                        ts = m.date.strftime("%H:%M:%S") if m.date else "??:??:??"
                        print(f"  [{ts}] {n}{b}: {m.message}")
            else:
                await client.send_message(entity, line)

    print("Commands: /train /generate <prompt> /ai <prompt> /status /save /history /quit")
    print(f"Anything else is sent as a message. Auto-train every {AUTO_TRAIN_INTERVAL}s.\n")

    await asyncio.gather(auto_train(), input_loop())

def _sender_name(sender):
    if sender is None:
        return "Unknown"
    if isinstance(sender, User):
        parts = [sender.first_name or '', sender.last_name or '']
        name = ' '.join(p for p in parts if p)
        return name or sender.username or f'User#{sender.id}'
    if hasattr(sender, 'title'):
        return sender.title or f'Chat#{sender.id}'
    return f'#{getattr(sender, "id", "?")}'

def _is_bot(sender):
    return isinstance(sender, User) and sender.bot

# --- main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='nanodurov — telegram chat that learns')
    parser.add_argument('--generate', type=str, help='generate from prompt (offline)')
    parser.add_argument('--train-only', type=str, help='train on text file (no telegram)')
    parser.add_argument('--steps', type=int, default=200, help='training steps')
    args = parser.parse_args()

    nt_seed(42)

    bpe = BPE(max_merges=256)
    bpe.load('nanodurov_bpe.bin')
    trainer = Trainer(bpe)
    trainer.load('nanodurov_weights.bin')

    if args.train_only:
        print(f"[main] training on {args.train_only}")
        with open(args.train_only, 'r') as f:
            text = f.read()
        bpe.ingest(text)
        trainer.tokenize()
        for r in range(args.steps // TRAIN_STEPS_PER_ROUND + 1):
            remaining = args.steps - r * TRAIN_STEPS_PER_ROUND
            if remaining <= 0:
                break
            trainer.train(steps=min(TRAIN_STEPS_PER_ROUND, remaining))
        bpe.save('nanodurov_bpe.bin')
        trainer.save('nanodurov_weights.bin')
        print("[main] done.")
        return

    if args.generate:
        text = trainer.generate(args.generate)
        print(text)
        return

    if not TELETHON_AVAILABLE:
        print("pip install telethon")
        sys.exit(1)

    print("""
╔═══════════════════════════════════════════════════╗
║               n a n o d u r o v                   ║
║     telegram client that learns from chat         ║
║     notorch + chuck. no pytorch. no excuses.      ║
╚═══════════════════════════════════════════════════╝
    """)

    asyncio.run(run_telegram(trainer))

if __name__ == '__main__':
    main()
