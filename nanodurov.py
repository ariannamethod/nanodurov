"""
nanodurov.py — a telegram client that trains a language model on chat messages.

one file. telethon + pytorch. connect to a group, watch bots and humans talk,
learn their patterns, generate text in their style. the chat is the corpus.
the model grows with the conversation.

inspired by karpathy's microGPT. dedicated to Pavel Durov, who built the
platform where bots can't see each other but we're training on them anyway.

usage:
    pip install telethon torch
    python nanodurov.py                         # interactive mode
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
from collections import defaultdict

# --- optional imports (graceful degradation) ------------------------------------

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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

BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 0.01
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
        """Add text to corpus with dedup and quality filter.
        Rejects: too short, duplicate, too repetitive, pure URLs,
        sticker-only, emoji-only, single-word noise."""
        if isinstance(text, str):
            raw = text
            text = text.encode('utf-8', errors='replace')
        else:
            raw = text.decode('utf-8', errors='replace')
        if len(text) < 15:
            return False
        # dedup
        h = hashlib.sha256(text).hexdigest()[:16]
        if h in self.seen_hashes:
            return False
        # quality filters
        stripped = raw.strip()
        # skip pure URLs
        if stripped.startswith('http://') or stripped.startswith('https://'):
            if ' ' not in stripped:
                return False
        # skip if >70% non-alpha (stickers, emoji floods, binary)
        alpha = sum(1 for c in stripped if c.isalpha() or c.isspace())
        if len(stripped) > 0 and alpha / len(stripped) < 0.3:
            return False
        # skip too repetitive (same char >50%)
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

# --- transformer model ---------------------------------------------------------
# RMSNorm, RoPE, SwiGLU, causal attention. the microGPT recipe.
# every line here was written by someone who stared at karpathy's code
# for too long and started seeing attention patterns in their dreams.

if TORCH_AVAILABLE:

    class RMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.w = nn.Parameter(torch.ones(dim))
        def forward(self, x):
            return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-5).type_as(x) * self.w

    class Attention(nn.Module):
        def __init__(self, dim, n_heads):
            super().__init__()
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            self.wq = nn.Linear(dim, dim, bias=False)
            self.wk = nn.Linear(dim, dim, bias=False)
            self.wv = nn.Linear(dim, dim, bias=False)
            self.wo = nn.Linear(dim, dim, bias=False)

        def forward(self, x, freqs_cos, freqs_sin):
            B, T, D = x.shape
            H, HD = self.n_heads, self.head_dim

            q = self.wq(x).view(B, T, H, HD).transpose(1, 2)  # [B, H, T, HD]
            k = self.wk(x).view(B, T, H, HD).transpose(1, 2)
            v = self.wv(x).view(B, T, H, HD).transpose(1, 2)

            # RoPE
            q = apply_rope(q, freqs_cos, freqs_sin)
            k = apply_rope(k, freqs_cos, freqs_sin)

            # causal attention
            att = (q @ k.transpose(-2, -1)) / math.sqrt(HD)
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            att = att.masked_fill(mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            out = att @ v  # [B, H, T, HD]

            out = out.transpose(1, 2).contiguous().view(B, T, D)
            return self.wo(out)

    class MLP(nn.Module):
        def __init__(self, dim, hidden):
            super().__init__()
            self.w_gate = nn.Linear(dim, hidden, bias=False)
            self.w_up = nn.Linear(dim, hidden, bias=False)
            self.w_down = nn.Linear(hidden, dim, bias=False)
        def forward(self, x):
            return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

    class Block(nn.Module):
        def __init__(self, dim, n_heads, hidden):
            super().__init__()
            self.norm1 = RMSNorm(dim)
            self.attn = Attention(dim, n_heads)
            self.norm2 = RMSNorm(dim)
            self.mlp = MLP(dim, hidden)
        def forward(self, x, freqs_cos, freqs_sin):
            x = x + self.attn(self.norm1(x), freqs_cos, freqs_sin)
            x = x + self.mlp(self.norm2(x))
            return x

    class NanoDurov(nn.Module):
        def __init__(self, vocab_size, dim, n_heads, n_layers, ctx_len):
            super().__init__()
            self.ctx_len = ctx_len
            self.tok_emb = nn.Embedding(vocab_size, dim)
            self.blocks = nn.ModuleList([
                Block(dim, n_heads, dim * 4) for _ in range(n_layers)
            ])
            self.norm_f = RMSNorm(dim)
            self.head = nn.Linear(dim, vocab_size, bias=False)
            # weight tying
            self.head.weight = self.tok_emb.weight
            # precompute RoPE
            self.register_buffer('freqs_cos', None)
            self.register_buffer('freqs_sin', None)
            self._build_rope(ctx_len, dim // n_heads)
            self.apply(self._init_weights)

        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

        def _build_rope(self, max_len, head_dim):
            pos = torch.arange(max_len).unsqueeze(1)  # [T, 1]
            dim_pairs = torch.arange(0, head_dim, 2).float()  # [HD/2]
            freqs = 1.0 / (10000 ** (dim_pairs / head_dim))  # [HD/2]
            angles = pos * freqs  # [T, HD/2]
            self.freqs_cos = angles.cos()  # [T, HD/2]
            self.freqs_sin = angles.sin()

        def forward(self, idx, targets=None):
            B, T = idx.shape
            x = self.tok_emb(idx)
            fc = self.freqs_cos[:T].unsqueeze(0)  # [1, T, HD/2]
            fs = self.freqs_sin[:T].unsqueeze(0)
            for block in self.blocks:
                x = block(x, fc, fs)
            x = self.norm_f(x)
            logits = self.head(x)

            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return logits, loss

        def generate(self, idx, max_new=100, temperature=0.8, top_k=40):
            for _ in range(max_new):
                ctx = idx[:, -self.ctx_len:]
                logits, _ = self(ctx)
                logits = logits[:, -1, :] / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, [-1]]] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
                idx = torch.cat([idx, next_id], dim=1)
                # stop on newline after some output
                if idx.shape[1] > 10 and next_id.item() == 10:
                    break
            return idx

    def apply_rope(x, cos, sin):
        """Apply rotary position embedding."""
        # x: [B, H, T, HD]
        d2 = x.shape[-1] // 2
        x1 = x[..., :d2]
        x2 = x[..., d2:]
        # cos, sin: [1, T, HD/2] → need [1, 1, T, HD/2] for broadcasting
        cos = cos.unsqueeze(1)  # [1, 1, T, HD/2]
        sin = sin.unsqueeze(1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# --- chuck optimizer ------------------------------------------------------------
# θ -= (α × S × λ_Ψ × λ_l × σ) × m̂/(√v̂ + ε) + η
# Adam is blind. Chuck sees. Chuck remembers.
# In memory of Carlos Ray "Chuck" Norris (1940–2026).
#
# Compact version: Levels 1 (loss trend), 2 (grad trend), 9 (macro patience).
# Full version: github.com/ariannamethod/chuck

if TORCH_AVAILABLE:

    class Chuck(torch.optim.Optimizer):
        """Self-aware optimizer. Drop-in AdamW replacement with dampen/boost.

        When loss is falling → boost (dampen > 1). When rising → brake (dampen < 1).
        When stagnating → inject noise. Macro patience drops LR on plateaus.
        """

        def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
                     weight_decay=0.01, window=16, macro_int=500, macro_pat=3,
                     macro_decay=0.5, verbose=0):
            defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            super().__init__(params, defaults)
            self.window = window
            self.macro_int = macro_int
            self.macro_pat = macro_pat
            self.macro_decay = macro_decay
            self.verbose = verbose

            # Chuck's soul
            self.dampen = 1.0
            self.noise = 0.0
            self.loss_ema = 0.0
            self.gnorm_ema = 0.0
            self.macro_ema = 0.0
            self.best_macro = 1e9
            self.lr_scale = 1.0
            self.macro_stag = 0
            self.macro_drops = 0
            self.global_step = 0

            # loss ring buffer
            self._hist = [0.0] * window
            self._hpos = 0
            self._hfull = False
            self._stag = 0

        @torch.no_grad()
        def step(self, closure=None, *, loss=None):
            if closure is not None:
                with torch.enable_grad():
                    lv = closure()
                    if loss is None:
                        loss = lv.item()

            self.global_step += 1
            W = self.window

            # === Level 1: loss trend → dampen/boost ===
            if loss is not None:
                if self.loss_ema == 0.0:
                    self.loss_ema = loss
                else:
                    self.loss_ema = 0.99 * self.loss_ema + 0.01 * loss

                self._hist[self._hpos % W] = self.loss_ema
                self._hpos += 1
                if self._hpos >= W:
                    self._hfull = True

                if self._hfull:
                    q = W // 4
                    recent = sum(self._hist[(self._hpos - 1 - i) % W] for i in range(q)) / q
                    old = sum(self._hist[(self._hpos - W + i) % W] for i in range(q)) / q
                    trend = (recent - old) / (old + 1e-8)
                    if trend > 0.02:
                        self.dampen *= 0.97   # loss rising → brake
                    elif trend < -0.02:
                        self.dampen *= 1.03   # loss falling → push
                    if abs(trend) < 0.001:
                        self._stag += 1
                        if self._stag > 8:
                            self.noise = 0.001
                            self._stag = 0
                    else:
                        self._stag = 0
                        self.noise *= 0.9
                    # mean reversion
                    self.dampen = 0.999 * self.dampen + 0.001 * 1.0
                    self.dampen = max(0.3, min(2.0, self.dampen))

                # === Level 9: macro patience ===
                if self.macro_ema == 0.0:
                    self.macro_ema = loss
                else:
                    self.macro_ema = 0.999 * self.macro_ema + 0.001 * loss
                if self.global_step % self.macro_int == 0 and self.global_step > W:
                    if self.macro_ema > self.best_macro * 0.999:
                        self.macro_stag += 1
                        if self.macro_stag >= self.macro_pat:
                            self.lr_scale *= self.macro_decay
                            if self.lr_scale < 0.05:
                                self.lr_scale = 0.05
                            self.macro_stag = 0
                            self.macro_drops += 1
                    else:
                        self.best_macro = self.macro_ema
                        self.macro_stag = 0
                        if self.lr_scale < 1.0:
                            self.lr_scale = min(1.0, self.lr_scale * 1.2)

            # === Adam update with Chuck modulation ===
            effective_dampen = self.dampen * self.lr_scale
            for group in self.param_groups:
                lr = group['lr'] * effective_dampen
                beta1, beta2 = group['betas']
                eps = group['eps']
                wd = group['weight_decay']

                for p in group['params']:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]

                    if len(state) == 0:
                        state['step'] = 0
                        state['m'] = torch.zeros_like(p)
                        state['v'] = torch.zeros_like(p)

                    state['step'] += 1
                    m, v = state['m'], state['v']

                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    bc1 = 1 - beta1 ** state['step']
                    bc2 = 1 - beta2 ** state['step']
                    m_hat = m / bc1
                    v_hat = v / bc2

                    # noise injection on stagnation
                    if self.noise > 0:
                        m_hat = m_hat + self.noise * torch.randn_like(m_hat)

                    # weight decay (decoupled)
                    if wd > 0:
                        p.add_(p, alpha=-lr * wd)

                    # update
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)

            if self.verbose > 0 and self.global_step % self.verbose == 0:
                print(f"  chuck: step={self.global_step} λ={self.dampen:.3f} "
                      f"lr_scale={self.lr_scale:.3f} noise={self.noise:.4f} "
                      f"macro_drops={self.macro_drops}")

# --- training loop --------------------------------------------------------------
# the part where numbers go down and hope goes up.
# or numbers go up and you stare at the ceiling.

class Trainer:
    def __init__(self, bpe, device='cpu'):
        self.bpe = bpe
        self.device = device
        self.model = None
        self.optimizer = None
        self.stage_name = None
        self.total_steps = 0
        self.best_loss = float('inf')
        self._token_ids = None

    def _ensure_model(self):
        """Create or grow model based on corpus size."""
        if not TORCH_AVAILABLE:
            print("[train] no pytorch. install: pip install torch")
            return False

        stage = get_stage(len(self.bpe.corpus))
        _, dim, n_heads, n_layers, ctx_len, max_merges, name = stage

        if self.stage_name == name and self.model is not None:
            return True

        old_name = self.stage_name
        old_state = self.model.state_dict() if self.model else None

        self.bpe.max_merges = max_merges
        vocab_size = 256 + max_merges

        self.model = NanoDurov(vocab_size, dim, n_heads, n_layers, ctx_len)
        self.model.to(self.device)

        # copy weights from old model where shapes match
        if old_state:
            new_state = self.model.state_dict()
            copied = 0
            for k in old_state:
                if k in new_state and old_state[k].shape == new_state[k].shape:
                    new_state[k] = old_state[k]
                    copied += 1
                elif k in new_state and len(old_state[k].shape) == len(new_state[k].shape):
                    # partial copy: take min of each dim
                    old_t = old_state[k]
                    new_t = new_state[k]
                    slices = tuple(slice(0, min(o, n)) for o, n in zip(old_t.shape, new_t.shape))
                    new_state[k][slices] = old_t[slices]
                    copied += 1
            self.model.load_state_dict(new_state)
            print(f"[model] GREW: {old_name} → {name} (copied {copied} tensors)")

        self.optimizer = Chuck(
            self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
            verbose=0)
        self.stage_name = name

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[model] {name}: {n_params:,} params, dim={dim}, "
              f"layers={n_layers}, heads={n_heads}, ctx={ctx_len}")
        return True

    def _get_batch(self, token_ids, batch_size, ctx_len):
        """Random batch of training windows."""
        n = len(token_ids)
        if n <= ctx_len + 1:
            return None, None
        ix = torch.randint(0, n - ctx_len - 1, (batch_size,))
        x = torch.stack([torch.tensor(token_ids[i:i+ctx_len], dtype=torch.long) for i in ix])
        y = torch.stack([torch.tensor(token_ids[i+1:i+ctx_len+1], dtype=torch.long) for i in ix])
        return x.to(self.device), y.to(self.device)

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

        # tokenize if needed
        if self._token_ids is None:
            self.tokenize()
        if self._token_ids is None or len(self._token_ids) < ctx_len + 1:
            if verbose:
                print(f"[train] not enough tokens ({len(self._token_ids) if self._token_ids else 0})")
            return None

        # clamp token ids to vocab
        vocab = self.bpe.vocab_size
        ids = [min(t, vocab - 1) for t in self._token_ids]

        self.model.train()
        losses = []
        t0 = time.time()

        for step in range(steps):
            x, y = self._get_batch(ids, BATCH_SIZE, ctx_len)
            if x is None:
                break
            _, loss = self.model(x, y)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step(loss=loss.item())

            losses.append(loss.item())
            self.total_steps += 1

            if verbose and (step + 1) % 10 == 0:
                avg = sum(losses[-10:]) / len(losses[-10:])
                print(f"  step {self.total_steps} | loss {avg:.4f}")

        elapsed = time.time() - t0
        avg_loss = sum(losses) / len(losses) if losses else 0
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss

        if verbose:
            print(f"[train] {len(losses)} steps in {elapsed:.1f}s | "
                  f"loss {avg_loss:.4f} | best {self.best_loss:.4f} | "
                  f"stage={self.stage_name}")

        # check growth after training
        self._ensure_model()
        return avg_loss

    @torch.no_grad()
    def generate(self, prompt, max_new=100, temperature=0.8):
        """Generate text from prompt."""
        if not self.model:
            return "[no model trained yet]"
        self.model.eval()
        ids = self.bpe.encode(prompt)
        if not ids:
            ids = [0]
        # clamp to vocab
        ids = [min(t, self.bpe.vocab_size - 1) for t in ids]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(idx, max_new=max_new, temperature=temperature)
        generated = out[0, len(ids):].tolist()
        return self.bpe.decode(generated)

    def save(self, path):
        if not self.model:
            return
        ckpt = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'stage': self.stage_name,
            'total_steps': self.total_steps,
            'best_loss': self.best_loss,
        }
        # save Chuck's soul
        if hasattr(self.optimizer, 'dampen'):
            ckpt['chuck'] = {
                'dampen': self.optimizer.dampen,
                'lr_scale': self.optimizer.lr_scale,
                'loss_ema': self.optimizer.loss_ema,
                'macro_ema': self.optimizer.macro_ema,
                'best_macro': self.optimizer.best_macro,
                'macro_drops': self.optimizer.macro_drops,
                'global_step': self.optimizer.global_step,
            }
        torch.save(ckpt, path)
        print(f"[train] saved checkpoint to {path}")

    def load(self, path):
        if not os.path.exists(path):
            return False
        if not self._ensure_model():
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        try:
            self.model.load_state_dict(ckpt['model'], strict=False)
            self.total_steps = ckpt.get('total_steps', 0)
            self.best_loss = ckpt.get('best_loss', float('inf'))
            # restore Chuck's soul
            if 'chuck' in ckpt and hasattr(self.optimizer, 'dampen'):
                cs = ckpt['chuck']
                self.optimizer.dampen = cs.get('dampen', 1.0)
                self.optimizer.lr_scale = cs.get('lr_scale', 1.0)
                self.optimizer.loss_ema = cs.get('loss_ema', 0.0)
                self.optimizer.macro_ema = cs.get('macro_ema', 0.0)
                self.optimizer.best_macro = cs.get('best_macro', 1e9)
                self.optimizer.macro_drops = cs.get('macro_drops', 0)
                self.optimizer.global_step = cs.get('global_step', 0)
            # try to restore optimizer state (may fail after growth)
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
            except (ValueError, KeyError):
                pass  # model grew, optimizer state doesn't match — fresh Adam state
            print(f"[train] loaded checkpoint: step={self.total_steps}, loss={self.best_loss:.4f}")
            return True
        except Exception as e:
            print(f"[train] checkpoint load failed (model grew?): {e}")
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

    # choose group
    group_input = input("\nGroup @username or ID: ").strip()
    try:
        entity = await client.get_entity(group_input)
        title = getattr(entity, 'title', group_input)
        print(f"[telegram] watching: {title}\n")
    except Exception as e:
        print(f"[telegram] can't find group: {e}")
        return

    # load history
    print("[telegram] loading history...")
    messages = await client.get_messages(entity, limit=500)
    for msg in reversed(messages):
        if msg.message:
            sender = await msg.get_sender()
            name = _sender_name(sender)
            bot = _is_bot(sender)
            line = f"[{name}]: {msg.message}"
            trainer.bpe.ingest(line)
    print(f"[telegram] ingested {len(trainer.bpe.corpus)} bytes from history")

    # initial train if we have data
    if len(trainer.bpe.corpus) > 500:
        trainer.tokenize()
        trainer.train(steps=TRAIN_STEPS_PER_ROUND)

    # message handler — observe + ingest
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

        # ingest
        line = f"[{name}]: {msg.message}"
        trainer.bpe.ingest(line)

    # auto-train loop — only trains when there's meaningful new data
    last_corpus_size = len(trainer.bpe.corpus)
    async def auto_train():
        nonlocal last_corpus_size
        while True:
            await asyncio.sleep(AUTO_TRAIN_INTERVAL)
            corpus_size = len(trainer.bpe.corpus)
            new_bytes = corpus_size - last_corpus_size
            # only train if at least 1KB of new data since last train
            if corpus_size > 500 and new_bytes > 1024:
                print(f"\n[auto-train] +{new_bytes/1024:.1f}KB new data, training...")
                last_corpus_size = corpus_size
                trainer.tokenize()
                trainer.train(steps=TRAIN_STEPS_PER_ROUND)
                trainer.bpe.save('nanodurov_bpe.bin')
                trainer.save('nanodurov_ckpt.pt')
                print("[auto-train] done. watching...\n")

    # input handler — user can type messages or commands
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
                trainer.save('nanodurov_ckpt.pt')
                await client.disconnect()
                break
            elif line == '/train':
                trainer.tokenize()
                trainer.train(steps=TRAIN_STEPS_PER_ROUND)
            elif line.startswith('/generate') or line.startswith('/ai'):
                prompt = line.split(' ', 1)[1] if ' ' in line else '[User]: '
                text = trainer.generate(prompt)
                print(f"  🧠 {text}")
            elif line == '/status':
                n = sum(p.numel() for p in trainer.model.parameters()) if trainer.model else 0
                print(f"  stage={trainer.stage_name} params={n:,} "
                      f"steps={trainer.total_steps} loss={trainer.best_loss:.4f} "
                      f"corpus={len(trainer.bpe.corpus)/1024:.1f}KB "
                      f"vocab={trainer.bpe.vocab_size}")
            elif line == '/save':
                trainer.bpe.save('nanodurov_bpe.bin')
                trainer.save('nanodurov_ckpt.pt')
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
                # send as user message
                await client.send_message(entity, line)

    print("Commands: /train /generate <prompt> /ai <prompt> /status /save /history /quit")
    print("Anything else is sent as a message. Auto-train runs every "
          f"{AUTO_TRAIN_INTERVAL}s.\n")

    await asyncio.gather(
        auto_train(),
        input_loop(),
    )

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
# where the threads converge and the magic begins.
# or crashes. usually crashes first, then magic.

def main():
    parser = argparse.ArgumentParser(description='nanodurov — telegram chat that learns')
    parser.add_argument('--generate', type=str, help='generate from prompt (offline)')
    parser.add_argument('--train-only', type=str, help='train on text file (no telegram)')
    parser.add_argument('--steps', type=int, default=200, help='training steps')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda or mps')
    args = parser.parse_args()

    # init
    bpe = BPE(max_merges=256)
    bpe.load('nanodurov_bpe.bin')
    trainer = Trainer(bpe, device=args.device)
    trainer.load('nanodurov_ckpt.pt')

    if args.train_only:
        # offline training on text file
        print(f"[main] training on {args.train_only}")
        with open(args.train_only, 'r') as f:
            text = f.read()
        bpe.ingest(text)
        trainer.tokenize()
        for r in range(args.steps // TRAIN_STEPS_PER_ROUND + 1):
            trainer.train(steps=min(TRAIN_STEPS_PER_ROUND, args.steps - r * TRAIN_STEPS_PER_ROUND))
        bpe.save('nanodurov_bpe.bin')
        trainer.save('nanodurov_ckpt.pt')
        print("[main] done.")
        return

    if args.generate:
        text = trainer.generate(args.generate)
        print(text)
        return

    # telegram mode
    if not TELETHON_AVAILABLE:
        print("pip install telethon")
        sys.exit(1)
    if not TORCH_AVAILABLE:
        print("pip install torch")
        sys.exit(1)

    print("""
╔═══════════════════════════════════════════════════╗
║               n a n o d u r o v                   ║
║     telegram client that learns from chat         ║
║     one file. one model. one act of defiance.     ║
╚═══════════════════════════════════════════════════╝
    """)

    asyncio.run(run_telegram(trainer))

if __name__ == '__main__':
    main()
