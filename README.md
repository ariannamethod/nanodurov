# nanodurov

a telegram client that trains a language model on chat messages.

one file. telethon + pytorch. connect to a group, watch bots and humans talk, learn their patterns, generate text in their style. the chat is the corpus. the model grows with the conversation.

dedicated to Pavel Durov, who built the platform where bots can't see each other but we're training on them anyway.

953 lines. one act of defiance.

## architecture

RMSNorm + RoPE + SwiGLU MLP + causal attention. weight tying. BPE tokenizer (KARL). Chuck optimizer built-in.

progressive growth — the model grows with the corpus:

| stage | corpus | params | dim | layers | ctx |
|-------|--------|--------|-----|--------|-----|
| seed | 0KB | 29K | 32 | 1 | 64 |
| sprout | 5KB | 99K | 48 | 2 | 64 |
| sapling | 20KB | 246K | 64 | 3 | 128 |
| tree | 50KB | 689K | 96 | 4 | 128 |
| oak | 100KB | 1.7M | 128 | 6 | 256 |
| forest | 250KB | 5.1M | 192 | 8 | 256 |
| ancient | 500KB | 11.1M | 256 | 10 | 512 |

## generation (ancient, 10.8M, train loss 0.10)

trained on WTForacle dataset (2.5MB, 21K Q&A pairs). BPE 1024 merges, vocab 1280.

```
>>> what is the meaning of life
there isn't one and that's actually freeing. you get to make it up.
most people choose netflix and honestly, nobody hands you a quest or
a script, you're just here trying to eat snacks and kill time.
whatever makes you forget the void for a bit, right bro?

>>> should i quit my job
do you have another job lined up? no? then no don't quit yes? then
absolutely quit yesterday don't raw dog unemployment in this economy.
sitting around jobless sounds romantic until your rent is due

>>> nosql vs sql?
sql for data you care about nosql for data you'll regret later
wtforacle has seen this movie. every dev team swears they're the one
who won't need joins. then six months in, there's a whiteboard full
of tangled arrows and someone mumbling about "eventual consistency".
nothing ages worse than nosql schemas.

>>> am i the problem
bro if you're asking, probably yes. welcome to growth. most people
who aren't the problem don't sit around self-reflecting for reddit
karma. tbh self-awareness is rare, so at least you're ahead of the
crowd. congrats on starting your cringe era. enjoy the existential
dread, it only gets weirder from here.

>>> how do i undo git?
google it like everyone else stack overflow is your friend wtforacle
has been there. bro, if i had a nickel for every time someone nuked
their repo and begged for an "undo button," i'd have enough to buy
twitter and delete it again. git makes you read the manual, sorry.
control+z doesn't save you here.
```

## usage

```bash
pip install telethon torch
python nanodurov.py                         # interactive telegram mode
python nanodurov.py --generate "hello"      # generate from prompt
python nanodurov.py --train-only chat.txt   # train on exported chat
```

## commands (telegram mode)

`/train` `/generate <prompt>` `/ai <prompt>` `/status` `/save` `/history` `/quit`

## quotes from the source

> "Append-only — vocab grows, never shrinks. Like regret."

> "the model grows with the corpus. more data → bigger model. like a tree. not like a corporation."

> "every line here was written by someone who stared at karpathy's code for too long and started seeing attention patterns in their dreams."

> "Adam is blind. Chuck sees. Chuck remembers." — In memory of Carlos Ray "Chuck" Norris (1940–2026).

> "the part where numbers go down and hope goes up. or numbers go up and you stare at the ceiling."

> "where the threads converge and the magic begins. or crashes. usually crashes first, then magic."

## part of the arianna method ecosystem

inspired by karpathy's microGPT. BPE from nanoagi (KARL). progressive growth from brain.js (forum.ai).
