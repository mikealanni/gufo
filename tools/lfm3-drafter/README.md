# LFM3 drafter heads

Medusa-style heads for the NPU generation engine, trained on the model's own greedy continuations.

1. `build_prompts.py` writes `gen_prompts.docs`; append `<think></think>` after the assistant header (the serving format).
2. `gufo judge --model LFM3-1.7B-Q8_0.gguf --gen-data gen_prompts2.docs --gen-out selfgen2 --streams 128 --gen-len 224`
3. `gufo judge --model LFM3-1.7B-Q8_0.gguf --capture selfgen2.tokens --capture-out sg2`
4. `python train_heads2.py 5 3 sg2 heads3.bin`
5. `gufo judge --model LFM3-1.7B-Q8_0.gguf --generate "..." --drafter heads3.bin`

The drafter format must match the serving prompt format; heads trained without the `<think></think>` block lose most of their acceptance.
