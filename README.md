# VoiceChanger for Rust Impl

High Quarity and Performance Voice Changer using WORLD Vocoder.


## Dependencies

- vst
- Rust-WORLD (my forked)
- rand


## Features

- [x] Synthesize voice with WORLD
- [ ] Choose F0 Detector from DIO and CREPE
- [x] Basic Buffering
- [ ] Multi-threaded Synthesize and Buffering
- [ ] GUI Configure
- [ ] Debug feature
- [ ] Tune Precision


## Design (phase 1)

1. Sync stream to buffer
2. Detect f0/formants/aperiodicity
3. Shift f0/formants
4. Synthesize voice
5. Apply output gain
6. Output signal


## Design (phase 2)

- Params
    - WORLD_INBUF_SIZE: WORLD Processing buffer size likely 4096, 8192,..
    - ADDER_SIZE: Over Adder
- Initialize
    1. Run processing thread
    2. Init source/dest queues
    3. Append empty signals to queues
- Stream process
    1. Add in-buffer to source queue
    2. Pop dest queue if it was queued, else return empty if not provied
    3. Apply output gain
    3. Output signal
- Background process
    1. Wait that source queue is stored to longer size than WORLD_INBUF_SIZE
    2. Pop source signal from queue as WORLD_INBUF_SIZE
    3. Detect f0/formants/aperiodicity
    4. Shift f0/formants, Update ap gain
    5. Synthesize voice
    6. Compose dest signals of previous and current
    7. Push dest signal to queue
    8. SAFETY: Drop overflow dest queue

## Author

&copy; 2021 sabamotto.
