# Engineering Thought Process: Intelligent Video Chunking (C-MASP)


## PAGE 1: The Problem, Our Engineering Trade-Offs & The Pipeline

### 1. What Problem Are We Actually Solving?
Imagine a company has thousands of hours of mixed videos—meetings, lectures, sports, concerts, and security camera feeds. They want users to search inside these videos by asking text questions (Video RAG / AI search). 

Before an AI or vector database can index a video, we have to slice (chunk) the long video into smaller clips (between **10 and 45 seconds** long). When we looked at how standard systems do this, we found **three major traps**:

1. **Trap 1 — Cutting on a Fixed Timer (e.g., every 30 seconds):**
   * Cutting a video every 30 seconds is like ripping pages out of a book without looking at the sentences. On our benchmark, a 30-second timer **cuts speakers off mid-sentence 47.8% of the time**, ruining the transcript for text search. It also mixes unrelated scenes into the same clip (like 15 seconds of a presenter's face + 15 seconds of a code screen), which confuses the search engine.
2. **Trap 2 — Cutting Only When the Camera Angle Changes (Visual Shot Detection):**
   * In sports or commercials, the camera switches every 1 to 2 seconds. Cutting on every camera switch creates hundreds of tiny micro-clips, multiplying database storage costs by 10x.
   * In a 30-minute lecture or static security video, the camera *never* switches—creating one giant 30-minute chunk that crashes downstream AI token limits.
   * Even worse, editors change camera angles while people are talking, so visual-only cuts still **break 53.7% of spoken sentences**.
3. **Trap 3 — Running Heavy AI Models on Every Frame:**
   * A 1-hour video at 30 FPS has **108,000 frames**. Running a heavy Vision-Language Model on every frame is far too slow and expensive for large video libraries.

---

### 2. How We Balanced Accuracy, Latency (Speed), and Model Size
Our goal was to build a **100% training-free system** that is fast, cheap to run at scale, and accurate across all video types. We made three core engineering trade-offs:

```text
                             ACCURACY (Overall Quality Q_chunk)
                                       ▲
                                       │
                      C-MASP (Fast-Spatial CPU) ◄── PRODUCTION SWEET SPOT
                         [Q = 1.2817 | 89.8x RT]    0 MB Weights, 0 MB VRAM, Runs on CPU!
                                       │
                      C-MASP (OpenCLIP ViT GPU)
                         [Q = 1.1128 | 66.0x RT]    605 MB Weights, GPU FP16 Batching
                                       │
           Visual-Cut-Only             │
             [Q = 0.4548]              │
                                       │
    Fixed-30s                          │
   [Q = 0.2881]                        │
  ───────┼─────────────────────────────┼───────────────────────► SPEED / THROUGHPUT
        10x                           90x                        (Realtime Multiplier)
```

* **Decision 1: Sample Sparsely at 2 Frames Per Second (Latency vs. Precision):**
  Instead of decoding 30 frames per second, we grab **2 frames per second** (1 frame every 0.5 seconds) in a single pass. This skips **93.3% of unnecessary frame decoding** while still giving us 0.5-second accuracy to spot scene changes.
* **Decision 2: Dual Engine Choice — 0 MB CPU Math vs. 605 MB GPU Neural Model (Model Size vs. Cost):**
  * **Mode A (`fast_spatial` on CPU — 0 MB weights, 0 MB VRAM):** Uses pure math and classical image features (color layout, texture frequency, and edge changes). It requires **zero model downloads**, runs on any basic CPU at **89.8x real-time speed (2,693 FPS)**, and actually beats deep neural nets on lectures and slide presentations because it catches sharp text/diagram changes.
  * **Mode B (`open_clip` ViT-B-32 on GPU — 605 MB weights, ~591 MB VRAM):** Uses a pre-trained Vision Transformer in fast FP16 batches on GPU at **66.0x real-time speed (1,979 FPS)**. Best when customers have GPUs and want deep semantic grouping across movie/documentary camera angles.
* **Decision 3: Hard Time Limits ($[10\text{s}, 45\text{s}]$ with a $20\text{s}$ Target):**
  We enforce a strict rule that no chunk can be shorter than 10s or longer than 45s. Because the solver only searches within this 10s–45s window, its compute cost is **strictly linear $\mathcal{O}(T)$**—solving a 1-hour video in **under 2 milliseconds**.

---

### 3. How Our 4-Tier Pipeline Works (Step-by-Step)
1. **Tier 0 — Fast Frame Grabber (`visual.py`):** Reads the video once at 2 FPS into small thumbnails ($64\times 64$ for CPU, $224\times 224$ for GPU).
2. **Tier 1 — Visual Cut & Idle Camera Detector (`visual.py`):** Measures color shifts (HSV) and text/edge changes (Sobel). It uses a sliding median filter (CFAR) so continuous motion in sports doesn't trigger false cuts. If a security camera has zero motion and no speech for $\ge 15\text{s}$, it marks the chunk as `STATIC_IDLE` so the database can skip indexing empty footage.
3. **Tier 2 — Speech Pause Detector (`discourse.py`):** Reads `.srt` subtitles (or falls back to raw audio volume envelopes if subtitles are missing) to map out where people are speaking and where they pause.
4. **Tier 3 — Scene Meaning Tracker (`semantic.py`):** Extracts a 192-D visual fingerprint (`fast_spatial`) or 512-D ViT embedding (`open_clip`) for each frame and spots major topic shifts.
5. **Tier 4 — Global Timeline Optimizer & Speech Snapping (`dp_solver.py`, `pipeline.py`):** Uses Dynamic Programming (Viterbi shortest-path search) to pick the best cut points where **the visual scene changes AND the speaker finishes their sentence**, then snaps the cut to the exact millisecond speech pause.

---

## PAGE 2: Evaluation Metrics, Key Debugging Wins & Final Results

### 4. How Do We Grade a Chunker Without Human Labels? (Our 5 Metrics)
Since real customer videos don't come with manual "ground-truth cut" labels, we combined **standard operational checks** with **novel statistical metrics**:

| Metric | Simple Meaning | How It Works | Why It Matters for Video Search |
| :--- | :--- | :---: | :--- |
| **$\text{SCR} \uparrow$**<br>*(Semantic Cohesion Ratio)* | **One chunk = One clean scene.** | $\frac{\text{Difference between neighbor chunks}}{\text{Difference of frames inside chunk}}$ | High $\text{SCR}$ proves frames inside a chunk belong together, giving a sharp vector embedding. |
| **$\text{DBFI} \downarrow$**<br>*(Discourse Fracture Index)* | **Never cut mid-sentence.** | $\frac{\text{Cuts that break a sentence}}{\text{Total internal cuts}}$ | **$0.0000 = 0\%$ broken sentences.** Keeps spoken thoughts 100% intact for text search. |
| **$\text{IPU} \uparrow$**<br>*(Information Path Uniformity)* | **Balanced action per chunk.** | $\frac{1}{1 + \text{Variation of action per chunk}}$ | High $\text{IPU}$ (closer to $1.0$) means every chunk carries a similar amount of visual change. |
| **$\text{Q}_{\text{chunk}} \uparrow$**<br>*(Composite Quality Score)* | **The overall "North Star" grade.** | $\text{SCR} \times (1 - \text{DBFI}) \times \text{IPU}$ | Combines scene purity, intact speech, and balanced action into one honest score. |
| **$\text{CSR} \uparrow$**<br>*(Constraint Satisfaction)* | **Obey the $[10\text{s}, 45\text{s}]$ rules.** | $\frac{\text{Chunks between 10s and 45s}}{\text{Total chunks}}$ | **$1.0000 = 100\%$ compliance.** Prevents tiny micro-clips or huge context-breaking clips. |

---

### 5. Three Real-World Bugs We Discovered & Solved on VideoMME
When we first ran our system on real YouTube videos from the VideoMME benchmark, we investigated the data and solved three real-world engineering problems:

1. **The YouTube Overlapping Rolling-Caption Bug:**
   * *What happened:* On several videos, our sentence fracture rate ($\text{DBFI}$) was unexpectedly high. Looking at the `.srt` files, we realized YouTube auto-captions display **2 rolling lines at a time**—so Caption #2's start time is 2–4 seconds *before* Caption #1's end time! A standard parser thinks someone is talking non-stop with zero pauses. Also, `[Music]` and `[Applause]` tags were treated as spoken sentences.
   * *Our Fix:* We filtered out `[Music]`/`[Applause]` tags, removed repeated rolling words, and clamped overlapping timestamps so every caption ends when the next line starts. This immediately dropped sentence fractures to **0.0% ($\text{DBFI} = 0.0000$)**.
2. **The Brightness Spike Bug in Classical Image Math (2D-DCT):**
   * *What happened:* In OpenCV's Discrete Cosine Transform (DCT), the top-left `[0, 0]` number measures overall screen brightness and is scaled **32x larger** than all the other 63 texture numbers—making up over 90% of the vector! Because of this, two completely different scenes with similar brightness looked 96% identical to the math solver.
   * *Our Fix:* We set the `[0, 0]` brightness spike to zero and balanced all 4 feature blocks equally, increasing scene separation ($\text{SCR}$) by **+33.6%**.
3. **Boundary Frame Bleed & Chunk-Length Weighting:**
   * *What happened:* When a camera cut happened at frame $j$, using closed intervals $[i, j]$ accidentally put the first frame of the *new* scene at the end of the *old* chunk. Also, unweighted variance discouraged splitting short commercials into scenes.
   * *Our Fix:* We switched to half-open intervals $[i, j)$ (so frame $j$ cleanly starts the new chunk) and weighted variance by chunk duration, doubling our overall quality score.

---

### 6. Final Benchmark Results (VideoMME 10% Sample — 2.9 Hours of Video)
We tested C-MASP against two baselines across 10 diverse videos (**10,389 seconds / ~2.9 hours** of sports, lectures, commercials, concerts, and movies):

| Method | Model Size & Hardware | $\text{SCR} \uparrow$ *(Scene Purity)* | $\text{DBFI} \downarrow$ *(Broken Speech)* | $\text{IPU} \uparrow$ *(Balance)* | $\text{Q}_{\text{chunk}} \uparrow$ *(Overall Grade)* | $\text{CSR} \uparrow$ *(10s–45s Rule)* | Speed ($\times\text{RT}$) | Total Time (2.9h Video) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed 30s Timer** | $0\text{ MB}$ (CPU) | $0.8232$ | $0.4778$ *(47.8% broken)* | $0.7103$ | $0.2881$ | $0.9254$ | $>300\times$ | $35\text{ s}$ |
| **Visual Cut Only** | $0\text{ MB}$ (CPU) | $2.0461$ | $0.5374$ *(53.7% broken)* | $0.6716$ | $0.4548$ | $0.9227$ | $150\times$ | $70\text{ s}$ |
| **C-MASP (`fast_spatial`)** | **$0\text{ MB}$ (CPU)** | **$1.8186$** | **$0.0000$ (0% broken!)** | **$0.7568$** | **$1.2817$ (4.45x better)** | **$1.0000$ (100%)** | **$89.8\times$ (2,693 FPS)** | **$120.4\text{ s}$ (2m 00s)** |
| **C-MASP (`open_clip`)** | **$605\text{ MB}$ (GPU)** | **$1.5187$** | **$0.0000$ (0% broken!)** | **$0.7812$** | **$1.1128$ (3.86x better)** | **$1.0000$ (100%)** | **$66.0\times$ (1,979 FPS)** | **$153.6\text{ s}$ (2m 33s)** |

### Key Takeaways for Production
* **0.0% Broken Sentences ($\text{DBFI} = 0.0000$):** While fixed timers and shot detectors chop ~50% of sentences in half, C-MASP keeps every spoken sentence intact across all 2.9 hours of video.
* **4.45x Higher Overall Quality ($\text{Q}_{\text{chunk}} = 1.2817$):** Delivers clean, single-topic scenes with 100% legal chunk lengths ($\text{CSR} = 1.0000$).
* **Fast & Cheap at Scale:** Processes **nearly 3 hours of video in just 2 minutes on a standard CPU** with zero GPU or model weight costs.