# Intelligent Video Chunking (C-MASP)
### Cascaded Multi-Modal Adaptive Semantic Partitioning for Video-RAG & VLM Ingestion

[![PyTest](https://img.shields.io/badge/pytest-9%20passed%20(0.58s)-success.svg)](tests/test_chunker.py)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](requirements.txt)
[![Throughput CPU](https://img.shields.io/badge/CPU%20Throughput-89.8x%20Realtime%20(2693%20FPS)-brightgreen.svg)](#5-empirical-benchmark-results-videomme-sample-10)
[![Throughput GPU](https://img.shields.io/badge/GPU%20Throughput-66.0x%20Realtime%20(1979%20FPS)-brightgreen.svg)](#5-empirical-benchmark-results-videomme-sample-10)
[![Zero-Weight Mode](https://img.shields.io/badge/Fast--Spatial%20Weights-0%20MB%20(Zero--Param)-orange.svg)](#3-balancing-accuracy-latency-and-model-size)
[![ViT Mode](https://img.shields.io/badge/OpenCLIP%20ViT--B--32-605%20MB%20(FP16%20CUDA)-purple.svg)](#3-balancing-accuracy-latency-and-model-size)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Thought Process](https://img.shields.io/badge/Thought%20Process-2--Page%20Executive%20Summary-blueviolet.svg)](docs/THOUGHT_PROCESS.md)

An open-source, 100% training-free, production-grade video chunking pipeline built for **Multimodal RAG (Retrieval-Augmented Generation)**, **Vision-Language Model (VLM) ingestion**, **surveillance archival**, and **enterprise video search**.

> [!TIP]
> **Candidate Submission Summary**: Looking for the 2-page concise thought process explanation required for the submission? See [**docs/THOUGHT_PROCESS.md**](docs/THOUGHT_PROCESS.md) (or the PDF at [**docs/THOUGHT_PROCESS.pdf**](docs/THOUGHT_PROCESS.pdf)) for an intuitive walkthrough of the engineering decisions, trade-offs, and empirical results.

---

## 1. Problem Statement

Imagine a customer gives you a massive video library containing meetings, screen-shares, surveillance feeds, sports broadcasts, concerts, and gameplay. They want to ask text questions to search for specific events or actions across the entire library. Before indexing these videos into a vector database, you must slice (chunk) them into meaningful segments.

### Why Naive Approaches Fail at Scale
1. **Naive Fixed-Time Slicing (e.g., cutting every 30 seconds):**
   * **Breaks Sentences in Half:** A 30-second timer cuts blindly while a person is in the middle of speaking ($47.8\%$ of cuts chop an active sentence in half on VideoMME). This destroys the transcript meaning for text search.
   * **Blurs Unrelated Scenes Together:** One 30-second clip might contain 15 seconds of a presenter's face and 15 seconds of a code editor, creating a muddy, inaccurate vector embedding.
2. **Visual-Shot-Only Detection (e.g., PySceneDetect / Histogram Thresholding):**
   * **Explodes Database Costs on Fast Action:** In sports, concerts, or commercials, camera angles change every 1.5–3 seconds. Indexing every camera cut creates $10\times$ more chunks than necessary.
   * **Fails on Lectures & Surveillance:** A 30-minute lecture or static security camera has almost zero hard camera cuts, producing giant runaway chunks that overflow VLM token limits.
   * **Still Chops Speech:** Editors frequently switch camera angles while a narrator is mid-sentence ($53.7\%$ sentence fracture rate on VideoMME).
3. **Dense Frame-by-Frame Neural Models:**
   * Running heavy VLMs or 3D-CNNs across all 108,000 frames of a 1-hour 30fps video introduces superlinear compute costs and severe latency bottlenecks.

### How C-MASP Solves This (In Simple Terms)
**C-MASP (Cascaded Multi-Modal Adaptive Semantic Partitioning)** treats video chunking as a multi-signal optimization problem that runs in **strictly linear $\mathcal{O}(T)$ time**:
1. **Look Sparsely (Tier 0):** We sample the video at **2.0 FPS** in a single pass—reducing frame decoding work by $15\times$ while preserving 0.5-second temporal precision.
2. **Extract 3 Lightweight Signals (Tiers 1–3):**
   * **Visual Signal:** Detects sharp camera cuts, slide/UI text changes (via HSV color + Sobel edge flux), and flags empty static surveillance segments (`STATIC_IDLE`).
   * **Speech/Discourse Signal:** Parses subtitles (automatically fixing overlapping YouTube 2-line rolling captions and ignoring `[Music]`/`[Applause]` tags) or falls back to raw audio Voice Activity Detection (RMS energy envelope) to find natural pauses between sentences.
   * **Semantic Signal:** Measures scene meaning using either a **0 MB training-free Spatial-Lab-DCT descriptor** on CPU or **605 MB OpenCLIP ViT-B-32** on GPU, smoothed with Foote's Checkerboard Novelty kernel.
3. **Find the Globally Optimal Cuts (Tier 4):** A **Constrained Viterbi Dynamic Programming Solver** finds the exact cut points that keep visual scenes pure, keep chunk lengths strictly inside $[10.0\text{s}, 45.0\text{s}]$, and **snap boundaries to millisecond-exact speech pauses** so no speaker is ever cut off mid-sentence ($\text{DBFI} = 0.0000$).

---

## 2. Evaluation Criteria: Standard Metrics vs. Novel Statistical Metrics

Because real-world customer video corpora do not come with manual ground-truth "chunk boundary" labels, evaluating a chunker requires both **standard system criteria** and **unsupervised statistical & information-theoretic metrics**.

### A. Standard Operational Criteria
| Metric | Definition | Production Target | Why It Matters |
| :--- | :--- | :---: | :--- |
| **$\text{CSR} \uparrow$**<br>*(Constraint Satisfaction Rate)* | Fraction of chunks strictly obeying duration bounds $[L_{\min}, L_{\max}] = [10\text{s}, 45\text{s}]$. | **$1.0000$ (100%)** | Prevents micro-chunks (which bloat vector DB storage) and oversized chunks (which overflow VLM context windows). |
| **Throughput ($\times\text{RT}$ / FPS) $\uparrow$** | Ratio of video duration to total wall-clock processing time. | **$>50\times$ Realtime** | Ensures multi-thousand-hour archives can be ingested cheaply without superlinear compute costs. |
| **Model Size & VRAM $\downarrow$** | Disk weight size and peak GPU memory footprint. | **$0\text{ MB}$ – $605\text{ MB}$** | Determines whether the pipeline can run on serverless CPU workers or requires GPU nodes. |

### B. Novel Statistical & Information-Theoretic Metrics *(Custom Suite)*
To evaluate chunk quality without human labels, we formulated four statistical metrics grounded in directional statistics (von Mises-Fisher spherical dispersion), rate-distortion theory, and linguistic continuity:

| Metric | Plain-English Meaning | Mathematical Formulation | Why It Matters in Production |
| :--- | :--- | :--- | :--- |
| **$\text{SCR} \uparrow$**<br>*(Semantic Cohesion Ratio)* | **One chunk = One clean scene.**<br>Frames *inside* a chunk should look similar (low internal spread), while *neighboring* chunks should look different (high separation). | $$\text{SCR} = \frac{\frac{1}{K-1} \sum_{k=1}^{K-1} d_{\cos}(\boldsymbol{\mu}_k, \boldsymbol{\mu}_{k+1})}{\frac{1}{K} \sum_{k=1}^K \left( \frac{1}{\vert{}C_k\vert{}} \sum_{t \in C_k} d_{\cos}(\mathbf{e}_t, \boldsymbol{\mu}_k) \right) + \epsilon}$$ | Analogous to the **Calinski-Harabasz / Fisher Discriminant Ratio** on the unit hypersphere $\mathbb{S}^{D-1}$. High $\text{SCR}$ proves each chunk maps to a sharp, unblurred vector embedding. |
| **$\text{DBFI} \downarrow$**<br>*(Discourse Boundary Fracture Index)* | **Never cut people off mid-sentence.**<br>The fraction of internal chunk boundaries that slice through an active spoken utterance. | $$\text{DBFI} = \frac{1}{K-1} \sum_{k=1}^{K-1} \mathbb{I}\Big(\exists m : u_m^{\text{start}} + \delta < \tau_k < u_m^{\text{end}} - \delta\Big)$$ | **Lower is better ($0.0000 = 0\%$ broken sentences).** Chopping a sentence in half splits context across two chunks and ruins RAG text retrieval. |
| **$\text{IPU} \uparrow$**<br>*(Information Path Uniformity)* | **Every chunk carries a balanced amount of action.**<br>Measures whether semantic trajectory variation $\mathcal{V}_k = \sum_{t \in C_k} \Vert{}\mathbf{e}_t - \mathbf{e}_{t-1}\Vert{}_2$ is evenly distributed across chunks. | $$\text{IPU} = \frac{1}{1 + \text{CV}(\mathcal{V})}, \quad \text{CV}(\mathcal{V}) = \frac{\sigma(\mathcal{V})}{\bar{\mathcal{V}} + \epsilon}$$ | Rooted in **Shannon Rate-Distortion Equipartition**. High $\text{IPU} \in (0, 1]$ ensures no chunk is information-starved or chaotic. |
| **$\text{Q}_{\text{chunk}} \uparrow$**<br>*(Composite Quality Score)* | **The overall "North Star" grade.**<br>Combines scene cohesion ($\text{SCR}$), intact speech ($1 - \text{DBFI}$), and uniform information load ($\text{IPU}$). | $$\text{Q}_{\text{chunk}} = \text{SCR} \times (1 - \text{DBFI}) \times \text{IPU}$$ | Prevents single-metric gaming: `Visual_Cut_Only` gets high $\text{SCR}$ by cutting on every camera switch, but gets penalized because it fractures $53.7\%$ of spoken sentences. |

---

## 3. Balancing Accuracy, Latency, and Model Size

A core design goal of C-MASP is giving engineers explicit control over the **Accuracy vs. Latency vs. Model Size Pareto Frontier** via two pluggable semantic encoders (`--encoder fast_spatial` vs. `--encoder open_clip`):

```
                                  COMPOSITE QUALITY (Q_chunk)
                                               ▲
                                               │
                              C-MASP (fast_spatial CPU) ◄─── PRODUCTION SWEET SPOT
                                 [Q = 1.2817 | 89.8x RT]     0 MB Weights, 0 MB VRAM,
                                               │             Zero Sentence Fractures!
                        C-MASP (OpenCLIP ViT-B-32 GPU)
                           [Q = 1.1128 | 66.0x RT]
                                               │
                                               │
                  Visual-Cut-Only              │
                    [Q = 0.4548]               │
                                               │
           Fixed-30s                           │
          [Q = 0.2881]                         │
  ─────────────┼───────────────────────────────┼────────────────────────► SPEED / THROUGHPUT
              10x                             90x                        (Realtime Factor)
```

### Head-to-Head Trade-Off Matrix (Evaluated on 2.9 Hours of VideoMME)

| Dimension / Metric | Baseline 1: `Fixed_30s` | Baseline 2: `Visual_Cut_Only` | **C-MASP Mode 1: `fast_spatial` (CPU)** | **C-MASP Mode 2: `open_clip` (GPU)** |
| :--- | :---: | :---: | :---: | :---: |
| **Semantic Backbone** | None (Clock Timer) | HSV + Sobel Z-Score | **192-D Spatial-Lab + 2D-AC-DCT Pyramid** | **512-D OpenCLIP `ViT-B-32` (FP16)** |
| **Model Weights Size** | $0\text{ MB}$ | $0\text{ MB}$ | **$0\text{ MB}$ (100% Training-Free Math)** | **$605\text{ MB}$ (`safetensors`)** |
| **Hardware & VRAM** | CPU ($0\text{ MB}$) | CPU ($0\text{ MB}$) | **Single CPU ($0\text{ MB}$ VRAM)** | **NVIDIA RTX 4070 ($\sim 591\text{ MB}$ VRAM)** |
| **Decoded Frame Size** | — | $64 \times 64$ | **$64 \times 64$** | **$224 \times 224$ (Bicubic Native)** |
| **Throughput ($\times\text{Realtime}$)** | $>300\times$ | $150\times$ | **$89.8\times\text{ Realtime}$** | **$66.0\times\text{ Realtime}$** |
| **Effective FPS (@30fps)** | $>9,000\text{ FPS}$ | $4,500\text{ FPS}$ | **$2,693\text{ FPS}$** | **$1,979\text{ FPS}$** |
| **Wall Clock (2.9h Video)** | $<35\text{ s}$ | $\approx 70\text{ s}$ | **$120.4\text{ seconds}$ (2m 00s)** | **$153.6\text{ seconds}$ (2m 33s)** |
| **Sentence Fracture ($\text{DBFI} \downarrow$)** | $0.4778$ *(47.8% broken)* | $0.5374$ *(53.7% broken)* | **$0.0000$ (0.0% broken!)** | **$0.0000$ (0.0% broken!)** |
| **Bounds Compliance ($\text{CSR} \uparrow$)** | $0.9254$ | $0.9227$ | **$1.0000$ (100% compliant)** | **$1.0000$ (100% compliant)** |
| **Path Uniformity ($\text{IPU} \uparrow$)** | $0.7103$ | $0.6716$ | **$0.7568$** | **$0.7812$ (Highest)** |
| **Composite Quality ($\text{Q}_{\text{chunk}} \uparrow$)** | $0.2881$ | $0.4548$ | **$1.2817$ ($4.45\times$ vs. Fixed)** | **$1.1128$ ($3.86\times$ vs. Fixed)** |

### Why Does `fast_spatial` (0 MB) Score Higher on $\text{SCR}$ While `open_clip` (605 MB) Scores Higher on $\text{IPU}$?
1. **When to Deploy `fast_spatial` (Default — Serverless CPU Ingestion):**
   * Our 192-D descriptor combines $4 \times 4$ CIE-Lab spatial layout, regional texture variance, **zero-DC 2D-DCT structural frequency coefficients**, and HSV histograms. Because it is sensitive to structural layout and slide/UI geometry, it achieves higher intra-scene contrast ($\text{SCR} = 1.8186$) at **$0\text{ MB}$ model size** and **$89.8\times$ real-time speed on any CPU**.
2. **When to Deploy `open_clip` (`ViT-B-32` — GPU Semantic Ingestion):**
   * Pre-trained `ViT-B-32` maps different camera angles of the *same semantic concept* (e.g., two different angles of a basketball court or concert stage) closer together on the hypersphere. This yields smoother semantic trajectories and the highest **Information Path Uniformity ($\text{IPU} = 0.7812$)** across cinematic footage, while still running at **$66.0\times$ real-time** via FP16 CUDA batching.

---

## 4. System Architecture: The 4-Tier Cascade

```
               Raw Video Stream (.mp4) + Audio / Subtitles (.srt)
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ TIER 0: Single-Pass Sparse Hardware-Friendly Decoder                            │
│ • Sparse 2.0 FPS grab/retrieve decoding (64x64 for CPU, 224x224 for ViT GPU)    │
│ • Eliminates 93.3% of redundant frame decodes at 30fps native                   │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
┌─────────────────────────┐ ┌─────────────────────────┐ ┌─────────────────────────┐
│ TIER 1: Visual          │ │ TIER 2: Discourse &     │ │ TIER 3: Semantic Shifts │
│ Kinematics & Gate       │ │ Speech Alignment        │ │ & Novelty Kernel        │
│ • HSV Hist Cosine Flux  │ │ • YouTube Rolling SRT   │ │ • Fast Spatial Pyramid  │
│ • Sobel Edge Ratio Flux │ │   De-overlap & Dedup    │ │   (192-D Lab+AC-DCT)    │
│ • CFAR Median/MAD       │ │ • [Music]/[Applause]    │ │   OR OpenCLIP ViT-B-32  │
│   Sliding Z-Score + NMS │ │   non-speech filtering  │ │ • Foote's Gaussian      │
│ • Static Camera Gate    │ │ • Audio RMS VAD Fallback│ │   Checkerboard Kernel   │
└───────────┬─────────────┘ └───────────┬─────────────┘ └───────────┬─────────────┘
            │                           │                           │
            └───────────────────────────┼───────────────────────────┘
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ TIER 4: O(T) Constrained Viterbi Dynamic Programming Solver + Pause Snapping    │
│ • O(1) Prefix-Sum Spherical Variance (Ward's Duration-Weighted Formulation)     │
│ • O(1) Predecessor + Lookahead Centroid Contrast & Path Equipartition           │
│ • Internal Snapping Cushion (+-1.25s) + 2-Pass Sub-Frame Speech Pause Snapping  │
└───────────────────────────────────────┬─────────────────────────────────────────┘
                                        ▼
                   Production JSON Chunks + Quality Telemetry
```

### Key Engineering Innovations Inside the 4 Tiers

1. **Tier 1 — CFAR Median/MAD Visual Cut Detection & Static Surveillance Gate (`visual.py`):**
   * Combines HSV color histogram cosine distance ($0.65$) with Sobel gradient edge change ratio ($0.35$) so both cinematic hard cuts and presentation slide text transitions are captured.
   * Applies a **Constant False Alarm Rate (CFAR) Sliding Median + MAD (Median Absolute Deviation)** filter with 3-tap Non-Maximum Suppression (NMS) so continuous motion in sports does not trigger false cut spikes.
   * **Static Dead-Zone Gate:** Automatically flags stationary segments ($\ge 15\text{s}$ with flux $< 0.015$ and no speech) as `STATIC_IDLE` so downstream pipelines can skip expensive VLM indexing on empty surveillance feeds.

2. **Tier 2 — Rolling-Caption De-Overlapping & Zero-Training Audio VAD Fallback (`discourse.py`):**
   * **Solving the YouTube Two-Line Rolling-Caption Bug:** Auto-generated YouTube `.srt` files display 2 rolling lines at a time where Cue $k+1$ starts $2\text{–}4\text{ seconds}$ *before* Cue $k$ ends. A naive parser sees continuous overlapping intervals across the entire video with zero silence gaps! Our parser strips non-speech closed-caption tags (`[Music]`, `[Applause]`), deduplicates repeated rolling phrases, and clamps $u_k^{\text{end}} = \min(u_k^{\text{end}}, u_{k+1}^{\text{start}})$, guaranteeing clean, disjoint utterance intervals.
   * **Short-Time Audio RMS VAD Fallback:** When subtitles are absent, C-MASP extracts 16kHz mono audio via `ffmpeg`/`scipy` and runs a training-free Short-Time RMS Energy Envelope detector with morphological silence bridging.

3. **Tier 3 — Block-Balanced Embeddings & Foote's Checkerboard Kernel (`semantic.py`):**
   * **Removing the 2D-DCT DC Spike:** In standard 2D-DCT, the $(0,0)$ DC brightness coefficient is scaled by $\sqrt{H \cdot W} = 32\times$, dominating $>90\%$ of the L2 norm. We explicitly zero out `dct[0, 0]` and independently L2-normalize all 4 feature blocks (CIE-Lab layout, spatial std, AC-DCT geometry, HSV distribution) so cosine distance reflects true structural change.
   * **Foote's Gaussian-Tapered Checkerboard Kernel:** Convolves a $[-1, +1; +1, -1]$ Gaussian checkerboard kernel along the diagonal of the local self-similarity matrix to detect sustained semantic regime shifts while ignoring single-frame camera flashes.

4. **Tier 4 — $\mathcal{O}(T)$ Constrained Viterbi Solver & Sub-Frame Snapping (`dp_solver.py`, `pipeline.py`):**
   * **Half-Open Intervals $[i, j)$ with $\mathcal{O}(1)$ Prefix Sums:** When a scene cut happens at frame $j$, frame $j$ is the first frame of the *new* scene. Evaluating chunks on half-open intervals $[i, j)$ prevents boundary frame bleed. Using cumulative embedding tables $\mathbf{P}_t = \sum_{u=0}^{t-1} \mathbf{e}_u$, the spherical variance of any candidate chunk $[i, j)$ is evaluated in **strictly $\mathcal{O}(1)$ time**:
     $$D_{\text{intra}}(i, j) = 1 - \frac{\Vert{}\mathbf{P}_j - \mathbf{P}_i\Vert{}_2}{j - i}$$
   * **V3 Global Objective Function:**
     $$\mathcal{J}(i \to j) = w_{\text{var}} \left(\frac{\Delta \tau}{L_{\text{target}}}\right) D_{\text{intra}}(i, j) - w_{\text{inter}} D_{\text{inter}}(i, j) + w_{\text{ipu}} \left(\frac{\mathcal{V}_{i,j} - \mathcal{V}_{\text{target}}}{\mathcal{V}_{\text{target}}}\right)^2 + w_{\text{len}} \left(\frac{\Delta \tau - L_{\text{target}}}{L_{\text{target}}}\right)^2 + \Phi_{\text{cut}}(j)$$
   * **Two-Pass Sub-Frame Linguistic Snapping:** After the DP solver finds optimal coarse boundaries on the 2 FPS grid (with a $\pm 1.25\text{s}$ internal duration cushion), a bidirectional continuous refinement pass snaps any boundary near speech to the exact millisecond silence gap $[u_m^{\text{end}}, u_{m+1}^{\text{start}}]$.

---

## 5. Empirical Benchmark Results (VideoMME Sample-10)

We benchmarked C-MASP on the 10 representative videos in `aintropy-ai/video-mme-sample10` totaling **10,389 seconds ($\approx 2.9\text{ hours}$)** of diverse real-world footage (short 64s clips to 59-minute broadcasts across sports, lectures, commercials, and movies).

### Benchmark 1: CPU Zero-Weight Mode (`--encoder fast_spatial --device cpu`)
```text
========================================================================================
PHASE 2 COMPLETE: SCIENTIFIC BENCHMARK & ABLATION RESULTS (fast_spatial on CPU, 0 MB)
========================================================================================
Method                 | SCR (higher) | DBFI (lower) | IPU (higher) | Q_chunk (higher)  | CSR (higher)
----------------------------------------------------------------------------------------
Fixed_30s              | 0.8232       | 0.4778       | 0.7103       | 0.2881            | 0.9254      
Visual_Cut_Only        | 2.0461       | 0.5374       | 0.6716       | 0.4548            | 0.9227      
C_MASP_Ours            | 1.8186       | 0.0000       | 0.7568       | 1.2817            | 1.0000      
========================================================================================
Throughput Speed:   89.8x Realtime (2693 effective FPS at 30fps native)
Total Wall Clock:   120.4 seconds (for 2.9 hours of video on a single CPU!)
```

### Benchmark 2: GPU Foundation Model Mode (`--encoder open_clip --device cuda`)
```text
========================================================================================
PHASE 2 COMPLETE: SCIENTIFIC BENCHMARK & ABLATION RESULTS (OpenCLIP ViT-B-32 on RTX 4070)
========================================================================================
Method                 | SCR (higher) | DBFI (lower) | IPU (higher) | Q_chunk (higher)  | CSR (higher)
----------------------------------------------------------------------------------------
Fixed_30s              | 0.7906       | 0.4778       | 0.7633       | 0.2881            | 0.9254      
Visual_Cut_Only        | 1.8831       | 0.5365       | 0.7099       | 0.4301            | 0.9227      
C_MASP_Ours            | 1.5187       | 0.0000       | 0.7812       | 1.1128            | 1.0000      
========================================================================================
Throughput Speed:   66.0x Realtime (1979 effective FPS at 30fps native)
Total Wall Clock:   153.6 seconds (for 2.9 hours of video on RTX 4070 Laptop GPU)
```


## 6. Repository Structure

```text
problem-1-4-intelligent-video-chunking/
├── run.yaml                         # Automated setup, run, and test specifications for grader bot
├── requirements.txt                 # Open-source dependencies (OpenCV, NumPy, SciPy, PyTest, OpenCLIP)
├── README.md                        # Master architectural documentation, trade-offs & benchmark report
├── .gitignore                       # Clean repo rules (ignores __pycache__, .pytest_cache, data/)
├── docs/                            # Candidate write-ups, slides, and evaluation notes
│   ├── README.md                    # Write-up summary and navigation
│   ├── THOUGHT_PROCESS.md           # 2-page intuitive explanation of engineering thought process
│   └── THOUGHT_PROCESS.pdf          # PDF formatted 2-page thought process summary
├── src/                             # ALL implementation code, scripts, and modules
│   ├── main.py                      # Default entrypoint for evaluator: python src/main.py --help
│   ├── chunk_video.py               # Production CLI for chunking single videos or directories
│   ├── chunk_and_benchmark.py       # Unified one-shot runner: production chunking + evaluation
│   ├── benchmark_videomme.py        # Automated 3-way ablation benchmark runner on VideoMME
│   ├── download_videomme.py         # Script to download VideoMME sample-10 from Hugging Face
│   └── chunker/                     # Core C-MASP 4-tier package
│       ├── __init__.py
│       ├── config.py                # Typed dataclasses: ChunkerConfig, VideoChunk, ChunkingResult
│       ├── pipeline.py              # IntelligentVideoChunker orchestrator & baseline comparators
│       ├── signals/                 # Tier 1 (visual), Tier 2 (discourse), Tier 3 (semantic)
│       ├── optimizer/               # Tier 4 (O(T) constrained Viterbi DP solver)
│       └── evaluation/              # Statistical metrics (SCR, DBFI, IPU, Q_chunk, CSR, xRT)
└── tests/                           # Deterministic unit test suite
    ├── __init__.py
    └── test_chunker.py              # 9 automated unit tests (passing in <1s)
```

---

## 7. Quickstart & Installation

```bash
# Clone your forked repository
git clone [https://github.com/](https://github.com/)<your-username>/intelligent-video-chunking.git
cd intelligent-video-chunking

# Create or activate Python 3.10+ environment
conda create -n myenv python=3.10 -y
conda activate myenv

# Install open-source dependencies
pip install -r requirements.txt
```

---

## 8. Commands to Run the Code

### 1. Primary Entrypoint (`src/main.py`)
Executed automatically by the AIntropy evaluator (`run.yaml`). Can be invoked directly from the project root:
```bash
# Print clean CLI usage & flags
python src/main.py --help

# Chunk a single video using the 0 MB CPU engine
python src/main.py --input path/to/video.mp4 --encoder fast_spatial --device cpu --output chunks.json

# Chunk a single video using OpenCLIP ViT on GPU
python src/main.py --input path/to/video.mp4 --encoder open_clip --device cuda --output chunks_vit.json

# Run full VideoMME benchmark suite
python src/main.py --benchmark --data-dir ./data --output benchmark_report.json
```

### 2. Unified Production Chunking + Benchmark Evaluation (`src/chunk_and_benchmark.py`)
Runs Phase 1 (saves production chunks to JSON) and Phase 2 (runs `Fixed_30s` and `Visual_Cut_Only` baselines and prints the full evaluation table):

```bash
# Mode 1: Fast-Spatial CPU Engine (0 MB weights, 89.8x Realtime)
python src/chunk_and_benchmark.py \
  --data-dir ../data \
  --encoder fast_spatial \
  --device cpu \
  --output-chunks production_chunks.json \
  --output-benchmark benchmark_report.json

# Mode 2: OpenCLIP ViT-B-32 GPU Engine (605 MB weights, 66.0x Realtime on CUDA)
python src/chunk_and_benchmark.py \
  --data-dir ../data \
  --encoder open_clip \
  --device cuda \
  --output-chunks production_chunks_vit.json \
  --output-benchmark benchmark_report_vit.json
```

### 3. Standalone VideoMME Benchmark Script (`src/benchmark_videomme.py`)
*(Automatically downloads `aintropy-ai/video-mme-sample10` from Hugging Face if `--data-dir` is empty)*
```bash
python src/benchmark_videomme.py --data-dir ../data --encoder fast_spatial --device cpu
```

---

## 9. Unit Testing & Verification

Run the deterministic `pytest` suite (generates synthetic multi-scene videos, overlapping YouTube SRTs, and synthetic audio waveforms on the fly):

```bash
pytest tests/test_chunker.py -v
```

### All 9 Automated Tests Included:
1. `test_cmasp_outperforms_fixed_window_on_synthetic_stream`: Verifies C-MASP recovers ground-truth scene cuts at $20\text{s}$ and $42\text{s}$ with $\text{DBFI} = 0.0000$ while outperforming `Fixed_30s` on $\text{SCR}$ and $\text{Q}_{\text{chunk}}$.
2. `test_static_gate_surveillance_detection`: Confirms stationary surveillance feeds are classified as `SURVEILLANCE_STATIC_SPARSE` and collapsed into `STATIC_IDLE` chunks.
3. `test_discourse_srt_parser_edge_cases`: Tests resilience against missing files, empty SRTs, and unpunctuated clauses.
4. `test_youtube_rolling_caption_deoverlap_and_music_filter`: Verifies `[Music]`/`[Applause]` filtering, rolling 2-line timestamp clamping, and duplicate phrase removal.
5. `test_subframe_discourse_boundary_snapping`: Confirms mid-sentence coarse cuts snap into exterior silence gaps ($19.2\text{s}\text{–}19.6\text{s}$) with zero fracture.
6. `test_evaluator_metrics_edge_cases`: Ensures graceful handling of empty chunk lists and zero-length clips.
7. `test_dp_solver_prefix_variance_numerical_equivalence`: Proves $\mathcal{O}(1)$ prefix-sum spherical variance matches $\mathcal{O}(N)$ calculation to $< 10^{-5}$.
8. `test_audio_rms_vad_synthetic_signal`: Validates zero-training Short-Time Audio RMS VAD on synthetic 16kHz speech bursts.
9. `test_semantic_extractor_properties`: Confirms the 192-D `fast_spatial` descriptor produces unit-norm vectors even on pure black or pure white frames.

---

## 10. License

MIT License. Open-source and free for commercial and academic use.
