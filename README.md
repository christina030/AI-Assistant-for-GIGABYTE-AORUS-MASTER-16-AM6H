# AI-Assistant-for-GIGABYTE-AORUS-MASTER-16-AM6H

## 🚀 啟動步驟

本專案使用 [uv](https://github.com/astral-sh/uv) 進行極速的 Python 套件管理。

### 1. 安裝環境與依賴套件
為了確保 `llama-cpp-python` 能正確使用 GPU (CUDA 12.x) 加速，請執行以下指令進行安裝：

```bash
# 安裝 uv
pip install uv

# 建立環境
uv sync
```
### 2. 下載量化模型 (GGUF)

下載 1.5B ~ 3B 級別的 GGUF 量化模型：

`mkdir -p models`

* Qwen2.5-3B-Instruct (Q4_K_M)
> 參數較大，但被量化到 4-bit。推論能力強，檔案約 2.15GB。

`wget https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf -O ./models/qwen_3b_q4.gguf`

* Qwen2.5-1.5B-Instruct (Q8_0)
> 參數砍半，但保留 8-bit 高精度權重。檔案約 1.7GB。

`wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf -O ./models/qwen_1.5b_q8.gguf`

* Llama-3.2-3B-Instruct (Q4_K_M)
> 作為跨架構對照組，測試不同模型家族在繁體中文規格問答上的表現。

`wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf -O ./models/llama_3b_q4.gguf`

### 3. 執行

`python main.py`

## 📊 結果分析

### 模型選擇（如何符合 4GB 限制）：

1. 檢索方法中的 Embedding 存在 CPU，節省 VRAM。
2. 選用 1.5B ~ 3B 的模型：

| 模型配置 | 檔案大小 | VRAM | 峰值 | 平均 TTFT | 平均 TPS | 表現 |
| Qwen2.5-3B-Instruct (Q4_K_M) | 1.96 GB | ~2.3 GB | 0.45s | 28.5 |  |  |
| Qwen2.5-1.5B-Instruct (Q8_0)  | 1.76 GB | ~2.0 GB | 0.32s | 41.2 |  |  |
| Llama-3.2-3B-Instruct (Q4_K_M)  | 1.88 GB | ~2.6 GB | 0.48s | 26.8 |  |  |







模型配置,檔案大小,VRAM 峰值,平均 TTFT,平均 TPS,幻覺抵抗力
Qwen-3B (Q4_K_M),2.15 GB,~2.5 GB,0.45s,28.5,極高 (基準)
Qwen-1.5B (Q8_0),1.70 GB,~2.0 GB,0.32s,41.2,中等
Llama-3.2-3B (Q4_K),2.20 GB,~2.6 GB,0.48s,26.8,繁中偶發破圖
