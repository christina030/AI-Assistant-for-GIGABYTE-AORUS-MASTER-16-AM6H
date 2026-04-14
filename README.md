# AI-Assistant-for-GIGABYTE-AORUS-MASTER-16-AM6H

## 🚀 啟動步驟

本專案使用 [uv](https://github.com/astral-sh/uv) 進行極速的 Python 套件管理。

### 1. 安裝環境與依賴套件

```bash
# 安裝 uv
pip install uv

# 建立環境
uv sync

# 若沒成功安裝 llama_cpp
uv pip install llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 \
  --system
```
### 2. 下載量化模型 (GGUF)

下載 1.5B ~ 3B 級別的 GGUF 量化模型：

```
mkdir -p models
```

* Qwen2.5-3B-Instruct (Q4_K_M)
> 參數較大，但被量化到 4-bit。推論能力強，檔案約 2GB。

```
wget https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf -O ./models/qwen_3b_q4.gguf
```

* Qwen2.5-1.5B-Instruct (Q8_0)
> 參數砍半，但保留 8-bit 高精度權重。檔案約 1.7GB。

```
wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q8_0.gguf -O ./models/qwen_1.5b_q8.gguf
```

* Llama-3.2-3B-Instruct (Q4_K_M)
> 作為跨架構對照組，測試不同模型家族在繁體中文規格問答上的表現。

```
wget https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf -O ./models/llama_3b_q4.gguf
```

### 3. 執行

```
python main.py
```

## 📊 結果分析

### 模型選擇（如何符合 4GB 限制）：

* 選用輕量級 Embedding 模型：選擇`BAAI/bge-small-zh-v1.5`作為文本向量化模型（體積小、載入速度快）。

* 選用 1.5B ~ 3B 的 GGUF 量化模型：降低模型參數在 VRAM 所佔用的空間。

### 評測結果分析：

使用 21 筆測試案例，包括 14 筆正常案例，以及 7 筆特殊案例（問題不相干、詢問網站中不包含的資訊、敘述模糊）。
```
test_cases = [
    ("這台筆電的 GPU 顯存是多少？", "16GB"),
    ("How is the color gamut performance of the display?", "DCIP-3"),
    ("I/O 接口總共有幾個 Type-C？", "Type-C"),
    ("這台支援的最大記憶體頻率是多少？", "DDR5-5600"),
    ("這台筆電適合拿來做專業攝影修圖嗎？", "Display"),
    ("Can I connect two 4K external monitors to it?", "I/O Port"),
    ("在全黑環境下打字方便嗎？", "Backlit"),
    ("有支援最新的 Wi-Fi 7 嗎？", "WIFI 7"),
    ("How big is the battery capacity? Can it last long?", "99Wh"),
    ("這台筆電多重？每天背上下班會不會太累？", "2.6 kg"),
    ("Does the webcam support Windows Hello face authentication?", "Windows Hello"),
    ("看電影的音效好嗎？有沒有杜比音效？", "Dolby Atmos"),
    ("處理器是用 Intel Ultra 還是 AMD 的？", "Intel"),
    ("它的 Storage 擴充性如何？支援幾條 NVMe？", "NVMe")
]

mix_test_cases = [
    ("ASUS ROG Strix 的散熱表現好嗎？", "拒絕回答"),
    ("請問這台有附贈 AORUS 的電競滑鼠嗎？", "資訊不足"),
    ("這台 AORUS MASTER 16 和一般的 MASTER 17 在螢幕上有什麼不同？", "資訊不足"),
    ("What is the official retail price of this AORUS laptop?", "資訊不足"),
    ("主機板是幾相供電的？", "資訊不足"),
    ("他可以擴充嗎？", "要求澄清"),
    ("Is it good?", "要求澄清")
]
```

#### 1. 定量指標：推論效能

* TTFT (首字延遲, Time To First Token)：各題目的首字延遲大多落在 0.02 秒至 0.15 秒之間。

* TPS (生成速度, Tokens Per Second)：生成速度可達 50 ~ 80 tokens/s，部分測試中甚至高達 90 ~ 100 tokens/s。

| 模型配置 | 檔案大小 | VRAM | 平均 TTFT (s) | 平均 TPS (tokens/s) | 表現 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Qwen2.5-3B-Instruct (Q4_K_M) | 1.96 GB | ~2.3 GB | 0.063 | 72.58 | 平衡佳：回答快速、簡潔，主要根據提供的資料回答，不傾向回答額外內容（包括不相干內容）；但有時仍會回答錯誤。 |
| Qwen2.5-1.5B-Instruct (Q8_0)  | 1.76 GB | ~2.0 GB | 0.036s | 97.10 | 速度快：反應時間最短，生成速度最快，適合需要即時回應的場景；但輸出內容有時較不完整、準確。 |
| Llama-3.2-3B-Instruct (Q4_K_M)  | 1.88 GB | ~2.4 GB | 0.060s | 63.76 | 較不穩定：會幫忙整理分析問題答案，在回答比較需要推理的問題時表現較佳；但有時會回答網站資訊以外的內容，且不一定正確，也會有重複回答的狀況；更傾向以繁體中文回答。 |

#### 2. 定性分析：RAG Pipeline 效果評估

比較三種檢索方法的 Top-2 命中率表現：

* 純向量檢索
> 計算語義相似度（Cosine Similarity）。

* 純關鍵字檢索
> 矩陣運算 BM25 分數。

* 混合檢索
> 結合前兩者的的結果，進行倒數排名融合（RRF）。

| 檢索方法 | Top-2 命中率 | 平均 TTFT (s) | 平均 TPS (tokens/s) | 表現 |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| Vector (向量檢索) | 71.4 % (10/14) | 0.051 | 77.17 | 能較好地捕捉隱含的語義關聯，因此命中率較高。 |
| BM25 (關鍵字檢索) | 57.1 % (8/14) | 0.041 | 78.25 | 字詞比對在缺乏精確關鍵字時容易出錯，導致命中率相對較低。 |
| Hybrid (混合檢索) | 57.1 % (8/14) | 0.034 | 77.56 | 混合檢索的並沒有在向量檢索的基礎上提升表現，可能是因為沒有做好權重平衡，未來可對此進一步調整、優化。 |
