import time
import numpy as np
import math
import re
from collections import Counter
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
from llama_cpp import Llama

"""# 1. 資料解析"""

class SpecParser:
    def __init__(self, url="https://www.gigabyte.com/tw/Laptop/AORUS-MASTER-16-AM6H/sp"):
        self.url = url
        self.chunks = []

    def get_chunks(self):
        if self.chunks:
            return self.chunks

        print(f"[SpecParser] 正在嘗試從 {self.url} 擷取網頁資料...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        try:
            response = requests.get(self.url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            # 左側：屬性名 | 右側：內容值
            spec_names = soup.select('.spec-name, .spec-title, .item-name, th')
            spec_values = soup.select('.spec-list, .spec-content, .item-content, td')

            if spec_names and spec_values and len(spec_names) == len(spec_values):
                for name, value in zip(spec_names, spec_values):
                    key = name.get_text(strip=True)
                    # 將 <br> 標籤或其他 HTML 區塊轉換為換行，保持排版與語意
                    val = value.get_text(separator="\n", strip=True)

                    if key and val:
                        self.chunks.append({
                            "id": key,
                            "text": f"[{key}]\n{val}"
                        })

            # 針對可能出現的標準 Table 格式作為備用解析
            if not self.chunks:
                for table in soup.find_all('table'):
                    for row in table.find_all('tr'):
                        cols = row.find_all(['th', 'td'])
                        if len(cols) == 2:
                            k = cols[0].get_text(strip=True)
                            v = cols[1].get_text(separator="\n", strip=True)
                            if k and v:
                                self.chunks.append({"id": k, "text": f"[{k}]\n{v}"})

        except Exception as e:
            print(f"[SpecParser] 網路請求或解析失敗: {e}")

        # 靜態備用資料
        if not self.chunks:
            print("[SpecParser] 觸發備用機制：載入 GIGABYTE AORUS MASTER 16 AM6H 靜態規格快取...")
            self.chunks = self._get_fallback_data()
        else:
            print(f"[SpecParser] 成功解析 {len(self.chunks)} 筆規格資料！")

        return self.chunks

    def _get_fallback_data(self):
        return [
            {"id": "OS", "text": "[OS]\nWindows 11 Pro\nWindows 11 Home\nUEFI Shell OS"},
            {"id": "CPU", "text": "[CPU]\nIntel® Core™ Ultra 9 Processor 275HX (36MB cache, up to 5.4 GHz, 24 cores, 24 threads)"},
            {"id": "Video Graphics", "text": "[Video Graphics]\nNVIDIA® GeForce RTX™ 5090 Laptop GPU 24GB GDDR7 / NVIDIA® GeForce RTX™ 5080 Laptop GPU 16GB GDDR7 / NVIDIA® GeForce RTX™ 5070 Ti Laptop GPU 12GB GDDR7\n175W Maximum Graphics Power with Dynamic Boost*AI Boost : 1902 MHz (1702 MHz Boost Clock + 200 MHz OC)*"},
            {"id": "Display", "text": "[Display]\n16.0\" 16:10\nOLED WQXGA (2560×1600) 240Hz, 1ms, DCIP-3 100%, 500nits (peak), 1,000,000:1\nSupports NVIDIA Advanced Optimus\nNVIDIA® G-SYNC\nVESA DisplayHDR True Black 500\nVESA ClearMR 10000\nPantone® Validated\nTÜV Rheinland Low Blue Light\nDolby Vision"},
            {"id": "System Memory", "text": "[System Memory]\nUp to 64GB DDR5 5600MHz\n2x SO-DIMM sockets for expansion"},
            {"id": "Storage", "text": "[Storage]\n1x PCIe Gen5 M.2 slot\n1x PCIe Gen4x4 M.2 slots\nUp to 4TB PCIe NVMe™ M.2 SSD"},
            {"id": "Keyboard Type", "text": "[Keyboard Type]\n3-zone RGB Backlit Keyboard, Up to 1.7mm Key-travel (Support N-Key)"},
            {"id": "I/O Port", "text": "[I/O Port]\nLeft Side:\n1 x DC in\n1 x RJ-45\n1 x HDMI 2.1\n1 x Type-A support USB3.2 Gen2\n1 x Type-C with Thunderbolt™5 (support USB4, DisplayPort™ 2.1 and Power Delivery 3.0)\nRight Side:\n1 x Type-A support USB3.2 Gen2\n1 x Type-C with Thunderbolt™4 (support USB4, DisplayPort™ 1.4 and Power Delivery 3.0)\n1 x MicroSD (UHS-II)\n1 x Audio Jack support mic / headphone combo"},
            {"id": "Audio", "text": "[Audio]\n4x 2W speakers\nMicrophone\nDolby Atmos®\nSmart Amp Technology"},
            {"id": "Communications", "text": "[Communications]\nWIFI 7 (802.11be 2x2)\nLAN: 1G\nBluetooth v5.4"},
            {"id": "Webcam", "text": "[Webcam]\nFHD (1080p) IR Webcam\nBuild-in array Microphone\nSupport Windows Hello Face Authentication"},
            {"id": "Safety Device", "text": "[Safety Device]\nFirmware-based TPM, supports Intel® Platform Trust Technology (Intel® PTT)"},
            {"id": "Battery", "text": "[Battery]\nLi Polymer 99Wh"},
            {"id": "Adapter", "text": "[Adapter]\n330W AC Adapter"},
            {"id": "Size", "text": "[Size]\n357 x 254 x 23~29.9 mm"},
            {"id": "Weight", "text": "[Weight]\n~2.5 kg"},
            {"id": "Color", "text": "[Color]\nDark Tide"}
        ]

parser = SpecParser()
chunks = parser.get_chunks()
for ch in chunks:
    print(ch)

"""# 2. RAG - 檢索

*   純向量（PureVectorIndex）
*   純關鍵字（BM25Retriever）
*   混合檢索（HybridRetriever）

"""

class PureVectorIndex:
    def __init__(self, model_name="BAAI/bge-small-zh-v1.5"):
        # Embedding 放 CPU，節省 VRAM
        self.embedder = SentenceTransformer(model_name, device="cpu")
        self.documents = []
        self.embeddings = None

    def add_documents(self, chunks):
        self.documents = chunks
        texts = [chunk["text"] for chunk in chunks]
        self.embeddings = self.embedder.encode(texts, normalize_embeddings=True)

    def search(self, query, top_k=2):
        query_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        similarities = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [{"score": similarities[idx], "document": self.documents[idx]["text"]} for idx in top_indices]

class BM25Retriever:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.vocab = {}
        self.idf = {}
        self.avgdl = 0
        self.doc_len = []
        self.term_doc_matrix = None

    def _tokenize(self, text):
        # 英文：單字 | 中文：字元
        text = text.lower()
        tokens = re.findall(r'[a-z0-9]+|[\u4e00-\u9fa5]', text)
        return tokens

    def add_documents(self, chunks):
        self.documents = chunks
        corpus_tokens = [self._tokenize(chunk["text"]) for chunk in chunks]

        self.doc_len = np.array([len(tokens) for tokens in corpus_tokens])
        self.avgdl = np.mean(self.doc_len)
        N = len(corpus_tokens)

        df = Counter()
        for tokens in corpus_tokens:
            df.update(set(tokens))

        for idx, (term, freq) in enumerate(df.items()):
            self.vocab[term] = idx
            # IDF 計算
            self.idf[term] = math.log(1 + (N - freq + 0.5) / (freq + 0.5))

        # Term-Document 加速運算
        vocab_size = len(self.vocab)
        self.term_doc_matrix = np.zeros((vocab_size, N))

        for doc_id, tokens in enumerate(corpus_tokens):
            term_counts = Counter(tokens)
            for term, count in term_counts.items():
                if term in self.vocab:
                    term_id = self.vocab[term]
                    self.term_doc_matrix[term_id, doc_id] = count

    def search(self, query, top_k=2):
        tokens = self._tokenize(query)
        scores = np.zeros(len(self.documents))

        for term in tokens:
            if term not in self.vocab:
                continue
            term_id = self.vocab[term]
            # 出現頻率向量
            f_q_D = self.term_doc_matrix[term_id]

            # BM25 分數
            numerator = f_q_D * (self.k1 + 1)
            denominator = f_q_D + self.k1 * (1 - self.b + self.b * (self.doc_len / self.avgdl))
            scores += self.idf[term] * (numerator / denominator)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"id": self.documents[idx]["id"], "score": scores[idx], "document": self.documents[idx]["text"]}
                for idx in top_indices]

class HybridRetriever:
    def __init__(self, vector_index, bm25_index, rrf_k=60):
        self.vector_index = vector_index
        self.bm25_index = bm25_index
        self.rrf_k = rrf_k

    def search(self, query, top_k=2):
        # 前兩者檢索結果
        vec_results = self.vector_index.search(query, top_k=5)
        bm25_results = self.bm25_index.search(query, top_k=5)

        rrf_scores = {}

        # Vector 排行
        for rank, item in enumerate(vec_results):
            doc_id = item.get("id", item["document"][:10]) # 若無 id 暫以文本前綴代替
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"score": 0, "document": item["document"]}
            rrf_scores[doc_id]["score"] += 1.0 / (self.rrf_k + rank + 1)

        # BM25 排行
        for rank, item in enumerate(bm25_results):
            doc_id = item["id"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"score": 0, "document": item["document"]}
            rrf_scores[doc_id]["score"] += 1.0 / (self.rrf_k + rank + 1)

        # RRF 分數 -> 排序
        sorted_results = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k]

"""# 3. RAG - 生成"""

class AorusAssistant:
    def __init__(self, chunks, model_path):
        print("[System] 初始化檢索系統與 Embedding (CPU)...")

        # Init 檢索器
        print("[Init] 建立 Vector Index...")
        self.vec_index = PureVectorIndex()
        self.vec_index.add_documents(chunks)
        for i, doc in enumerate(self.vec_index.documents):
            doc["id"] = chunks[i]["id"]

        print("[Init] 建立 BM25 Index...")
        self.bm25_index = BM25Retriever()
        self.bm25_index.add_documents(chunks)

        print("[Init] 建立 Hybrid Index...")
        self.hybrid_index = HybridRetriever(self.vec_index, self.bm25_index)

        print(f"[System] 載入 LLM 模型至 GPU: {model_path} ...")
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=1024,
            verbose=False
        )

    def ask(self, test_cases):

        # 評測
        results = {"Vector": 0, "BM25": 0, "Hybrid": 0}

        print("\n--- 檢索演算法評測開始 (Hit Rate@2) ---")
        for query, expected_key in test_cases:
            print(f"\n{'='*50}\n[User]：{query}\n{'-'*50}")

            retrieved_docs = {}
            # Vector
            retrieved_docs["Vector"] = self.vec_index.search(query, top_k=2)
            if any(expected_key in r["document"] for r in retrieved_docs["Vector"]): results["Vector"] += 1

            # BM25
            retrieved_docs["BM25"] = self.bm25_index.search(query, top_k=2)
            if any(expected_key in r["document"] for r in retrieved_docs["BM25"]): results["BM25"] += 1

            # Hybrid
            retrieved_docs["Hybrid"] = self.hybrid_index.search(query, top_k=2)
            if any(expected_key in r["document"] for r in retrieved_docs["Hybrid"]): results["Hybrid"] += 1


            # retrieved_docs = self.index.search(query, top_k=2)
            for k in ["Vector", "BM25", "Hybrid"]:
                context = "\n---\n".join([doc["document"] for doc in retrieved_docs[k]])

                prompt = f"""<|im_start|>system
你是一個專業的 GIGABYTE 筆電硬體客服助手。請根據以下提供的產品規格資訊，精準且簡潔地回答使用者的問題。若資訊不足，請明確告知。支援繁體中文與英文。
【參考資訊】
{context}
<|im_end|>
<|im_start|>user
{query}
<|im_end|>
<|im_start|>assistant
"""
                start_time = time.time()
                first_token_time = None
                token_count = 0

                print(f"[檢索方法]：{k}")
                print("[Assistant]：", end="", flush=True)

                stream = self.llm(
                    prompt,
                    max_tokens=256,
                    stop=["<|im_end|>"],
                    stream=True
                )

                for output in stream:
                    token = output["choices"][0]["text"]
                    if first_token_time is None and token.strip():
                        first_token_time = time.time()

                    print(token, end="", flush=True)
                    token_count += 1

                end_time = time.time()
                print()

                ttft = first_token_time - start_time if first_token_time else 0
                generation_time = end_time - first_token_time if first_token_time else 0
                tps = (token_count - 1) / generation_time if generation_time > 0 else 0

                print(f"[效能指標]：TTFT (首字延遲): {ttft:.3f} s | TPS (生成速度): {tps:.2f} tokens/s")
                print()

        # 4. 輸出結果
        total = len(test_cases)
        print(f"{'Algorithm':<10} | {'Hit Rate@2':<10}")
        print("-" * 25)
        for algo, hits in results.items():
            accuracy = (hits / total) * 100
            print(f"{algo:<10} | {accuracy:>6.1f} % ({hits}/{total})")

"""# 4. 系統評測"""

parser = SpecParser()
chunks = parser.get_chunks()

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
    ("處理器是用 Intel Ultra 還是 AMD 的？", "Intel® Core™ Ultra"),
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

if __name__ == "__main__":
    assistant = AorusAssistant(chunks, model_path="./models/qwen_3b_q4.gguf")
    assistant.ask(test_cases)
    assistant.ask(mix_test_cases)

    assistant = AorusAssistant(chunks, model_path="./models/qwen_1.5b_q8.gguf")
    assistant.ask(test_cases)
    assistant.ask(mix_test_cases)

    assistant = AorusAssistant(chunks, model_path="./models/llama_3b_q4.gguf")
    assistant.ask(test_cases)
    assistant.ask(mix_test_cases)
