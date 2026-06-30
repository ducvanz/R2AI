from __future__ import annotations
import pandas as pd
import numpy as np
import re
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from llama_cpp import Llama
from typing import Callable, List



from sentence_transformers import CrossEncoder
from .retrieval_pipeline import PrecisionRetrieval
from .data_presentation import RetrievalQuery
from .preprocess_parquet import restore_article


def min_max_regularization(logits) :
        max_l = logits.max()
        min_l = logits.min()
                
        # Tránh lỗi chia cho 0 nếu tất cả logits bằng nhau
        if max_l == min_l:
            scores =  np.ones_like(logits)
        else :        
            scores = (logits - min_l) / (max_l - min_l)

        return "MinMax", scores

def z_score_regularization(logits):
    """
    Chuẩn hóa Z-score cho logits. 
    Lưu ý: Bạn có thể truyền thẳng numpy array hoặc torch.Tensor đã detach().numpy()
    """
    # Đảm bảo logits là numpy array để dùng các hàm np
    logits_array = np.array(logits) 
    
    mean_l = np.mean(logits_array)
    std_l = np.std(logits_array)
    
    # Tránh lỗi chia cho 0 nếu tất cả logits bằng nhau
    if std_l == 0:
        # Nếu mọi thứ bằng nhau, Z-score = 0 (tất cả đều nằm ở mức trung bình)
        scores = np.zeros_like(logits_array)
    else:
        scores = (logits_array - mean_l) / std_l + 0.9

    return "Z", scores

def gap_threshold(logits):
    """
    Tính toán ngưỡng (threshold) động dựa trên độ lệch (gap) lớn nhất giữa các logits.
    """
    logits_array = np.array(logits)
    
    # Nếu có quá ít tài liệu, không đủ để tạo gap, trả về giá trị thấp nhất
    if len(logits_array) < 2:
        return np.min(logits_array)
        
    # Sắp xếp điểm giảm dần
    sorted_logits = np.sort(logits_array)[::-1]
    
    # Tính khoảng cách (gap) giữa vị trí thứ i và i+1
    gaps = sorted_logits[:-1] - sorted_logits[1:]
    
    # Tìm index của gap lớn nhất
    max_gap_idx = np.argmax(gaps)
    
    
    # Ngưỡng (Threshold) sẽ là điểm số nằm ngay trước khi bị "rớt hố"
    dynamic_threshold = sorted_logits[max_gap_idx]

    binary_scores = np.where(logits_array >= dynamic_threshold, 1, 0)
    
    return "Gap", binary_scores

def robust_regularization(logits, p_min=5, p_max=95):
    logits_array = np.array(logits)
    
    if len(logits_array) < 2:
        return np.ones_like(logits_array)
        
    # Lấy phân vị (percentile) thay vì min/max tuyệt đối
    val_max = np.percentile(logits_array, p_max)
    val_min = np.percentile(logits_array, p_min)
    
    if val_max == val_min:
        return np.ones_like(logits_array)
        
    # Scale và cắt (clip) những giá trị vượt biên về đúng [0, 1]
    scores = (logits_array - val_min) / (val_max - val_min)
    
    return "Robust", scores

def temperature_sigmoid(logits, T=4.0):
    """
    T = 1.0: Tương đương Sigmoid gốc (dễ bão hòa).
    T > 1.0: Làm mềm phân phối, tránh bão hòa (Khuyên dùng T từ 2.0 đến 5.0).
    """
    logits_array = np.array(logits)
    
    # Chia logits cho Nhiệt độ T để tránh bão hòa
    scaled_logits = logits_array / T
    
    # Tính Sigmoid an toàn (tránh tràn số exp)
    # np.exp(-x) có thể bị overflow nếu x quá âm, nên dùng công thức ổn định
    scores = np.where(
        scaled_logits >= 0, 
        1 / (1 + np.exp(-scaled_logits)), 
        np.exp(scaled_logits) / (1 + np.exp(scaled_logits))
    )
    
    return 'Sigmoid', scores

class CrossEncoderReranker(PrecisionRetrieval):
    def __init__(self, model: CrossEncoder, 
                 reg_funcs:List[Callable] = None,
                 batch_size:int = 32, max_length:int = 8192, 
                 name: str = "CrossEncoder"):
        """
        Khởi tạo Precision Retrieval sử dụng Cross-Encoder.

        Args:
            model (CrossEncoder): Instance của CrossEncoder (VD: BAAI/bge-reranker-v2-m3).
            top_k (int): Số lượng kết quả cuối cùng mong muốn.
            name (str): Tên định danh.
        """
        self.model = model
        self.regularization:List[Callable] = reg_funcs or [min_max_regularization, z_score_regularization, gap_threshold, robust_regularization, temperature_sigmoid]

        self.model.max_seq_length = max_length
        self.batch_size = batch_size
        self.name = name

    def forward(self, query: RetrievalQuery, document: pd.Series) -> pd.DataFrame:
        # 1. Chuẩn bị cặp (Query, Passage)
        # Giả sử document có cột 'content_text' chứa nội dung cần rerank
        pairs = [[query.content, restore_article(doc_text)] for doc_text in document]
        
        # 2. Dự đoán điểm số (Cross-Encoder không cần vector, nó trả về logits)
        with torch.inference_mode():
            logits = self.model.predict(pairs, batch_size=self.batch_size)
        
        final_results = pd.DataFrame(index=document.index)
        for reg_func in self.regularization :
            name, scores = reg_func(logits)
            final_results[f'{self.name}_{name}_score'] = scores

        # 3. Ghi điểm vào DataFrame
        return final_results


    
class HyDEProcessor:
    def __init__(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        system_prompt: str = None,
        max_new_tokens: int = 256,
        temperature: float = 0.2
    ):
        """
        Khởi tạo HyDE Processor chạy trực tiếp bằng Local Model.
        
        Args:
            model: Mô hình CausalLM của HuggingFace (đã nạp sẵn lên VRAM/CPU).
            tokenizer: Tokenizer tương ứng của mô hình.
            max_new_tokens: Độ dài tối đa của văn bản giả định được sinh ra.
            temperature: Độ sáng tạo. Với văn bản pháp luật, nên giữ thấp (0.1 - 0.3).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Prompt được tối ưu, yêu cầu model đóng vai chuyên gia pháp lý
        self.system_prompt = system_prompt or (
            "Là một chuyên gia pháp lý Việt Nam, hãy viết một đoạn văn bản ngắn (3-4 câu) "
            "sử dụng đúng thuật ngữ chuyên ngành để trả lời trực tiếp vấn đề sau.\n"
            "Chỉ viết nội dung luật, tuyệt đối không giải thích hay chào hỏi.\n\n"
        )

    def enhance(self, query: str, return_mode: str = "concat") -> str:
        """
        Sinh ra truy vấn mở rộng bằng HyDE.
        """
        user_prompt = (
            f"Câu hỏi:\n{query}\n\n"
            "Đoạn văn:"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,

                max_new_tokens=self.max_new_tokens,

                do_sample=False,

                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,

                repetition_penalty=1.05,
            )

        generated = outputs[0][inputs.input_ids.shape[1]:]

        hypothetical_doc = self.tokenizer.decode(
            generated,
            skip_special_tokens=True
        ).strip()

        # ==========================
        # Clean output
        # ==========================

        hypothetical_doc = re.sub(
            r"^(Assistant|User|System)\s*:\s*",
            "",
            hypothetical_doc,
            flags=re.IGNORECASE,
        )

        # fallback nếu đầu ra bất thường
        if (
            len(hypothetical_doc) < 20
            or "This is a great start" in hypothetical_doc
            or "Certainly!" in hypothetical_doc
        ):
            hypothetical_doc = query

        if return_mode == "concat":
            return f"{query}\n{hypothetical_doc}"

        return hypothetical_doc

class LlamaHyDEProcessor:
    def __init__(
        self, 
        model_path: str,
        n_gpu_layers: int = -1,  # -1 có nghĩa là đẩy tối đa các layer lên GPU (như cấu hình Kaggle của bạn)
        n_ctx: int = 2048,       # Context window size
        n_batch: int = 512,
        system_prompt: str = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        verbose: bool = False,
        **kwargs
    ):
        """
        Khởi tạo HyDE Processor chạy trực tiếp bằng Local Model thông qua llama-cpp-python (GGUF).
        
        Args:
            model_path: Đường dẫn tới file model .gguf (VD: "/kaggle/working/models/...gguf")
            n_gpu_layers: Số lượng layer đẩy lên GPU (-1 là toàn bộ, 0 là chạy CPU thuần)
            n_ctx: Kích thước cửa sổ ngữ cảnh (Context Window)
            n_batch: Số lượng token xử lý đồng thời trong một batch
            system_prompt: Chỉ thị hệ thống dành cho chuyên gia pháp lý
            max_new_tokens: Độ dài tối đa của văn bản giả định được sinh ra
            temperature: Độ sáng tạo khi sinh văn bản
        """
        # Khởi tạo instance Llama từ llama-cpp-python
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_batch=n_batch,
            verbose=verbose,
            **kwargs
        )
        
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        # Prompt được tối ưu, yêu cầu model đóng vai chuyên gia pháp lý
        self.system_prompt = system_prompt or (
            "Là một chuyên gia pháp lý Việt Nam, hãy viết một đoạn văn bản ngắn (3-4 câu) "
            "sử dụng đúng thuật ngữ chuyên ngành để trả lời trực tiếp vấn đề sau.\n"
            "Chỉ viết nội dung luật, tuyệt đối không giải thích hay chào hỏi.\n\n"
        )

    def enhance(self, query: str, return_mode: str = "concat") -> str:
        """
        Sinh ra truy vấn mở rộng bằng HyDE sử dụng cấu trúc create_chat_completion.
        """
        user_prompt = (
            f"Câu hỏi:\n{query}\n\n"
            "Đoạn văn:"
        )

        # Định dạng tin nhắn chuẩn OpenAI format
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            # Gọi API chat completion nội bộ của llama-cpp. 
            # Hàm này tự động wrap chat template tương ứng với model đang chạy.
            output = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                # Lưu ý: Không đặt stop=["\n"] ở đây vì HyDE cần sinh 1 đoạn văn (paragraph) chứa nhiều câu.
            )
            
            # Trích xuất nội dung văn bản giả định (Hypothetical Document)
            hypothetical_doc = output["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            print(f"[WARN] Lỗi model khi chạy HyDE: {e}. Kích hoạt cơ chế Fallback (dùng query gốc).")
            hypothetical_doc = query

        # ==========================
        # Clean output (Giữ nguyên logic xử lý chuỗi ban đầu của bạn)
        # ==========================
        hypothetical_doc = re.sub(
            r"^(Assistant|User|System)\s*:\s*",
            "",
            hypothetical_doc,
            flags=re.IGNORECASE,
        )

        # Fallback nếu đầu ra bất thường hoặc quá ngắn
        if (
            len(hypothetical_doc) < 20
            or "This is a great start" in hypothetical_doc
            or "Certainly!" in hypothetical_doc
        ):
            hypothetical_doc = query

        if return_mode == "concat":
            return f"{query}\n{hypothetical_doc}"

        return hypothetical_doc