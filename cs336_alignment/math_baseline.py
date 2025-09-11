from typing import Callable, List
from vllm import LLM, SamplingParams
import json
import os
import re
from datetime import datetime

# -------------------------
# evaluate_vllm 函数
# -------------------------
def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """

    # 1. 生成模型输出
    print(f"[INFO] Generating responses for {len(prompts)} prompts...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []
    total_scores = {}

    # 2. 逐条计算分数
    for prompt, output in zip(prompts, outputs):
        text = output.outputs[0].text.strip()
        scores = reward_fn(prompt, text)

        # 更新总分（方便后续计算平均值）
        for k, v in scores.items():
            total_scores[k] = total_scores.get(k, 0.0) + v

        results.append({
            "prompt": prompt,
            "response": text,
            "scores": scores
        })

    # 3. 计算平均分
    avg_scores = {k: v / len(prompts) for k, v in total_scores.items()}

    # 4. 准备输出数据
    output_data = {
        "results": results,
        "average_scores": avg_scores,
        "num_samples": len(prompts),
        "timestamp": datetime.now().isoformat()
    }

    # 5. 保存到磁盘
    os.makedirs("eval_results", exist_ok=True)
    save_path = os.path.join("eval_results", f"vllm_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Evaluation complete! Results saved to: {save_path}")

# -------------------------
# GSM8K 专用 reward_fn 构造
# -------------------------
def make_gsm8k_reward_fn(answers: List[str]) -> Callable[[str, str], dict[str, float]]:
    """
    构造一个 GSM8K 专用 reward_fn。
    answers: 正确答案列表，与 prompts 一一对应。
    """
    def reward_fn(prompt: str, response: str, idx=[0]) -> dict[str, float]:
        i = idx[0]
        idx[0] += 1

        gt = answers[i]
        # 提取最后一个数字作为预测答案
        match = re.findall(r'\d+', response)
        pred = match[-1] if match else ""
        correct = 1.0 if pred == gt else 0.0
        return {"accuracy": correct}
    return reward_fn

# -------------------------
# 读取 GSM8K .jsonl 数据
# -------------------------
def load_gsm8k_jsonl(path: str):
    prompts, answers = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                q = item["question"]
                # 答案在最后一行 #### 后面
                ans_line = item["answer"].strip().split("####")[-1].strip()
                prompts.append(q)
                answers.append(ans_line)
    return prompts, answers

# -------------------------
# 主程序
# -------------------------
if __name__ == "__main__":
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    # Hugging Face hub 镜像
    os.environ["HF_HOME"] = "data"  # 可选：自定义缓存目录

    # 国内镜像加速（例如清华镜像）
    os.environ["HF_HUB_URL"] = "https://huggingface.tuna.tsinghua.edu.cn"  # 官方源也可
    os.environ["HF_HUB_OFFLINE"] = "0"  # 强制在线下载

    # 1. 加载 GSM8K 测试数据
    prompts, answers = load_gsm8k_jsonl("data/gsm8k/test.jsonl")  # 请将 test.jsonl 放在同目录

    # 2. 初始化 vLLM 模型
    llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")  # 可换成你自己的模型
    sampling_params = SamplingParams(temperature=0, max_tokens=100)

    # 3. 构造 GSM8K reward_fn
    reward_fn = make_gsm8k_reward_fn(answers)

    # 4. 调用评估函数
    evaluate_vllm(llm, reward_fn, prompts, sampling_params)
