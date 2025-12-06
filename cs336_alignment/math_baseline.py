from typing import Callable, List
from vllm import LLM, SamplingParams
from transformers import AutoModel
from drgrpo_grader import r1_zero_reward_fn
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
    answers: List[str],
    eval_sampling_params: SamplingParams
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    # 创建结果目录
    os.makedirs("results", exist_ok=True)
    
    # 生成时间戳用于文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/evaluation_{timestamp}.json"
    
    # 1. 生成模型输出
    print(f"[INFO] Generating responses for {len(prompts)} prompts...")
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    
    # 准备保存所有结果
    all_results = []
    total = 0
    format_reward = 0
    answer_reward = 0
    reward = 0
    
    # 2. 计算分数并收集结果
    for i, (output, answer, prompt) in enumerate(zip(outputs, answers, prompts)):
        total += 1
        generated_text = output.outputs[0].text
        score = reward_fn(generated_text, answer)
        
        # 累加分数
        format_reward += score["format_reward"]
        answer_reward += score["answer_reward"]
        reward += score["reward"]
        
        # 保存每个样本的详细信息
        sample_result = {
            "index": i,
            "prompt": prompt,
            "generated_text": generated_text,
            "reference_answer": answer,
            "format_reward": score["format_reward"],
            "answer_reward": score["answer_reward"],
            "total_reward": score["reward"]
        }
        all_results.append(sample_result)
    
    # 计算平均分数
    avg_format_reward = format_reward / total
    avg_answer_reward = answer_reward / total
    avg_reward = reward / total
    
    # 创建包含所有结果的字典
    final_results = {
        "timestamp": timestamp,
        "model": vllm_model.llm_engine.model_config.model,
        "sampling_params": {
            "temperature": eval_sampling_params.temperature,
            "max_tokens": eval_sampling_params.max_tokens
        },
        "samples": all_results,
        "summary": {
            "total_samples": total,
            "average_format_score": avg_format_reward,
            "average_answer_score": avg_answer_reward,
            "average_total_score": avg_reward
        }
    }
    
    # 3. 将结果保存到JSON文件
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # 4. 打印结果
    print(f"\n=== EVALUATION RESULTS ===")
    print(f"Results saved to: {results_file}")
    print(f"Total samples: {total}")
    print(f"Average format score: {avg_format_reward:.4f}")
    print(f"Average answer score: {avg_answer_reward:.4f}")
    print(f"Average total score: {avg_reward:.4f}")
    
    # 5. 同时创建一个简化的文本摘要
    summary_file = f"results/summary_{timestamp}.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"Evaluation Summary - {timestamp}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {final_results['model']}\n")
        f.write(f"Temperature: {final_results['sampling_params']['temperature']}\n")
        f.write(f"Max tokens: {final_results['sampling_params']['max_tokens']}\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Average format score: {avg_format_reward:.4f}\n")
        f.write(f"Average answer score: {avg_answer_reward:.4f}\n")
        f.write(f"Average total score: {avg_reward:.4f}\n")
    
    print(f"Summary saved to: {summary_file}")

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
                
                # 格式化为指定的对话prompt格式
                formatted_prompt = format_think_answer_prompt(q)
                
                # 答案在最后一行 #### 后面
                ans_line = item["answer"].strip().split("####")[-1].strip()
                prompts.append(formatted_prompt)  # 使用格式化后的prompt
                answers.append(ans_line)
    return prompts, answers

def format_think_answer_prompt(question: str) -> str:
    """
    将问题格式化为指定的对话prompt格式：
    A conversation between User and Assistant...
    """
    prompt_template = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.
    User: {question}
    Assistant: <think>"""
    
    return prompt_template.format(question=question)

# -------------------------
# 主程序
# -------------------------
if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    # 1. 加载 GSM8K 测试数据
    prompts, answers = load_gsm8k_jsonl("data/gsm8k/test.jsonl")  # 请将 test.jsonl 放在同目录

    # 2. 初始化 vLLM 模型
    # cache_path = AutoModel.from_pretrained("Qwen/Qwen2.5-Math-1.5B", local_files_only=True).name_or_path
    cache_path = "/home/realvm/assignment5-alignment/models/Qwen2.5-Math-1.5B"
    llm = LLM(model=cache_path)
    # llm = LLM(model="Qwen/Qwen2.5-Math-1.5B")
    sampling_params = SamplingParams(temperature=0.7,top_k=50, top_p=0.9, max_tokens=512, stop = "</answer>")
    sampling_params.include_stop_str_in_output = True

    # 3. 调用评估函数
    evaluate_vllm(llm, r1_zero_reward_fn, prompts, answers, sampling_params)
