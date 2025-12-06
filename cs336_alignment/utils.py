import torch
from typing import List
from transformers import PreTrainedTokenizer, PreTrainedModel

def tokenize_prompt_and_output(
    prompt_strs: List[str], 
    output_strs: List[str], 
    tokenizer: PreTrainedTokenizer
) -> dict:
    """
    Tokenize the prompt and output strings, and construct a mask that is 1 for the response tokens 
    and 0 for other tokens (prompt or padding).
    
    Args:
        prompt_strs: List of prompt strings.
        output_strs: List of output strings.
        tokenizer: Tokenizer to use for tokenization.
    
    Returns:
        Dictionary with keys:
        - input_ids: Tensor of shape (batch_size, max_len-1) - tokenized prompt and output with last token sliced off
        - labels: Tensor of shape (batch_size, max_len-1) - shifted input ids (without first token)
        - response_mask: Tensor of shape (batch_size, max_len-1) - mask for response tokens in labels
    """
    batch_size = len(prompt_strs)
    
    # Tokenize prompts and outputs separately, adding special tokens
    prompt_encodings = tokenizer(prompt_strs, add_special_tokens=True)
    output_encodings = tokenizer(output_strs, add_special_tokens=True)
    
    # Calculate lengths for each sample
    prompt_lens = [len(encoding) for encoding in prompt_encodings["input_ids"]]
    output_lens = [len(encoding) for encoding in output_encodings["input_ids"]]

    # Combine prompt and output tokens
    combined_input_ids = []
    for i in range(batch_size):
        combined = prompt_encodings["input_ids"][i] + output_encodings["input_ids"][i]
        combined_input_ids.append(combined)

    # Calculate total lengths and find max length
    total_lens = [p_len + o_len for p_len, o_len in zip(prompt_lens, output_lens)]
    max_len = max(total_lens) if total_lens else 0

    # Initialize tensors for input_ids, labels and response_mask
    # Use 0 as pad token ID to match test expectations
    input_ids_tensor = torch.full((batch_size, max_len - 1), 0, dtype=torch.long)
    labels_tensor = torch.full((batch_size, max_len - 1), -100, dtype=torch.long)
    response_mask_tensor = torch.zeros((batch_size, max_len - 1), dtype=torch.long)

    # Fill the tensors
    for i in range(batch_size):
        # Get the combined tokens for this sample
        tokens = combined_input_ids[i]
        p_len = prompt_lens[i]
        o_len = output_lens[i]
        
        response_mask = [0] * (p_len - 1) + [1] * o_len
        tokens_padded = tokens + [tokenizer.eos_token_id] * (max_len - len(tokens))
        input_ids_padded = tokens_padded[:-1]
        labels_padded = tokens_padded[1:]
        response_mask_padded = response_mask + [0] * (max_len - 1 - len(response_mask))
        input_ids_tensor[i] = torch.tensor(input_ids_padded, dtype=torch.long)
        labels_tensor[i] = torch.tensor(labels_padded, dtype=torch.long)
        response_mask_tensor[i] = torch.tensor(response_mask_padded, dtype=torch.long)

    return {
        'input_ids': input_ids_tensor,
        'labels': labels_tensor,
        'response_mask': response_mask_tensor,
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    
    Args:
        logits: torch.Tensor of shape (batch_size, sequence_length, vocab_size)
                containing unnormalized logits.
    
    Returns:
        torch.Tensor of shape (batch_size, sequence_length). The entropy for each next-token prediction.
    """
    log_probs = torch.nn.functional.log_softmax(logits, -1)
    probs = torch.softmax(logits, -1)

    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Args:
        model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
            and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
            response tokens as produced by your tokenization method.
        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your
            tokenization method.
        return_token_entropy: bool If True, also return per-token entropy by calling
            compute_entropy.
    
    Returns:
        dict[str, torch.Tensor].
            "log_probs" shape (batch_size, sequence_length), conditional log-probabilities
                log pθ(xt |x<t).
            "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy
                for each position (present only if return_token_entropy=True)
    """
    model.eval()

    input_ids = input_ids.to(model.device)
    labels = labels.to(model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    log_probs = torch.nn.functional.log_softmax(logits, -1)

    response_log_probs = torch.gather(
        log_probs, 
        dim=-1, 
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    mask = (labels != -100)
    response_log_probs = response_log_probs * mask.float()

    result = {}
    result["log_probs"] = response_log_probs
    
    if return_token_entropy:
        entropy = compute_entropy(logits)
        entropy = entropy * mask.float()
        result["token_entropy"] = entropy
    
    return result

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    """
        Sum over a dimension and normalize by a constant, considering only those elements where mask
            == 1.

        Args:
            tensor: torch.Tensor The tensor to sum and normalize.
            mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
            normalize_constant: float the constant to divide by for normalization.
            dim: int | None the dimension to sum along before normalization. If None, sum over all
                dimensions.
        Returns:
            torch.Tensor the normalized sum, where masked elements (mask == 0) don't contribute to
                the sum.
    """
    # 应用掩码：将mask为0的位置的值设为0
    masked_tensor = tensor * mask
    
    # 根据dim参数决定求和方式
    if dim is None:
        # 对所有维度求和
        summed = masked_tensor.sum()
    else:
        # 沿着指定维度求和
        summed = masked_tensor.sum(dim=dim)
    
    # 除以归一化常数
    normalized = summed / normalize_constant
    
    return normalized