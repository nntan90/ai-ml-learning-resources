# Prompt Engineering

Kỹ thuật viết prompts hiệu quả cho LLMs.

---

## 🐙 Prompt Engineering Guide — DAIR.AI

**Repo:** [dair-ai/Prompt-Engineering-Guide](https://github.com/dair-ai/Prompt-Engineering-Guide) | ⭐ 71,722  
**Website:** [promptingguide.ai](https://www.promptingguide.ai/) | 13 ngôn ngữ | 3M+ learners

### Kỹ thuật Prompting

#### Cơ bản
| Kỹ thuật | Mô tả | Link |
|----------|-------|------|
| Zero-Shot | Không có ví dụ | [Guide](https://www.promptingguide.ai/techniques/zeroshot) |
| Few-Shot | Cung cấp vài ví dụ | [Guide](https://www.promptingguide.ai/techniques/fewshot) |
| Chain-of-Thought (CoT) | Suy luận từng bước | [Guide](https://www.promptingguide.ai/techniques/cot) |
| Self-Consistency | Sample nhiều paths, vote | [Guide](https://www.promptingguide.ai/techniques/consistency) |

#### Nâng cao
| Kỹ thuật | Mô tả | Link |
|----------|-------|------|
| Tree of Thoughts (ToT) | Nhiều reasoning paths song song | [Guide](https://www.promptingguide.ai/techniques/tot) |
| Prompt Chaining | Kết nối nhiều prompts | [Guide](https://www.promptingguide.ai/techniques/prompt_chaining) |
| ReAct | Reasoning + Tool use | [Guide](https://www.promptingguide.ai/techniques/react) |
| RAG | Retrieval + Generation | [Guide](https://www.promptingguide.ai/techniques/rag) |
| APE | Auto Prompt Engineering | [Guide](https://www.promptingguide.ai/techniques/ape) |
| PAL | Program-Aided Language | [Guide](https://www.promptingguide.ai/techniques/pal) |
| Graph Prompting | Prompting với cấu trúc graph | [Guide](https://www.promptingguide.ai/techniques/graph) |

### LLM Settings
| Parameter | Chức năng |
|-----------|-----------|
| **Temperature** | Randomness (0=deterministic, 1+=creative) |
| **Top-p** | Nucleus sampling — probability mass |
| **Top-k** | Lấy k tokens có xác suất cao nhất |
| **Max Length** | Độ dài tối đa output |

### Rủi ro cần biết
- **Adversarial Prompting:** Prompt injection, jailbreaking → [Guide](https://www.promptingguide.ai/risks/adversarial)
- **Factuality:** Hallucination, factual errors → [Guide](https://www.promptingguide.ai/risks/factuality)
- **Biases:** Stereotypes, confirmation bias → [Guide](https://www.promptingguide.ai/risks/biases)

### Prompt Hub (Thư viện mẫu)
- [Classification](https://www.promptingguide.ai/prompts/classification)
- [Coding](https://www.promptingguide.ai/prompts/coding)
- [Reasoning](https://www.promptingguide.ai/prompts/reasoning)
- [Text Summarization](https://www.promptingguide.ai/prompts/text-summarization)
- [Mathematics](https://www.promptingguide.ai/prompts/mathematics)
- [Adversarial](https://www.promptingguide.ai/prompts/adversarial-prompting)
