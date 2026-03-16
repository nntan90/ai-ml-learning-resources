# 🤖 AI Agents

Tài liệu về xây dựng AI Agents — từ cơ bản đến nâng cao, bao gồm các frameworks phổ biến và ví dụ thực tế.

---

## 📚 Repos trong chủ đề này

| Repo | Stars | Tác giả | Mô tả |
|------|-------|---------|-------|
| [ai-agents-for-beginners](https://github.com/microsoft/ai-agents-for-beginners) | ⭐ 54K+ | Microsoft | 15-lesson course on building AI agents |
| [GenAI_Agents](https://github.com/NirDiamant/GenAI_Agents) | ⭐ 20K+ | NirDiamant | 45 GenAI agent tutorials, beginner → advanced |

---

## 1. microsoft/ai-agents-for-beginners

> **GitHub**: https://github.com/microsoft/ai-agents-for-beginners  
> **Stars**: ~54,000 ⭐  
> **Tác giả**: Microsoft  
> **License**: MIT

### Mô tả

Course 15 bài học từ Microsoft về xây dựng AI Agents. Phù hợp cho người mới bắt đầu, có hướng dẫn thực hành với các framework phổ biến nhất trong hệ sinh thái Microsoft AI.

### Nội dung khóa học

| Lesson | Chủ đề |
|--------|--------|
| 00 | Course Setup — cài đặt môi trường |
| 01 | Introduction to AI Agents |
| 02 | Agentic Frameworks Overview |
| 03 | Building Agents with Semantic Kernel |
| 04 | Multi-Agent Collaboration |
| 05 | Tool Use & Function Calling |
| 06 | Memory & State Management |
| 07 | Planning & Reasoning |
| 08 | AutoGen Framework |
| 09 | LangGraph for Workflow Agents |
| 10 | Azure AI Agent Service |
| 11 | Retrieval-Augmented Agents (RAG + Agents) |
| 12 | Evaluation & Safety |
| 13 | Production Deployment |
| 14 | Case Studies |
| 15 | Browser Use & Agentic Web |

### Frameworks được dạy

| Framework | Mô tả |
|-----------|-------|
| **AutoGen** | Multi-agent conversation framework của Microsoft Research |
| **Semantic Kernel** | SDK tích hợp LLM vào ứng dụng .NET / Python |
| **LangGraph** | Graph-based workflow orchestration cho LLM agents |
| **Azure AI Agent Service** | Managed service cho production agents trên Azure |

### Điểm nổi bật
- Jupyter notebooks cho từng bài học
- Video lectures kèm theo
- Hỗ trợ Python và .NET
- Tích hợp sâu với Azure OpenAI

---

## 2. NirDiamant/GenAI_Agents

> **GitHub**: https://github.com/NirDiamant/GenAI_Agents  
> **Stars**: ~20,000 ⭐  
> **Tác giả**: Nir Diamant  
> **License**: MIT

### Mô tả

Bộ sưu tập 45+ tutorials thực hành về GenAI Agents, được tổ chức từ beginner đến advanced. Mỗi tutorial là một Jupyter notebook hoàn chỉnh với code chạy được.

### Phân loại Agents

#### 🟢 Beginner Agents
| Agent | Mô tả |
|-------|-------|
| Simple Conversational Agent | Chatbot cơ bản với memory |
| Task-Specific Agent | Agent chuyên biệt cho 1 task |
| Tool-Using Agent | Agent gọi external tools/APIs |
| ReAct Agent | Reasoning + Acting pattern |

#### 🔵 Framework-Based Agents
| Agent | Framework |
|-------|-----------|
| LangChain Agent | LangChain + LCEL |
| LangGraph Agent | Graph-based workflow |
| CrewAI Agent | Multi-agent collaboration |
| AutoGen Agent | Microsoft AutoGen |

#### 📚 Educational Agents
| Agent | Mô tả |
|-------|-------|
| Tutoring Agent | Dạy học cá nhân hoá |
| Quiz Generation Agent | Tạo câu hỏi ôn tập |
| Explanation Agent | Giải thích khái niệm phức tạp |

#### 💼 Business Agents
| Agent | Mô tả |
|-------|-------|
| Customer Service Agent | Hỗ trợ khách hàng |
| Sales Agent | Tư vấn bán hàng |
| HR Agent | Tuyển dụng & onboarding |
| Finance Agent | Phân tích tài chính |

#### 🎨 Creative Agents
| Agent | Mô tả |
|-------|-------|
| Story Writing Agent | Viết truyện sáng tạo |
| Code Generation Agent | Sinh code từ mô tả |
| Image Prompt Agent | Tạo prompts cho image AI |

#### 📊 Analysis Agents
| Agent | Mô tả |
|-------|-------|
| Data Analysis Agent | Phân tích dữ liệu với Python |
| Research Agent | Tổng hợp thông tin từ nhiều nguồn |
| Sentiment Analysis Agent | Phân tích cảm xúc văn bản |

#### 📰 News & Information Agents
| Agent | Mô tả |
|-------|-------|
| News Summarization Agent | Tóm tắt tin tức hàng ngày |
| Fact-Checking Agent | Kiểm tra độ chính xác |
| Trend Analysis Agent | Phát hiện xu hướng |

#### 🛒 Shopping Agents
| Agent | Mô tả |
|-------|-------|
| Product Recommendation Agent | Gợi ý sản phẩm phù hợp |
| Price Comparison Agent | So sánh giá từ nhiều nguồn |

#### 🚀 Advanced Agents
| Agent | Mô tả |
|-------|-------|
| Multi-Agent Orchestration | Điều phối nhiều agents phối hợp |
| Self-Improving Agent | Agent tự tối ưu hóa |
| Long-horizon Planning Agent | Lập kế hoạch dài hạn |
| Autonomous Research Agent | Nghiên cứu hoàn toàn tự động |

### Điểm nổi bật
- 45+ notebooks thực hành hoàn chỉnh
- Phủ rộng từ simple → production-grade agents
- Kết hợp tốt với repo RAG_Techniques cùng tác giả
- Thường xuyên cập nhật với trends mới nhất

---

## 🗺️ Lộ trình học AI Agents

```
1. Nền tảng (prerequisite)
   └── Hiểu LLMs (→ 05-llm.md)
   └── Hiểu Prompt Engineering (→ 06-prompt-engineering.md)
   └── Hiểu RAG (→ 07-rag.md)

2. Bắt đầu với Agents
   └── ai-agents-for-beginners: Lesson 01-05
   └── GenAI_Agents: Beginner category

3. Framework Practice
   └── ai-agents-for-beginners: Lesson 06-10
   └── GenAI_Agents: Framework-Based category

4. Domain Applications
   └── GenAI_Agents: Business / Educational / Creative

5. Advanced & Production
   └── ai-agents-for-beginners: Lesson 11-15
   └── GenAI_Agents: Advanced category
```

---

## 🔗 Tài nguyên liên quan

- **RAG + Agents**: [07-rag.md](./07-rag.md) — RAG_Techniques cùng tác giả Nir Diamant
- **LLMs nền tảng**: [05-llm.md](./05-llm.md)
- **Prompt Engineering**: [06-prompt-engineering.md](./06-prompt-engineering.md)
- **MLOps cho Agents**: [11-mlops.md](./11-mlops.md)
