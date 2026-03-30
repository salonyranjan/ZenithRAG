# ZenithRAG 🚀 | Advanced RAG-based AI Analysis Pipeline

[![Live Website](https://img.shields.io/badge/Live-ZenithRAG-green?style=for-the-badge&logo=amazonaws)](http://13.60.233.173:8080)
[![Tech Stack](https://img.shields.io/badge/Stack-Llama_3.3_|_LangChain_|_Docker-blue?style=for-the-badge)](https://github.com/salonyranjan/ZenithRAG)

**ZenithRAG** is a high-performance Retrieval-Augmented Generation (RAG) platform designed to transform static PDF documents into interactive, intelligence-driven conversations. Built with a globally distributed CI/CD architecture, it leverages cutting-edge LLMs and vector databases to provide context-aware insights with millisecond latency.

---

## 🌐 Live Deployment
The application is currently deployed and accessible globally:
🔗 **[http://13.60.233.173:8080](http://13.60.233.173:8080)**

---

## 🛠️ Tech Stack & Architecture

### **Core Intelligence**
* **LLM Orchestration:** LangChain
* **Primary Brain:** Llama 3.3 (via Groq Inference Engine)
* **Embeddings:** HuggingFace `sentence-transformers`
* **Vector Store:** FAISS (Facebook AI Similarity Search)

### **DevOps & Cloud (The "Patna-to-Stockholm" Pipeline)**
* **Cloud Provider:** AWS (Amazon Web Services)
* **Compute:** EC2 (Stockholm Region - `eu-north-1`)
* **Containerization:** Docker & Amazon ECR
* **Automation:** GitHub Actions (Self-hosted runners)
* **Storage:** Amazon S3 (`zenithragbucket`)

---

## 🏗️ System Architecture
ZenithRAG utilizes a modular pipeline to ensure data privacy and retrieval accuracy:

1.  **Ingestion:** PDFs are retrieved via `gdown` or S3 and processed into normalized text chunks.
2.  **Vectorization:** HuggingFace embeddings convert text into high-dimensional vectors.
3.  **Indexing:** FAISS creates an efficient searchable index of the document context.
4.  **Retrieval:** When a user asks a question, the system finds the top-$k$ most relevant context snippets.
5.  **Generation:** The Llama 3.3 model synthesizes the final answer using only the retrieved facts.

---

## 🚀 CI/CD Pipeline
This project features a fully automated deployment workflow:
* **Continuous Integration:** Code linting and environment validation on every push.
* **Continuous Delivery:** Automated Docker builds pushed to **Amazon ECR**.
* **Continuous Deployment:** A self-hosted GitHub runner on an **AWS EC2** instance pulls the latest image and restarts the service instantly.

---

## 🔧 Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/salonyranjan/ZenithRAG.git](https://github.com/salonyranjan/ZenithRAG.git)
   cd ZenithRAG
