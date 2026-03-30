# ZenithRAG 🚀 | Advanced RAG-based AI Analysis Pipeline

[![Live Website](https://img.shields.io/badge/Live-ZenithRAG-green?style=for-the-badge&logo=amazonaws)](http://13.60.233.173:8080)
[![Tech Stack](https://img.shields.io/badge/Stack-Llama_3.3_|_LangChain_|_Docker-blue?style=for-the-badge)](https://github.com/salonyranjan/ZenithRAG)

**ZenithRAG** is a high-performance Retrieval-Augmented Generation (RAG) platform designed to transform static PDF documents into interactive, intelligence-driven conversations. Built with a globally distributed CI/CD architecture, it leverages cutting-edge LLMs and vector databases to provide context-aware insights with millisecond latency.

---

## 🌐 Live Deployment
The application is currently deployed and accessible globally via AWS Stockholm:
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
* **Containerization:** Docker & Amazon ECR (`577435557871.dkr.ecr.eu-north-1.amazonaws.com/zenithrag`)
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

## 🔧 Local Setup & Installation

### **STEP 01: Create Environment**
Open your terminal and create a dedicated Conda environment:
```bash
conda create -n zenith_env python=3.11 -y
conda activate zenith_env
```
### **STEP 02: Install Requirements**
Install the core RAG dependencies:

```bash
pip install -r requirements.txt
```
### **STEP 03: Run Locally**
Launch the application on your localhost:

```bash
python app/main.py
```
Access the UI at http://localhost:8080

## 🚀 AWS CI/CD Deployment Guide
### **1. IAM Configuration**
Create an IAM user for deployment with the following policies:

AmazonEC2ContainerRegistryFullAccess

AmazonEC2FullAccess

### **2. EC2 Environment Setup (Ubuntu 22.04)**
Install the Docker engine on your Stockholm instance:

```bash
sudo apt-get update -y
curl -fsSL [https://get.docker.com](https://get.docker.com) -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu && newgrp docker
```
### **3. GitHub Self-Hosted Runner**
Configure your EC2 as a runner:
Settings > Actions > Runners > New self-hosted runner (Follow the provided Ubuntu instructions).

### **4. GitHub Secrets Setup**
Add the following secrets to your repository:

AWS_ACCESS_KEY_ID: Your IAM Access Key

AWS_SECRET_ACCESS_KEY: Your IAM Secret Key

AWS_REGION: eu-north-1

GROQ_API_KEY: Your Groq API Key

OPENAI_API_KEY: Your OpenAI API Key
