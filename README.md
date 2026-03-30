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

## 🚀 # AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 315865595366.dkr.ecr.us-east-1.amazonaws.com/rag

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app

## 🎓 Academic Credit
Developer: Salony Ranjan	
