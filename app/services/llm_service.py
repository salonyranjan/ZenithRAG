import os
import logging

# --- 2026 CLASSIC STABLE IMPORTS ---
try:
    from langchain_classic.chains import ConversationalRetrievalChain
    from langchain_classic.memory import ConversationBufferMemory
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain_groq import ChatGroq
    from langchain_core.prompts import PromptTemplate
    ConversationBufferMemory = None
    ConversationalRetrievalChain = None

logger = logging.getLogger("ZenithRAG.LLMService")

class LLMService:
    def __init__(self, vector_store_wrapper):
        """
        ZenithRAG: High-speed Groq Engine (Llama 3.3).
        """
        self.vector_store_wrapper = vector_store_wrapper
        
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("LLM_SERVICE: GROQ_API_KEY missing!")
            raise ValueError("GROQ_API_KEY not found. Check your .env file.")

        self.llm = ChatGroq(
            temperature=0.2,
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile"
        )

        # --- CORRECTED TEMPLATE PLACEMENT ---
        self.template = """You are ZenithRAG, an expert Technical Instructor.
        
### INSTRUCTIONS:
1. If the question involves a calculation (like Manhattan Distance), explain the formula first.
2. Use **Markdown** for all formulas. Example: $|x1 - x2| + |y1 - y2|$.
3. Use bullet points for steps.
4. Do NOT include internal system notes like "(Note: The provided context...)" in your answer.

### CONTEXT FROM DOCUMENT:
{context}

### CONTEXT FROM CHAT:
{chat_history}

Question: {question}

Answer (Formatted as a Technical Guide):"""

        self.prompt = PromptTemplate(
            template=self.template, 
            input_variables=["context", "chat_history", "question"]
        )

        if ConversationBufferMemory:
            self.memory = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True,
                output_key='answer'
            )
        else:
            self.memory = None

        self.chain = None
        self._check_and_init_chain()

    def _check_and_init_chain(self):
        if not ConversationalRetrievalChain:
            logger.error("LLM_SERVICE: langchain-classic missing.")
            return False

        if self.vector_store_wrapper and self.vector_store_wrapper.vector_store:
            try:
                self.chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vector_store_wrapper.vector_store.as_retriever(search_kwargs={"k": 3}),
                    memory=self.memory,
                    combine_docs_chain_kwargs={"prompt": self.prompt}
                )
                logger.info("--- ZenithRAG: Technical RAG Chain built successfully ---")
                return True
            except Exception as e:
                logger.error(f"Chain Building Error: {e}")
        return False

    def get_response(self, query: str):
        if not self.chain:
            if not self._check_and_init_chain():
                return "ZenithRAG: System is in standby. Please upload a PDF first."
            
        try:
            result = self.chain.invoke({"question": query})
            return result.get('answer', "I couldn't extract a clear answer.")
        except Exception as e:
            logger.error(f"Inference Error: {e}")
            return f"ZenithRAG (Groq) Error: {str(e)}"

    def clear_history(self):
        if self.memory:
            self.memory.clear()
        logger.info("--- ZenithRAG: Memory Cleared ---")