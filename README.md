# Hindi RAG with Qwen and FAISS

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for answering questions in **Hindi**.

The system combines:

* A Hindi news/story dataset from Hugging Face
* Multilingual sentence embeddings for semantic search
* FAISS for efficient vector retrieval
* The **Qwen/Qwen1.5-1.8B-Chat** language model for Hindi response generation

Instead of asking the language model to answer using only its pre-trained knowledge, the system first retrieves relevant information from the dataset and provides it as context to the LLM.

## Project Pipeline

```text
Hindi Dataset
     ↓
Text Chunking
     ↓
Sentence Embeddings
     ↓
FAISS Vector Index
     ↓
User Question
     ↓
Semantic Retrieval
     ↓
Relevant Context
     ↓
Hindi RAG Prompt
     ↓
Qwen LLM
     ↓
Generated Hindi Answer
```

## Dataset

The project uses the **Hindi Story News** dataset:

**Hugging Face Dataset:** `Threatthriver/Hindi-story-news`

The dataset is loaded using the Hugging Face `datasets` library and converted into a pandas DataFrame for inspection and processing.

The main text used for retrieval is stored in the **`paragraphs`** column.

## Models Used

### Language Model

**Qwen/Qwen1.5-1.8B-Chat**

The Qwen chat model is used to generate the final response in Hindi.

The notebook:

* Loads the tokenizer
* Loads the causal language model
* Applies the tokenizer's chat template
* Generates responses using `model.generate()`

### Embedding Model

**paraphrase-multilingual-MiniLM-L12-v2**

This multilingual Sentence Transformer model converts Hindi text into numerical embeddings.

These embeddings allow the system to perform **semantic search**, meaning it can retrieve text based on meaning rather than exact keyword matches.

### Vector Database

**FAISS**

FAISS (`faiss-cpu`) is used to store and search the generated embeddings efficiently.

The notebook uses:

```python
faiss.IndexFlatL2
```

to perform similarity searches over the embedded text chunks.

## Main Components

### 1. Environment Setup

The required Python libraries are installed using `pip`.

Major dependencies include:

* `transformers`
* `datasets`
* `sentence-transformers`
* `faiss-cpu`
* `accelerate`
* `bitsandbytes`
* `pandas`
* `torch`

### 2. Dataset Loading and Inspection

The Hindi dataset is loaded from Hugging Face and converted into a pandas DataFrame.

The notebook performs basic inspection using:

* `df.head()`
* `df.info()`
* `df.iloc[]`
* pandas display options

This helps understand the dataset structure and inspect the available Hindi text.

### 3. Hindi LLM Response Generation

A function named `generate_hindi_llm_response` is implemented to:

1. Receive a prompt
2. Format it using the Qwen chat template
3. Tokenize the input
4. Generate a response using Qwen
5. Decode and clean the generated output

### 4. Hindi RAG Prompt

A custom `HINDI_RAG_PROMPT_TEMPLATE` is used to structure the retrieved information and user question before sending them to the language model.

This ensures that the LLM generates its answer using the retrieved context.

### 5. Hindi Retriever

The `HindiRetriever` class handles the retrieval process.

It:

1. Loads the multilingual embedding model
2. Processes and chunks the dataset's `paragraphs`
3. Generates embeddings for the chunks
4. Builds a FAISS index
5. Searches for the most relevant chunks for a given question

### 6. RAG Pipeline

The `answer_in_hindi` function connects retrieval and generation.

The process is:

```text
Question
   ↓
Generate Question Embedding
   ↓
Search FAISS Index
   ↓
Retrieve Relevant Hindi Context
   ↓
Insert Context + Question into RAG Prompt
   ↓
Send Prompt to Qwen
   ↓
Generate Hindi Answer
```

## Demonstration

The notebook demonstrates the complete RAG pipeline using two example Hindi questions.

For each question, the notebook displays:

* The user question
* Retrieved relevant context
* Generated Hindi answer

This demonstrates how external dataset information is retrieved and used by the LLM to produce a context-aware response.

## Technologies Used

| Technology            | Purpose                        |
| --------------------- | ------------------------------ |
| Python                | Implementation                 |
| Hugging Face Datasets | Dataset loading                |
| Transformers          | LLM and tokenizer              |
| Qwen                  | Hindi text generation          |
| Sentence Transformers | Text embeddings                |
| FAISS                 | Vector similarity search       |
| PyTorch               | Model execution                |
| Pandas                | Data inspection and processing |
| BitsAndBytes          | Quantization configuration     |

## Important Note

A **4-bit quantization configuration using `BitsAndBytesConfig`** is defined in the notebook, but it is **not currently applied when loading the Qwen model**.

Therefore, the current implementation does not use 4-bit quantized model weights.

## How to Run

1. Open the notebook in **Google Colab**.
2. Install the required dependencies.
3. Log in to Hugging Face if required.
4. Run the cells sequentially.
5. Load and inspect the dataset.
6. Load the Qwen model and tokenizer.
7. Initialize the `HindiRetriever`.
8. Build the FAISS index.
9. Run `answer_in_hindi()` with a Hindi question.
10. View the retrieved context and generated answer.

## Key Methods and APIs

The notebook uses:

* `datasets.load_dataset()`
* `transformers.AutoModelForCausalLM`
* `transformers.AutoTokenizer`
* `tokenizer.apply_chat_template()`
* `model.generate()`
* `SentenceTransformer.encode()`
* `faiss.IndexFlatL2`
* `index.add()`
* `index.search()`
* PyTorch CUDA/CPU device management
* Python string processing for prompt formatting and response cleaning

## Objective

The main objective of this project is to demonstrate how **RAG can improve Hindi question answering by combining semantic retrieval with a language model**.

The project provides a simple end-to-end implementation of:

**Hindi Data → Embeddings → FAISS Retrieval → Context → Qwen LLM → Hindi Answer**
