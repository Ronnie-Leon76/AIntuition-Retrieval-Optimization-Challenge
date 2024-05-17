# ‘AIntuition’: Retrieval Augmented Generation (RAG) for Public Services and Administration Tasks by ITU Zindi Challenge

- The objective of this challenge is to encourage the emergence of better open-source RAG tools and resources that could be leveraged by public institutions to support public administration services and tasks.

## Requirements
- Python 3.10
- Dependencies listed in `requirements.txt`

## Technologies Used
- LangChain
- Ollama
- FAISS Vector Database
- HuggingFace Embeddings


## Setup
- To run this solution, follow the following instructions:

1. Create a new virtual environment:
    - If using vanilla [virtualenv](https://virtualenv.pypa.io/en/latest/), run `virtualenv venv` and then activate the virtual environment with the following commands for linux `source venv/bin/activate` and for a Windows machine `<venv>\Scripts\activate.bat`
    - If using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.org/en/latest/), run `mkvirtualenv venv`
    - Before installing the project required dependencies, ensure you run the following commands to install some global dependencies required by the project:
      ```bash
      sudo apt install tesseract-ocr -y
      ```
      ```bash
      sudo apt install libtesseract-dev -y
      ```
      ```bash
      sudo apt-get install poppler-utils -y
      ```
    - Also, ensure you install ollama and pull llama2-7B model from Ollama hub using the install instructions shown below.

    - For Linux use the following:
      ```bash
      curl -fsSL https://ollama.com/install.sh | sh
      ```
    - For Windows and MacOS download the installer from the following [link](https://ollama.com/download)

2. Install the requirements with `pip install -r requirements.txt`
3. If you would like to recreate the FAISS Vector Database rename the current Vector Database variable in `main.py` or the folder with the name vectorestore to vectorstore_ If there is no vector database recreated, the process will start the ingestion process.
4. Once you have everything setup in your environment, run `python main.py ./Test_Documents/ Test.csv`
5. To checkout the various evaluations done on a base RAG pipeline using a base retriever i.e. FAISS retriever, ensemble retriever (FAISS retriever + BM25 retriever), and contextual compression retrievers checkout the Evaluation directory in the `Evaluate_the_retrieval_system.ipynb` notebook. In order to run the notebook you'll need an OpenAI API Key. Set it in the .env