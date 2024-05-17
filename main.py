import argparse
import os, sys
import warnings
import pandas as pd
import pickle
from tqdm.auto import tqdm
from langchain.schema.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_community.chat_models import ChatOllama
from Ingestion.ingest import extract_text_and_metadata_from_pdf_document, extract_text_and_metadata_from_docx_document


warnings.filterwarnings("ignore")

llm = ChatOllama(model="llama2")


BATCH_SIZE = 32
DB_FAISS_PATH = 'vectorstore/db_faiss'


# Helper function for printing docs
def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


def load_embedding_model():
    return HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1", model_kwargs={'device': 'cpu'})

def create_vector_db(documents, embedding_model):
    # Create a vector store
    db = FAISS.from_documents(documents, embedding_model)
    db.save_local(DB_FAISS_PATH)

def initialize_bm25_retriever(documents):
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5
    # Save bm25_retriever as a pickle file
    with open('bm25_retriever.pkl', 'wb') as f:
        pickle.dump(bm25_retriever, f)
    return bm25_retriever

def load_bm25_retriever():
    with open('bm25_retriever.pkl', 'rb') as f:
        bm25_retriever = pickle.load(f)
    return bm25_retriever



def main():
    parser = argparse.ArgumentParser(description='Partition PDF or DOCX files in a directory')
    parser.add_argument('dir_path', type=str, help='Path to the directory containing PDF or DOCX files')
    parser.add_argument('csv_path', type=str, help='Path to the Test.csv file')

    args = parser.parse_args()

    dir_path = args.dir_path
    csv_path = args.csv_path

    if not os.path.exists(dir_path):
        print(f"Test Documents Directory path {dir_path} does not exist")
        sys.exit(1)

    if not os.path.exists(csv_path):
        print(f"Test CSV path {csv_path} does not exist")
        sys.exit(1)

    bm25_retriever = None

    try:
        embedding_model = load_embedding_model()
        if not os.path.exists(DB_FAISS_PATH):
            pdf_files = [f for f in os.listdir(dir_path) if f.endswith('.pdf')]
            docx_files = [f for f in os.listdir(dir_path) if f.endswith('.docx')]

            documents = []

            for pdf_file in tqdm(pdf_files, desc='Processing PDF files'):
                pdf_path = os.path.join(dir_path, pdf_file)
                try:
                    df = extract_text_and_metadata_from_pdf_document(pdf_path)
                    print(f"Extracted text and metadata from {pdf_file}")
                    for index, row in tqdm(df.iterrows(), total=len(df), desc='Processing rows'):
                        file_name = row['Filename']
                        text = row['Text']
                        page_number = row['Page_Number']
                        document = Document(
                            page_content=text,
                            metadata = {
                                'id': str(index) + '_' + file_name + '_' + str(page_number),
                                'type': 'text',
                                'filename': file_name,
                                'page_number': page_number
                            }
                        )
                        documents.append(document)
                except Exception as e:
                    print(f"Error processing {pdf_file}: {str(e)}")

            for docx_file in tqdm(docx_files, desc='Processing DOCX files'):
                docx_path = os.path.join(dir_path, docx_file)
                try:
                    df = extract_text_and_metadata_from_docx_document(docx_path)
                    print(f"Extracted text and metadata from {docx_file}")
                    for index, row in tqdm(df.iterrows(), total=len(df), desc='Processing rows'):
                        parent_id = row['Parent_Id']
                        file_name = row['Filename']
                        text = row['Text']
                        page_number = row['Page_Number']
                        document = Document(
                            page_content=text,
                            metadata = {
                                'id': str(index) + '_' + str(parent_id) + '_' + file_name + '_' + str(page_number),
                                'type': 'text',
                                'filename': file_name,
                                'page_number': page_number
                            }
                        )
                        documents.append(document)
                except Exception as e:
                    print(f"Error processing {docx_file}: {str(e)}")

            create_vector_db(documents, embedding_model)
            bm25_retriever = initialize_bm25_retriever(documents)
        
        # Load the FAISS vector store
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        faiss_retriever = db.as_retriever()
        print("FAISS Loaded")
        # Load the BM25 Retriever if it does not exist
        if not bm25_retriever:
            bm25_retriever = load_bm25_retriever()
        print("BM25 Retriever loaded")
        # Create an ensemble retriever with the BM25 and FAISS retrievers
        # ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

        # compressor = LLMChainExtractor.from_llm(llm)

        # embeddings_filter = EmbeddingsFilter(embeddings=embedding_model, similarity_threshold=0.76)

        # compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble_retriever)

        test_df = pd.read_csv(csv_path)
        # Iterate through the rows in test_df Dataframe extracting the values in Query text and use it to query the Qdrant collection
        for i, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Processing queries'):
            query_text = row['Query text']
            search_results = db.similarity_search_with_score(query_text, top_k=5)
            
            # Check if we have fewer than 5 results
            num_results = min(len(search_results), 5)
        
            # Add the document contents of the top search results to the test_df DataFrame
            for j in range(num_results):
                result = search_results[j]
                doc_content = result[0].page_content
                test_df.at[i, f'Output_{j+1}'] = doc_content
            
            # Fill remaining Output columns with empty strings if there are fewer than 5 results
            for j in range(num_results, 5):
                test_df.at[i, f'Output_{j+1}'] = ""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        file_name = f"output_{timestamp}.csv"
        test_df.to_csv(file_name)
        print(f"Output saved to {file_name}")

            
    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
