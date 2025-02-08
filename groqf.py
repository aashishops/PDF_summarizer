import streamlit as st
from function_faiss import extract_pdf_text, split_text_into_chunks, create_knowledge_base, generate_response, read_excel_file, preprocess_excel_data

def main():
    st.set_page_config(page_title="Query the PDF and Excel Streamlit App")
    st.header("Query the PDF and Excel 💬")
    
    pdf_files = st.file_uploader(label="Upload your PDF files", type="pdf", accept_multiple_files=True)
    
    read_pdf_button = st.button("Read PDF")
    
    excel_file = st.file_uploader(label="Upload your Excel file", type="xlsx")
    
    knowledge_base = None
    
    if pdf_files and read_pdf_button:
        pdf_texts = extract_pdf_text(pdf_files)
        chunks = split_text_into_chunks(pdf_texts)
        print(chunks)
        knowledge_base = create_knowledge_base(chunks)
    
    if excel_file:
        excel_data = read_excel_file(excel_file)
        
        if excel_data is not None:
            excel_chunks = preprocess_excel_data(excel_data)
            knowledge_base = create_knowledge_base(excel_chunks)
    
    user_question = st.text_input(label="Ask your question")

    if user_question and knowledge_base:
        print(user_question)
        response, source = generate_response(user_question, knowledge_base)
        st.write("Answer:", response)
        st.write("Source:", ', '.join(str(s) for s in source))

if __name__ == '__main__':
    main()
