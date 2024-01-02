import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
 
import pinecone
import tempfile
import os
import time

os.environ['OPENAI_API_KEY'] = st.secrets['openai_key']
#pinecone.init(api_key='d49011f7-9f67-4f17-a092-38a22e0cbe83', environment='gcp-starter')

pinecone_index_name = "chatbot"

def gen_response(user_input):
    st_status.update(label="Generating Response", state="running", expanded=False)
    output = st.session_state.sess_chain({'question': user_input})
    #st.write(output)

    page_str = ''
    if not output['source_documents'] == []:
        pages = []
        primary = True
        for document in output['source_documents']:
            page = int(document.metadata['page'])+1
            if primary:
               pages.append(f':orange[{str(page)}]')
               primary = False 
            else: 
               pages.append(f':blue[{str(page)}]')  # Convert page numbers to strings for joining

        page_list = ", ".join(pages)  # Combine pages with commas
        
        if len(pages) == 1:
            page_str = f"**Source Page: {page_list}**"
        else:
            page_str = f"**Source Pages: {page_list}**"
        
    answer = output['answer']
    if "Call Us" in answer:
       answer = output['answer']
    else:
       answer = output['answer'] + "  \n\n  :green[" + page_str + "]"

    st_status.update(label="Response Generated", state="complete", expanded=False)
    return answer

def wait_on_index(index: str):
  """
  Takes the name of the index to wait for and blocks until it's available and ready.
  """
  ready = False
  while not ready: 
    try:
      desc = pinecone.describe_index(index)
      if desc[7]['ready']:
        return True
    except: # pinecone.core.client.exceptions.NotFoundException:
      # NotFoundException means the index is created yet.
      pass
    time.sleep(5)    

st.header("Kezava **DocuBot**", divider='rainbow')

if "sess_file_name" not in st.session_state:
    st.session_state.sess_file_name = ""

if "sess_is_new" not in st.session_state:    
    st.session_state.sess_is_new = True

#pinecone.init(api_key='d49011f7-9f67-4f17-a092-38a22e0cbe83', environment='gcp-starter')

uploaded_file = st.sidebar.file_uploader("Upload document", type="pdf")

if uploaded_file is None:
  st.info("""Upload document for analysis""")

elif uploaded_file :

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:

        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    if st.session_state.sess_file_name == uploaded_file.name:
        st.session_state.sess_is_new = False
    else:
        st.session_state.sess_file_name = uploaded_file.name
        st.session_state.sess_is_new = True

    st_status = st.sidebar.status('Processing',expanded=False)    

    if st.session_state.sess_is_new:
        st_status.update(label="Extracting PDF Content", state="running", expanded=False)
        loader = PyPDFLoader(file_path=tmp_file_path)
        st_status.update(label="Analyzing Pages", state="running", expanded=False)  
        data = loader.load_and_split()
        st_status.update(label="Document Loaded", state="running", expanded=False)

    tmp_file.close()
    
    if st.session_state.sess_is_new:   
        st.session_state.messages = []

        st_status.update(label="Processing Vectors", state="running", expanded=False)
        pinecone.init(api_key='35b8d2af-51b8-47fa-a118-0916d5851866', environment='gcp-starter')
        try:
          pinecone.delete_index(pinecone_index_name)        
        except:
          pass

        time.sleep(2)        
        st_status.update(label="Initializing Vector Store", state="running", expanded=False)
        pinecone.create_index(pinecone_index_name, dimension=1536)   
        st_status.update(label="Creating Vector Embeddings", state="running", expanded=False)
        wait_on_index(pinecone_index_name)

        vectorstore = Pinecone.from_documents(documents=data, embedding=OpenAIEmbeddings(), index_name=pinecone_index_name)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
        st.session_state.sess_retriever = retriever    
        
    else:
        st_status.update(label="Retrieving Vectors", state="running", expanded=False)
        
        retriever = st.session_state.sess_retriever 
    #    vectorstore = PineconeVectorStore(api_key="35b8d2af-51b8-47fa-a118-0916d5851866", index_name="chatbot")

    st_status.write('Done')
    st_status.update(label="Processing complete!", state="complete", expanded=False)
    if st.session_state.sess_is_new:  
    
        llm = ChatOpenAI(model_name="gpt-4-1106-preview", temperature=0)
        st.session_state.sess_llm = llm

        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
        st.session_state.sess_memory = memory

        system_prompt = PromptTemplate(input_variables=["context", "question"],
                                             template="""You are a very helpful research assistant who answers accurately and in great detail from the given context. Try to answer to the best of your ability from the given context which is your input document. You may be asked to summarize the document or explain the contents of the document. Be ready to answer these questions. If you are really unable to answer from the context, say this exactly: " I don't know the answer to the question. Click on the Call Us link to be connected to a coordinator " followed by a clickable link for tel:+18882547623 with caption "Call us"\n\nContext: {context}\n\nQuestion: {question}\n\n""")
    
        chain = ConversationalRetrievalChain.from_llm(
            llm,retriever=retriever,memory=memory,return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": system_prompt}
            )
        st.session_state.sess_chain = chain
    else:
        llm = st.session_state.sess_llm 
        memory = st.session_state.sess_memory 
        chain = st.session_state.sess_chain 

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_input := st.chat_input("What would you like to know about the document?"):
    
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
    
        response = gen_response(user_input)

        st.session_state.sess_llm  = llm
        st.session_state.sess_memory  = memory 
        st.session_state.sess_chain = chain
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
