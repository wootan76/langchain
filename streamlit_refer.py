import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# from streamlit_chat import message
from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory

def main():
    st.set_page_config(
    page_title="DirChat",   # 웹페이지 title 설정
    page_icon=":books:")    # 웹페이지 아이콘 모양 설정

    st.title("_Private Data :red[QA Chat]_ :books:")   # 웹페이지 제목 설정 (_로 감싸면 기울기 설정, red[]는 빨간색으로 설정, books는 아이콘모양양)

    if "conversation" not in st.session_state:  # conversation을 session_state.conversation 변수에 저장
        st.session_state.conversation = None    # session_state.conversation이라는 변수를 사용위해서 이러한 정의가 필요.

    if "chat_history" not in st.session_state:  # chat_history를 session_state.conversation변수에 저장
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
        
    if process:   # process 버튼을 눌렀을때 
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()   # key를 입력할때까지 잠시 대기
        files_text = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
     
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) 

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]  # 초기 인사말 설정

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])  # 질문을 할때 마다 그 역활에 따른 아이콘과 질문 하기 위한 창을 대기 시킨다.

    history = StreamlitChatMessageHistory(key="chat_messages")  # 질문이 메모리에 저장되어 답변하기 위해서 사용됨

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."):   # chat_input : 질문 창 - 질문을 입력 하면~
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):    # 답변하기 위한 부분
            chain = st.session_state.conversation

            with st.spinner("Thinking..."):   # 로딩중..
                result = chain({"question": query})   # chain에 question을 넣어 답변을 한 결과를 result에 저장
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']  # 답변한 결과를 chat_history에 저장
                response = result['answer']   # answer 부분을 response에 저장
                source_documents = result['source_documents']   # 참고한 source_documents를 source_documents에 저장

                st.markdown(response)
                with st.expander("참고 문서 확인"):   # expander : 참고문서를 접었다 펴서 확인 할 수 있도록 설정
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content)
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text):   # 토큰갯수를 기준으로 text를 split해주기 위한 함수.
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs):   # 업로드 된 파일들을 text로 변경하는 함수.

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())  # 빈파일을 열어서 
            logger.info(f"Uploaded {file_name}")  # 업로드된 파일을 저장
            
        if '.pdf' in doc.name:
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name:
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name:
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask",
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr', vervose = True), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),  # 이전 대화 기억
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    main()
