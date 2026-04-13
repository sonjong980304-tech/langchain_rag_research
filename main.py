import streamlit as st
from langchain_teddynote.prompts import load_prompt
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableParallel

# API 키 정보 로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("Myproject_PDF_rag")

st.title("리서치 리포트 분석 어시스턴트💬")


# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")


# 파일의 임시폴더 생성(파일올렸을 때 그 파일을 감쌀 폴더를 만들어줌)
if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

# 처음 한 번만 실행하기 위한 코드
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 아무런 파일을 업로드 하지 않은 경우
if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    # 파일 업로더
    uploaded_files = st.file_uploader(
        "파일업로드", type=["PDF"], accept_multiple_files=True
    )
    selected_prompt = "prompts/pdf-rag2.yaml"

    selected_model = st.selectbox(
        "llm선택", ["gpt-4.1-mini", "gpt-5.4", "gpt-5.4-mini"], index=0
    )
    st.caption("made by sonjong")


# 이전대화 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


print_messages()


# 새로운 매세지를 추가
def add_message(role, messages):
    st.session_state["messages"].append(ChatMessage(role=role, content=messages))


# 파일을 캐시에 저장(시간이 오래걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="파일을 처리 중입니다...")
def embed_files(files):
    all_docs = []
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    for file in files:
        file_content = file.read()
        file_path = f"./.cache/files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        # 단계 1: 문서 로드(Load Documents)
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()

        # 단계 2: 문서 분할(Split Documents)
        text_splitter = SemanticChunker(
            OpenAIEmbeddings()
        )  # 텍스트를 의미 단위로 나누는 도구입니다. OpenAIEmbeddings를 사용하여 텍스트의 의미를 이해하고 적절한 크기로 분할합니다. 이도구를 사용하였을 때 목표주가를 제대로 뽑은 것을 발견함
        split_documents = text_splitter.split_documents(docs)
        for doc in docs:
            doc.metadata["source"] = file.name

        split_documents = text_splitter.split_documents(docs)
        all_docs.extend(split_documents)

    # 단계 3: 임베딩(Embedding) 생성
    embeddings = OpenAIEmbeddings()
    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=all_docs, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
    return retriever


# 검색어 추출을 위한 간단한 체인 정의
def create_search_query_chain(llm):
    query_prompt = ChatPromptTemplate.from_template(
        """Analyze the user's question and generate exactly one optimal search query in English 
        to retrieve external market data, such as peer group valuation metrics (PER, PBR, etc.), 
        for comparative analysis with the company mentioned in the report. 
        Output only the query text without any explanations or quotes.
        User Question: {question}
        Search Query:"""
    )
    return query_prompt | llm | StrOutputParser()


# 체인생성
def create_chain(retriever, model_name="gpt-4.1-mini"):

    prompt = load_prompt("prompts/pdf-rag2.yaml", encoding="utf-8")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    llm = ChatOpenAI(model_name=model_name, temperature=0)

    search = TavilySearchResults(k=3)
    query_chain = create_search_query_chain(llm)

    def format_docs(docs):
        # 파일별로 가장 관련성 높은 조각 1~2개만 남기고 중복 제거
        unique_docs = {}
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            if source not in unique_docs:
                unique_docs[source] = doc.page_content
            else:
                # 이미 해당 파일 내용이 있다면 뒤에 덧붙여서 풍부하게 만듦
                unique_docs[source] += f"\n\n[추가 내용]\n{doc.page_content}"

        return "\n\n".join(
            [
                f"--- 리포트 출처: {src} ---\n{content}"
                for src, content in unique_docs.items()
            ]
        )

    # 단계 8: 체인(Chain) 생성
    chain = (
        RunnableParallel(
            {
                "context": retriever | format_docs,
                "peer_data": query_chain | search,
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if uploaded_files:
    retriever = embed_files(uploaded_files)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain


# 경고메시지를 띄우기 위한 빈셀입력
Warning_msg = st.empty()

# 채팅 입력창 만들기 (이 줄이 빠져있을 가능성이 커요!)
user_input = st.chat_input("궁금한 내용을 물어보세요!")
# 만약에 사용자 입력이 들어오면
if user_input:

    chain = st.session_state["chain"]

    if chain is not None:
        # st.write(f"사용자 입력: {user_input}")
        st.chat_message("user").write(user_input)

        response = chain.stream(user_input)
        # ai_answer = chain.invoke({"question": user_input})
        with st.chat_message("assistant"):
            # 빈 공간(컨테니어)를 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)
        # st.chat_message("assistant").write(ai_answer)

        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        Warning_msg.error("파일을 업로드 해주세요")
