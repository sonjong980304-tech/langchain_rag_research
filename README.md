# 리서치 리포트 분석 어시스턴트

증권사 리서치 리포트(PDF)를 업로드하면, 보고서 내용을 분석하는 동시에 **실시간 웹 검색**으로 동종 업계 기업들의 밸류에이션 지표(PER, PBR 등)를 자동으로 수집해 비교 분석을 제공하는 AI 어시스턴트입니다.

단순한 PDF 질의응답을 넘어, RAG와 외부 검색을 **병렬로 결합**해 보고서 내용과 시장 데이터를 함께 분석합니다.

---

## 주요 기능

- **PDF 리포트 분석**: 복수의 증권사 리포트를 동시에 업로드해 비교 분석 가능
- **실시간 피어 그룹 데이터 수집**: Tavily 검색으로 동종 업계 기업의 PER/PBR 등 밸류에이션 지표를 실시간 검색
- **병렬 처리**: PDF 벡터 검색과 웹 검색을 동시에 실행해 응답 속도 최적화
- **자동 검색 쿼리 생성**: 사용자의 한국어 질문을 최적의 영어 검색 쿼리로 자동 변환
- **출처 기반 답변**: 파일명, 페이지 번호, URL까지 명시한 근거 기반 분석

---

## 사용된 LangChain 핵심 기능

### 1. RunnableParallel — PDF 검색과 웹 검색 동시 실행

이 프로젝트의 핵심 기능입니다. `RunnableParallel`을 사용해 두 가지 데이터 소스를 **동시에** 처리합니다.

```python
chain = (
    RunnableParallel({
        "context":   retriever | format_docs,   # PDF 벡터 검색
        "peer_data": query_chain | search,       # 웹 검색 (Tavily)
        "question":  RunnablePassthrough(),      # 질문 그대로 전달
    })
    | prompt
    | llm
    | StrOutputParser()
)
```

- `context`: 업로드된 PDF에서 관련 청크 15개를 벡터 검색
- `peer_data`: 질문을 영어 검색 쿼리로 변환한 뒤 Tavily로 실시간 웹 검색
- 두 작업이 병렬로 실행되므로 순차 처리보다 빠릅니다

---

### 2. TavilySearchResults — 실시간 웹 검색 도구

LangChain의 `TavilySearchResults` 툴을 체인에 직접 연결해 외부 시장 데이터를 실시간으로 수집합니다.

```python
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults(k=3)
```

사용자가 "이 회사의 밸류에이션이 적정한가?"라고 물으면, 동종 업계 경쟁사 3~5개의 PER/PBR 데이터를 웹에서 검색해 보고서 내 수치와 직접 비교합니다.

---

### 3. 검색 쿼리 생성 서브체인 — 한국어 → 영어 쿼리 자동 변환

사용자의 한국어 질문을 그대로 검색에 쓰면 정확도가 떨어집니다. LLM을 활용한 별도의 서브체인으로 최적화된 영어 검색 쿼리를 자동 생성합니다.

```python
def create_search_query_chain(llm):
    query_prompt = ChatPromptTemplate.from_template(
        """Analyze the user's question and generate exactly one optimal search query in English 
        to retrieve external market data, such as peer group valuation metrics (PER, PBR, etc.)
        ...
        User Question: {question}
        Search Query:"""
    )
    return query_prompt | llm | StrOutputParser()
```

이 서브체인이 `RunnableParallel` 안에서 `peer_data` 경로의 첫 단계로 실행됩니다:
```
질문 → query_chain(한국어→영어 쿼리) → TavilySearch(실시간 검색) → peer_data
```

---

### 4. SemanticChunker — 의미 기반 문서 분할

일반적인 고정 크기 청킹(RecursiveCharacterTextSplitter 등) 대신 `SemanticChunker`를 사용해 **의미적으로 연관된 문장들이 같은 청크에 묶이도록** 분할합니다.

```python
from langchain_experimental.text_splitter import SemanticChunker

text_splitter = SemanticChunker(OpenAIEmbeddings())
split_documents = text_splitter.split_documents(docs)
```

재무 리포트는 "매출 전망", "리스크 요인", "밸류에이션 분석" 등 주제가 명확히 구분됩니다. 의미 단위로 청킹하면 질문에 관련 없는 내용이 검색 결과에 섞이는 것을 줄여 응답 품질이 높아집니다.

---

### 5. RunnablePassthrough — 입력값 그대로 전달

`RunnableParallel` 내에서 질문 텍스트를 변환 없이 프롬프트로 그대로 전달할 때 사용합니다.

```python
"question": RunnablePassthrough()
```

---

### 6. FAISS + OpenAIEmbeddings — 벡터 검색

복수의 PDF에서 추출한 청크를 `OpenAIEmbeddings`로 임베딩해 FAISS 벡터 DB에 저장합니다. 질문과 가장 유사한 상위 15개 청크를 검색합니다 (k=15).

```python
vectorstore = FAISS.from_documents(documents=all_docs, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
```

리서치 리포트는 분량이 길고 여러 파일을 동시에 다루므로, 이전 프로젝트(k=5)보다 더 많은 청크를 검색해 포괄적인 분석이 가능하도록 했습니다.

---

### 7. format_docs — 출처별 중복 병합

같은 파일에서 여러 청크가 검색될 경우, 같은 출처의 내용을 하나로 합쳐 프롬프트 토큰을 효율적으로 사용합니다.

```python
def format_docs(docs):
    unique_docs = {}
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if source not in unique_docs:
            unique_docs[source] = doc.page_content
        else:
            unique_docs[source] += f"\n\n[추가 내용]\n{doc.page_content}"
    return "\n\n".join([
        f"--- 리포트 출처: {src} ---\n{content}"
        for src, content in unique_docs.items()
    ])
```

---

### 8. load_prompt (YAML) — 프롬프트 외부 파일 관리

분석 프롬프트를 코드와 분리해 `prompts/pdf-rag2.yaml`에서 관리합니다. 프롬프트에는 `{context}`, `{peer_data}`, `{question}` 세 변수가 정의되어 있으며, 다음 구조의 출력을 강제합니다:

1. 핵심 요약 (2~3줄)
2. 주요 지표 테이블 (매출, 영업이익, 목표주가, Forward PER/PBR + 평균/표준편차)
3. 리포트 간 비교 분석
4. 피어 그룹 밸류에이션 비교 (경쟁사 3~5개)
5. 질문에 대한 상세 분석
6. 출처 목록 (파일명, 페이지, URL)

---

## 전체 데이터 흐름

```
사용자 질문
    │
    ├──▶ [PDF 경로] retriever(k=15) → format_docs → context
    │
    ├──▶ [웹 검색 경로] query_chain(한→영 변환) → TavilySearch(k=3) → peer_data
    │
    └──▶ [직접 전달] RunnablePassthrough() → question
         │
         ▼ (세 값이 모이면)
    pdf-rag2.yaml 프롬프트
         │
         ▼
    GPT-4.1-mini
         │
         ▼
    StrOutputParser → 스트리밍 출력
```

---

## 프로젝트 구조

```
.
├── main.py                # 메인 Streamlit 앱
├── prompts/
│   └── pdf-rag2.yaml      # 분석 프롬프트 템플릿 (YAML)
├── requirements.txt
└── pyproject.toml
```

---

## 실행 방법

```bash
pip install -r requirements.txt
```

`.env` 파일에 API 키를 설정합니다:
```
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
```

```bash
streamlit run main.py
```
