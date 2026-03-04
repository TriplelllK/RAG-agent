import streamlit as st
import os
from rag_core import VectorStore, rerank, make_answer_llm, DEFAULT_LLM_MODEL, validate_llm_config
from prompts import CITATION_FMT

DEFAULT_MODEL = DEFAULT_LLM_MODEL

st.set_page_config(page_title="RAG У-300", layout="wide")
st.title("RAG по У-300")

with st.expander("Настройки поиска"):
    top_k = st.slider("k из векторного поиска", 1, 10, 6)
    top_rerank = st.slider("k после rerank", 1, 10, 4)

if 'store' not in st.session_state:
    idx = os.path.join('storage', 'faiss.index')
    meta = os.path.join('storage', 'meta.json')
    if not (os.path.exists(idx) and os.path.exists(meta)):
        st.error("Нет индекса. Сначала запустите ingest.py")
        st.stop()
    st.session_state.store = VectorStore(idx, meta)

q = st.text_input("Ваш вопрос по документации:", "Какие уставки LIC-31050 и PDT-31016?")

if st.button("Ответить") and q.strip():
    ok, msg = validate_llm_config()
    if not ok:
        st.error(f"Проверьте LLM настройки: {msg}")
        st.stop()

    store = st.session_state.store
    raw = store.search(q, k=top_k)
    reranked = rerank(q, raw, top_k=top_rerank)
    top_score = reranked[0][1] if reranked else 0.0
    ctx = [x[0] for x in reranked]

    result = make_answer_llm(q, ctx, model=DEFAULT_MODEL, retrieval_score=top_score)

    st.markdown("## Ответ")
    st.write(result['answer'])
    st.markdown("## Цитаты")
    for c in result['citations']:
        st.write(CITATION_FMT.format(
            doc_name=c['doc_name'],
            page=c['page'],
            snippet=c['snippet']
        ))

    with st.expander("Контекст (фрагменты)"):
        for ch in ctx:
            st.markdown(f"**{ch.doc_name} — стр. {ch.page}**")
            st.text(ch.text)
with st.expander("Проверка структурных данных"):
    eq = st.text_input("Проверить оборудование:", "G-304")
    if st.button("Показать данные"):
        from rag_structured import load_norms, load_alarms, norms_by_equipment, alarms_by_equipment
        norms = norms_by_equipment(load_norms(), eq)
        alarms = alarms_by_equipment(load_alarms(), eq)
        st.markdown("### Нормы")
        for n in norms:
            st.text(f"{n.instrument}: {n.param} ({n.unit}) {n.range_min}-{n.range_max}, {n.work_min}-{n.work_max} [стр.{n.page}]")
        st.markdown("### Аварии")
        for a in alarms:
            st.text(f"{a.instrument}: {a.param} ({a.unit}), {a.setpoint}, {a.action}, {a.note} [стр.{a.page}]")
