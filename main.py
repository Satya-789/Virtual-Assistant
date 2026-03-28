import streamlit as st
from rag import process_urls, generate_answer

st.set_page_config(page_title="AI Virtual Assistant", page_icon="🤖")

st.title("🤖 AI Virtual Assistant")

# Sidebar
st.sidebar.header("Enter URLs")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

process_btn = st.sidebar.button("Process URLs")

# Session state
if "processed" not in st.session_state:
    st.session_state.processed = False

status = st.empty()

# Process URLs
if process_btn:
    urls = [u for u in [url1, url2, url3] if u.strip()]

    if not urls:
        status.error("⚠️ Enter at least one URL")
    else:
        with st.spinner("Processing..."):
            for step in process_urls(urls):
                status.text(step)

        st.session_state.processed = True
        status.success("✅ Ready!")

# Query
query = st.text_input("Ask your question:")

# Answer
if query:
    if not st.session_state.processed:
        st.warning("⚠️ Process URLs first")
    else:
        with st.spinner("Thinking..."):
            try:
                answer, sources = generate_answer(query)

                st.subheader("Answer")
                st.write(answer)

                if sources:
                    st.subheader("Sources")
                    for s in sources.split("\n"):
                        if s.strip():
                            st.write(s)

            except Exception as e:
                st.error(str(e))
