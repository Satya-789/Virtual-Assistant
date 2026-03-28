import streamlit as st
from rag import process_urls, generate_answer

st.title("Real Estate Research Tool 🏠")

# Sidebar inputs
st.sidebar.header("Enter URLs")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

process_url_button = st.sidebar.button("Process URLs")

# Session state to track processing
if "processed" not in st.session_state:
    st.session_state.processed = False

# Placeholder for status messages
status_placeholder = st.empty()

# Process URLs
if process_url_button:
    urls = [url for url in (url1, url2, url3) if url.strip() != ""]

    if len(urls) == 0:
        status_placeholder.error("⚠️ Please provide at least one valid URL")
    else:
        with st.spinner("Processing URLs..."):
            for status in process_urls(urls):
                status_placeholder.text(status)

        st.session_state.processed = True
        status_placeholder.success("✅ Processing completed!")

# Main input (NOT inside placeholder)
query = st.text_input("Ask a question:")

# Generate answer
if query:
    if not st.session_state.processed:
        st.warning("⚠️ Please process URLs first")
    else:
        with st.spinner("Thinking... 🤔"):
            try:
                answer, sources = generate_answer(query)

                st.header("Answer:")
                st.write(answer)

                if sources:
                    st.subheader("Sources:")
                    for source in sources.split("\n"):
                        if source.strip():
                            st.write(source)

            except Exception as e:
                st.error(f"Error: {str(e)}")
