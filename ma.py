import streamlit as st

st.title("RAG with Human Feedback Project")

n=10
idx = 0

if 'click_'+str(idx) not in st.session_state:
    st.session_state['click_'+str(idx)] = None

def form(idx):
    st.session_state['click_'+str(idx)] = True

st.session_state

while idx < 10:
    with st.form(f"my_form_{idx}"):
        genre = st.radio(
            f"Answer : \n {idx}",
            [":thumbsup:", ":thumbsdown:"],
            horizontal=True,index=None
        )
        click = st.form_submit_button("Feedback")
        if click and st.session_state['genre_'+str(idx)]==":thumbsup:":
            break
        if click and st.session_state['genre_'+str(idx)]==":thumbsdown:":
            idx = idx + 1
    st.session_state