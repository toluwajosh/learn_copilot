import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler

from agents.setup import get_agent
from agents.settings import PARAMS

LIBRARY = tuple(PARAMS.subjects.keys())


with st.sidebar:
    subject = st.selectbox(
        "Choose a subject",
        LIBRARY,
    )

    subject_request = st.text_input("Request a subject")
    st.button("Request")  # if button, do something


subject_params = PARAMS.subjects[subject]

agent = get_agent(
    subject,
    subject_params["collection_description"],
    subject_params["collection_path"],
    subject_params["persist_path"],
    PARAMS.persist,
    enable_search=PARAMS.enable_search,
)

st.title("🤖 Learn CoPilot")
st.write(
    "This is a learning chatbot that helps you to learn about various subjects directly from the source materials. You can choose from the different subjects from the sidebar. "
)
st.write(f"You have selected: {subject}")

if "messages" not in st.session_state:
    st.session_state["messages"] = {
        subject: [
            {
                "role": "assistant",
                "content": f"Hi, You can ask me any question about the {subject}. What would you like to learn about?",
            }
        ]
        for subject in LIBRARY
    }
if (
    not st.session_state.get("agents", False)
    or subject not in st.session_state["agents"]
):
    st.session_state["agents"] = {subject: agent}

for msg in st.session_state.messages[subject]:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages[subject].append(
        {"role": "user", "content": prompt}
    )
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        # response = agent.run(input=prompt, callbacks=[st_callback])
        response = st.session_state["agents"][subject].run(
            prompt + f" , in the {subject}", callbacks=[st_callback]
        )
        st.session_state.messages[subject].append(
            {"role": "assistant", "content": response}
        )
        st.write(response)
