import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler

from agents.setup import get_agent
from agents.settings import PARAMS

LIBRARY = tuple(PARAMS.subjects.keys())
MODELS = tuple(PARAMS.models.keys())

if "enable_search" not in st.session_state:
    st.session_state["enable_search"] = {}

with st.sidebar:
    subject = st.selectbox(
        "Libraries",
        LIBRARY,
    )

    subject_params = PARAMS.subjects[subject]

    # enable_search = st.checkbox(
    #     "Enable Search",
    #     value=st.session_state.enable_search.get(subject, False),
    #     disabled=st.session_state.enable_search.get(subject, None) is not None,
    #     key=subject,
    # )
    # if enable_search:
    #     st.session_state.enable_search[subject] = True

    #     agent = get_agent(
    #         subject,
    #         subject_params["collection_description"],
    #         subject_params["collection_path"],
    #         subject_params["persist_path"],
    #         PARAMS.persist,
    #         enable_search=enable_search,
    #     )

    st.empty()
    st.markdown(
        """<hr style="height:60vh;border:none;" /> """,
        unsafe_allow_html=True,
    )
    with st.expander("Settings"):
        model = st.selectbox(
            "Model",
            MODELS,
            key="model",
        )

    agent = get_agent(
        subject,
        subject_params["collection_description"],
        subject_params["collection_path"],
        subject_params["persist_path"],
        PARAMS.persist,
        # enable_search=enable_search,
        model=model,
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
    if not st.session_state.enable_search.get(subject, False):
        st.session_state.enable_search[subject] = False

    st.session_state.messages[subject].append(
        {"role": "user", "content": prompt}
    )
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = st.session_state["agents"][subject].run(
            input=prompt,
            callbacks=[st_callback],
        )
        st.session_state.messages[subject].append(
            {"role": "assistant", "content": response}
        )
        st.write(response)
