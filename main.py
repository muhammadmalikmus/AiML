# noqa E501
import streamlit as st

st.set_page_config(page_title="AI Prompt Engineer", page_icon="🦫", layout="wide")

from models import call_zero_shot_pipeline, convert_df  # noqa E402

df = None

if "prompt_count" not in st.session_state:
    st.session_state.prompt_count = 0

if "prompt_0" not in st.session_state:
    st.session_state["prompt_0"] = ""

if "demonstration_count" not in st.session_state:
    st.session_state.demonstration_count = 0

if "question_0" not in st.session_state:
    st.session_state["question_0"] = ""

if "answer_0" not in st.session_state:
    st.session_state["answer_0"] = ""

tab1, tab2, tab3 = st.tabs(["Home", "API Keys", "How to Use"])

with tab3:
    st.header("What it do")
    st.markdown(
        """This is an automated pipeline for testing ChatGPT prompts using OpenAI and Promptlayer.
    It's largely based on the zero shot example in this phenomenal
    [blog post]
    (https://mitchellh.com/writing/prompt-engineering-vs-blind-prompting#user-content-fnref-4).
    It runs each question/answer pair against each prompt candidate,
    generates a score for each output based on
    the ground-truth answer, and tracks each template
    in Promptlayer. It's designed as a zero shot system, but you
    can include examples in your prompt candidates to test our few shot candidates.
    Supporting few shot behavior is probably going to be added
    to this anyway though. Happy prompt engineering!"""
    )

    st.header("How it do")
    st.markdown(
        """
    1. Enter your [OpenAI api key](https://platform.openai.com/account/api-keys) and
    [Promptlayer api key](https://promptlayer.com/home)
    in the api keys tab.
    2. If this is your first time enter a few prompt candidates and give them names to
    track them in
    Promptlayer. For example, stealing from the inspiration for this work, you could use
    (the names can be whatever you want)
        - *Identify the date or day mentioned in the given
        text and provide it as the output*
        - *Identify the date or day mentioned in the given event description*
    3. Add Q/A pairs. E.g.,
        - Mother's day brunch on 5/14. 5/14
        - Succession finale two sundays from now. two sundays from now
    4. Run it
    5. If you've used it before and already have prompts published on
    the Promptlayer registry
    you can just provide the template
    name you want to evaluate."""
    )

    st.header("Tips")
    st.markdown(
        """
    - Scoring can be a bit tricky because it's evaluating for an exact match
    rather than semantic similarity.
    - It works best for questions with a specific, single ground truth answer
    although you could try passing multiple potential answers.
    """
    )

with tab2:
    st.header("API Keys")
    st.text_input(
        "OpenAI API Key",
        key="openai_api_key",
        placeholder="OpenAI API Key*",
        type="password",
        label_visibility="collapsed",
    )
    st.text_input(
        "Promptlayer API Key",
        key="promptlayer_api_key",
        placeholder="Promptlayer API Key*",
        type="password",
        label_visibility="collapsed",
    )

with tab1:
    st.header("AI Prompt Engineer")
    col_1, col_2, col_3, col_4 = st.columns(4, gap="medium")

    with st.form("prompt_test"):

        for i in range(0, st.session_state.prompt_count + 1):
            with col_1:
                st.text_area(
                    f"Prompt Candidate {i+1} Template name",
                    key=f"prompt_name_{i}",
                    placeholder=f"Prompt Candidate {i+1} Template Name",
                    label_visibility="collapsed",
                )
            with col_2:
                st.text_area(
                    f"Prompt Candidate {i + 1}",
                    help="no help",
                    key=f"prompt_{i}",
                    placeholder=f"Prompt Candidate {i+1}",
                    label_visibility="collapsed",
                )
        for i in range(0, st.session_state.demonstration_count + 1):
            with col_3:
                st.text_area(
                    f"Question {i + 1}",
                    help="no help",
                    key=f"question_{i}",
                    placeholder=f"Question {i+1}",
                    label_visibility="collapsed",
                )
            with col_4:
                st.text_area(
                    f"Answer {i + 1}",
                    help="no help",
                    key=f"answer_{i}",
                    placeholder=f"Answer {i+1}",
                    label_visibility="collapsed",
                )

        if st.form_submit_button():
            df = call_zero_shot_pipeline(dict(st.session_state))
            st.dataframe(df)

    if df is not None:
        st.download_button(
            label="Download results as csv",
            data=convert_df(df),
            file_name="results.csv",
            mime="text/csv",
        )

    with col_1:
        if st.button("Add", type="primary", key="add_prompt"):
            st.session_state.prompt_count += 1
            st.session_state[f"prompt_{st.session_state.prompt_count}"] = ""
            st.experimental_rerun()

    with col_3:
        if st.button("Add", type="primary", key="add_demonstration"):
            st.session_state.demonstration_count += 1
            st.session_state[f"question_{st.session_state.demonstration_count}"] = ""
            st.experimental_rerun()
