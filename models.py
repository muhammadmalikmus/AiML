from threading import Thread

import promptlayer
import streamlit as st
from langchain.chains import LLMChain
from langchain.llms import OpenAI, PromptLayerOpenAIChat
from langchain.prompts import PromptTemplate
from pandas import DataFrame
from promptlayer import prompts

from chains import PromptLayerLLMChain, PromptLayerSequentialChain
from track_prompts import write_to_prompt_layer


def load_evaluator_chain():
    template = """Give a score 0-100 to the AI output for how exactly it matches the ground truth.
    Do not penalize exact matches. Ground truth: {answer} AI: {ai_answer} Score:"""
    prompt = PromptTemplate(
        input_variables=["answer", "ai_answer"],
        template=template,
    )
    llm = OpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=st.session_state["openai_api_key"],
    )

    return LLMChain(llm=llm, prompt=prompt, output_key="score")


def load_zero_shot_chain(existing_prompt_template: PromptTemplate = None):
    template = """{prompt_candidate}. Q: {input} A:"""

    prompt = PromptTemplate(
        input_variables=["prompt_candidate", "input"], template=template
    )
    if existing_prompt_template:
        prompt = existing_prompt_template

    llm = PromptLayerOpenAIChat(
        openai_api_key=st.session_state["openai_api_key"],
        model_name="gpt-3.5-turbo",
        temperature=0,
        return_pl_id=True,
    )

    return PromptLayerLLMChain(llm=llm, prompt=prompt, output_key="ai_answer")


def load_zero_shot_pipeline(existing_prompt_template: PromptTemplate = None):
    zero_shot = load_zero_shot_chain(existing_prompt_template)
    evaluator = load_evaluator_chain()

    return PromptLayerSequentialChain(
        chains=[zero_shot, evaluator],
        input_variables=["input", "answer"]
        if existing_prompt_template
        else ["prompt_candidate", "input", "answer"],
        output_variables=["ai_answer", "score"],
    )


def call_zero_shot_pipeline(state: dict) -> DataFrame:
    promptlayer.api_key = st.session_state["promptlayer_api_key"]
    results = []
    for i in range(state["prompt_count"] + 1):
        existing_prompt_template = None
        if state[f"prompt_name_{i}"] and not state[f"prompt_{i}"]:
            existing_prompt_template = prompts.get(
                prompt_name=state[f"prompt_name_{i}"], langchain=True
            )
        zero_shot_pipeline = load_zero_shot_pipeline(existing_prompt_template)
        for j in range(state["demonstration_count"] + 1):
            output = zero_shot_pipeline(
                {
                    "prompt_candidate": state[f"prompt_{i}"],
                    "input": state[f"question_{j}"],
                    "answer": state[f"answer_{j}"],
                }
            )
            output.update({"prompt_template_name": state[f"prompt_name_{i}"]})
            if existing_prompt_template:
                output.update({"prompt_candidate": existing_prompt_template.template})
            results.append(output)
    df = DataFrame(results)
    thread = Thread(target=write_to_prompt_layer(df))
    thread.start()

    return df


def convert_df(df: DataFrame) -> str:
    return df.to_csv().encode("utf-8")
