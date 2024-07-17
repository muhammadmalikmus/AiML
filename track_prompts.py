from re import match

from langchain.prompts import PromptTemplate
from pandas import DataFrame
from promptlayer import prompts, track


def write_to_prompt_layer(df: DataFrame):
    df.apply(track_prompt_run, axis=1)


def track_prompt_run(prompt_run):
    success = track.prompt(
        request_id=str(prompt_run["pl_id"]),
        prompt_name=str(prompt_run["prompt_template_name"]),
        prompt_input_variables={"input": prompt_run["input"]},
    )
    # assuming that it fails becasue the template doesn't exist yet in promptlayer
    if not success:
        template = prompt_run["prompt_candidate"] + """ Q: {input} A:"""
        prompts.publish(
            prompt_name=str(prompt_run["prompt_template_name"]),
            prompt_template=PromptTemplate(
                input_variables=["input"], template=template
            ),
        )
        track.prompt(
            request_id=str(prompt_run["pl_id"]),
            prompt_name=str(prompt_run["prompt_template_name"]),
            prompt_input_variables={"input": prompt_run["input"]},
        )
    track.score(
        request_id=str(prompt_run["pl_id"]),
        score=int(match(r"\d+", prompt_run["score"])[0]),
    )
