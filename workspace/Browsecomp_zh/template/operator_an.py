# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 19:46 PM
# @Author  : didi
# @Desc    : action nodes for operator

from pydantic import BaseModel, Field


class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")

class ScEnsembleOp(BaseModel):
    thought: str = Field(default="", description="The thought of the most consistent solution.")
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")

class AnswerGenerateOp(BaseModel):
    thought: str = Field(default="", description="The step by step thinking process")
    answer: str = Field(default="", description="The final answer to the question")

class AgentGenerateOp(BaseModel):
    thought: str = Field(default="", description="The step by step thinking process")
    tool_call: str = Field(default="", description="{\"name\": <function-name>, \"args\": <args-json-object>}")

class ResumOp(BaseModel):
    Summary: str = Field(default="", description="The compressed history of the current progress over the problem.")

class AgentFoldDecisionOp(BaseModel):
    should_fold: bool = Field(default=False, description="Whether to fold the context based on current history and question")
    fold_level: str = Field(default="auto", description="The folding level: 'fine-grained' for detailed condensation, 'deep' for high-level integration, or 'auto' for automatic decision")
    reasoning: str = Field(default="", description="The reasoning process for the folding decision")

class AgentFoldOp(BaseModel):
    folded_context: str = Field(default="", description="The folded context after applying the folding operation")
    reasoning: str = Field(default="", description="The reasoning process for how the context was folded")