# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 19:46 PM
# @Author  : didi
# @Desc    : action nodes for operator

from pydantic import BaseModel, Field

class ReActOp(BaseModel):
    thought: str = Field(default="", description="The step by step thinking process")
    action: str = Field(default="", description="The action to be taken")


class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")


class CodeGenerateOp(BaseModel):
    code: str = Field(default="", description="Your complete code solution for this problem")


class AnswerGenerateOp(BaseModel):
    thought: str = Field(default="", description="The step by step thinking process")
    answer: str = Field(default="", description="The final answer to the question")


class FormatOp(BaseModel):
    solution: str = Field(default="", description="Your formatted answer for this problem")


class ScEnsembleOp(BaseModel):
    thought: str = Field(default="", description="The thought of the most consistent solution.")
    solution_letter: str = Field(default="", description="The letter of most consistent solution.")

class ResumOp(BaseModel):
    Summary: str = Field(default="", description="The compressed history of the current progress over the problem.")

class ReflectionTestOp(BaseModel):
    reflection_and_solution: str = Field(
        default="", description="Corrective solution for code execution errors or test case failures"
    )


class MdEnsembleOp(BaseModel):
    thought: str = Field(default="", description="Step-by-step analysis of the solutions to determine the best one.")
    solution_letter: str = Field(default="", description="The letter of the chosen best solution (only one letter).")


class ReviewOp(BaseModel):
    review_result: bool = Field(
        default=False,
        description="The Review Result (Bool). If you think this solution looks good for you, return 'true'; If not, return 'false'",
    )
    feedback: str = Field(
        default="",
        description="Your FeedBack for this problem based on the criteria. If the review result is true, you can put it 'nothing here'.",
    )


class ReviseOp(BaseModel):
    solution: str = Field(default="", description="Based on the feedback, revised solution for this problem")


class AgentFoldDecisionOp(BaseModel):
    should_fold: bool = Field(default=False, description="Whether to fold the context based on current history and question")
    fold_level: str = Field(default="auto", description="The folding level: 'fine-grained' for detailed condensation, 'deep' for high-level integration, or 'auto' for automatic decision")
    reasoning: str = Field(default="", description="The reasoning process for the folding decision")


class AgentFoldOp(BaseModel):
    folded_context: str = Field(default="", description="The folded context after applying the folding operation")
    reasoning: str = Field(default="", description="The reasoning process for how the context was folded")

class AgentGenerateOp(BaseModel):
    thought: str = Field(default="", description="The step by step thinking process")
    tool_call: str = Field(default="", description="{\"name\": <function-name>, \"args\": <args-json-object>}")

class AgentGenerateOpNative(BaseModel):
    think: str = Field(default="", description="The step by step thinking process")
    tool_call: str = Field(default="", description="{\"name\": <function-name>, \"args\": <args-json-object>}")
