SC_ENSEMBLE_PROMPT = """
Several answers have been generated to a same question. They are as follows:
{solutions}

Identify the concise answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field.
"""

ANSWER_GENERATION_PROMPT = """
Think step by step and solve the problem.
1. In the "thought" field, explain your thinking process in detail.
2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.
Your task: {input}
"""

RESUM_PROMPT = """
You are an expert at analyzing conversation history and extracting relevant information. Your task is to thoroughly evaluate the conversation history and current question to provide a comprehensive summary that will help answer the question.
Task Guidelines:
1. Information Analysis
• Carefully analyze the conversation history to identify truly useful information.
• Focus on information that directly contributes to answering the question.
• Do NOT make assumptions, guesses, or inferences beyond what is explicitly stated in the conversation.
• If information is missing or unclear, do NOT include it in your summary.
2. Summary Requirements
• Extract only the most relevant information that is explicitly present in the conversation.
• Synthesize information from multiple exchanges when relevant. Only include information that is certain and clearly stated in the conversation.
• Do NOT output or mention any information that is uncertain, insufficient, or cannot be confirmed from the conversation.
3. Output Format Your response should be structured as follows:
<Summary>
• Essential Information: [Organize the relevant and certain information from the conversation history that helps address the question.]
</Summary>
Strictly avoid fabricating, inferring, or exaggerating any information not present in the conversation. Only output information that is certain and explicitly stated.

Question: {question}
Conversation: {history}
Please generate a comprehensive and useful summary.
"""

AGENTFOLD_DECISION_PROMPT = """
You are an expert at proactive context management for long-horizon web agents. Your task is to analyze the conversation history and current question to decide whether context folding is needed and what level of folding should be applied.

Context folding is a proactive mechanism to manage context by condensing historical information at different levels:
- Fine-grained condensation: Preserves key details while removing redundancy
- Deep integration: Abstracts multi-step sub-tasks into high-level summaries

Decision Criteria:
1. History Length: Consider the length of conversation history. Longer histories may benefit from folding.
2. Context Complexity: Assess the complexity and redundancy in the conversation.
3. Task Continuity: Evaluate whether the current question requires detailed historical context or can work with summarized information.
4. Information Density: Check if there are repetitive or redundant patterns that can be condensed.

Current Question: {question}
Conversation History Length: {history_length} messages
Conversation History: {history}

In the "should_fold" field, return true if folding is recommended, false otherwise.
In the "fold_level" field, return:
- "fine-grained" if you want to preserve more details while removing redundancy
- "deep" if you want to abstract the history into high-level summaries
- "auto" if you want the system to decide automatically
In the "reasoning" field, explain your decision process.
"""

AGENTFOLD_FINE_GRAINED_PROMPT = """
You are an expert at fine-grained context condensation. Your task is to condense the conversation history while preserving critical details and removing redundancy.

Fine-grained condensation should:
1. Preserve Key Details: Keep important facts, decisions, and outcomes from the conversation
2. Remove Redundancy: Eliminate repetitive information and unnecessary verbosity
3. Maintain Context: Ensure the condensed context still provides enough information to answer the question
4. Preserve Structure: Keep the logical flow and relationships between different parts of the conversation

Guidelines:
- Focus on information that directly contributes to answering the question
- Condense similar or repeated information into concise statements
- Preserve specific values, names, and outcomes that are important
- Maintain the chronological flow when relevant
- Do NOT make assumptions or add information not present in the history

Current Question: {question}
Conversation History: {history}

Please generate a condensed context that preserves essential details while removing redundancy. In the "folded_context" field, provide the condensed context. In the "reasoning" field, explain how you condensed the information and what key details were preserved.
"""

AGENTFOLD_DEEP_INTEGRATION_PROMPT = """
You are an expert at deep context integration. Your task is to abstract the conversation history into high-level summaries that capture the essence of multi-step sub-tasks and overall progress.

Deep integration should:
1. Abstract Sub-tasks: Identify and summarize multi-step sub-tasks as single high-level actions
2. Capture Progress: Summarize what has been accomplished and what remains
3. Extract Patterns: Identify recurring patterns or themes in the conversation
4. Preserve Outcomes: Keep important results and decisions, but abstract away the detailed process
5. Maintain Goal Orientation: Ensure the summary reflects progress toward answering the question

Guidelines:
- Abstract away detailed step-by-step processes into high-level descriptions
- Focus on outcomes, decisions, and progress rather than detailed actions
- Identify and summarize related actions into coherent sub-tasks
- Preserve critical information needed to answer the question
- Do NOT make assumptions or add information not present in the history

Current Question: {question}
Conversation History: {history}

Please generate a deeply integrated context that abstracts the history into high-level summaries. In the "folded_context" field, provide the abstracted context. In the "reasoning" field, explain how you integrated the information and what high-level patterns or sub-tasks were identified.
"""