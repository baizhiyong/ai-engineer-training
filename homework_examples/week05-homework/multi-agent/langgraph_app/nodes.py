from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage
from .state import AgentState


class AgentNodes:
    def __init__(self, mcp_tools: list):
        self.mcp_tools = {tool.name: tool for tool in mcp_tools}
        self.llm = ChatTongyi(model="qwen-plus")

    async def _call_mcp_tool(self, tool_name: str, **kwargs):
        if tool_name not in self.mcp_tools:
            raise ValueError(f"Tool '{tool_name}' not found.")
        tool = self.mcp_tools[tool_name]
        return await tool.ainvoke(kwargs)

    async def research_node(self, state: AgentState) -> dict:
        print("--- 节点: 研究 ---")
        prompt = await self._call_mcp_tool("get_prompt", agent_name="research")
        search_results = await self._call_mcp_tool("search", topic=state["topic"])
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"主题：{state['topic']}\n\n搜索结果：\n{search_results}")
        ]
        response = await self.llm.ainvoke(messages)
        report = response.content
        print("✅ 研究报告生成完毕。")
        return {"research_report": report, "log": state["log"] + [f"## 研究报告\n\n{report}"]}

    async def writing_node(self, state: AgentState) -> dict:
        print("--- 节点: 撰写 ---")
        prompt_template = await self._call_mcp_tool("get_prompt", agent_name="write")
        prompt = prompt_template.format(style=state["style"], length=state["length"])
        messages = [SystemMessage(content=prompt), HumanMessage(content=state["research_report"])]
        response = await self.llm.ainvoke(messages)
        draft = response.content
        print("✅ 文章初稿完成。")
        return {"draft": draft, "log": state["log"] + [f"## 文章初稿\n\n{draft}"]}

    async def review_node(self, state: AgentState) -> dict:
        print("--- 节点: 审核 ---")
        prompt = await self._call_mcp_tool("get_prompt", agent_name="review")
        messages = [SystemMessage(content=prompt), HumanMessage(content=state["draft"])]
        response = await self.llm.ainvoke(messages)
        suggestions = response.content
        print("✅ 审核完成。")
        return {"review_suggestions": suggestions, "log": state["log"] + [f"## 审核建议\n\n{suggestions}"]}

    async def polishing_node(self, state: AgentState) -> dict:
        print("--- 节点: 润色 ---")
        prompt = await self._call_mcp_tool("get_prompt", agent_name="polish")
        user_input = f"文章初稿：\n\n{state['draft']}\n\n审核建议：\n\n{state['review_suggestions']}"
        messages = [SystemMessage(content=prompt), HumanMessage(content=user_input)]
        response = await self.llm.ainvoke(messages)
        final_article = response.content
        print("✅ 最终稿件完成！")
        return {"final_article": final_article}
    