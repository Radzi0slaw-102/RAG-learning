import re
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.llms import LLM, ChatMessage
from graph_store import GraphRAGStore

class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    llm: LLM
    
    def custom_query(self, query_str: str) -> str:
        # process all community summaries to generate answers to a specific query
        community_summaries = self.graph_store.get_community_summaries()
        community_answers = [
            self.generate_answer_from_summary(community_summary, query_str)
            for _, community_summary in community_summaries.items()
        ]
        return self.aggregate_answers(community_answers)
    
    def generate_answer_from_summary(self, community_summary, query):
        # generate an answer from a community summary based on a given query using LLM
        prompt = (
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query}"
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content="I need an answer based on above information"
            )
        ]
        response = self.llm.chat(messages)
        # clean response
        return re.sub(r"^assistant:\s*", "", str(response)).strip()

    def aggregate_answers(self, community_answers):
        # aggregate individual community answers into a final and coherent response
        prompt = "Combine the following intermediate answers into a final, concise response."
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content=f"Intermediate answers: {community_answers}"
            )
        ]
        response = self.llm.chat(messages)
        # clean response
        return re.sub(r"^assistant:\s*", "", str(response)).strip()