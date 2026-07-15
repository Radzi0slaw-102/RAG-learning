import re
import asyncio
from typing import Any, Callable, List, Optional, Union
from llama_index.core.async_utils import run_jobs
from llama_index.core.graph_stores.types import EntityNode, KG_NODES_KEY, KG_RELATIONS_KEY, Relation
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts import PromptTemplate
from llama_index.core.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.core.schema import TransformComponent, BaseNode

from config import KG_TRIPLET_EXTRACT_TMPL

ENTITY_PATTERN = r"entity_name:\ss*(.+?)\s*entity_type:\s*(.+?)\s*entity_description:\s*(.+?)\s*"
RELATIONSHIP_PATTERN = r"source_entity:\s*(.+?)\s*target_entity:\s*(.+?)\s*relation:\s*(.+?)\s*relationship_description:\s*(.+?)\s*"

def parse_fn(response_str: str) -> Any:
    entities = re.findall(ENTITY_PATTERN, response_str)
    relationships = re.findall(RELATIONSHIP_PATTERN, response_str)
    return entities, relationships


class GraphRAGEXtractor(TransformComponent):
    """Extract triples from a graph.
 
    Uses an LLM and a simple prompt + output parsing to extract paths (i.e. triples) and entity, relation descriptions from text.
 
    Args:
        llm (LLM):
            The language model to use.
        extract_prompt (Union[str, PromptTemplate]):
            The prompt to use for extracting triples.
        parse_fn (callable):
            A function to parse the output of the language model.
        num_workers (int):
            The number of workers to use for parallel processing.
        max_paths_per_chunk (int):
            The maximum number of paths to extract per chunk.
    """
    
    llm: LLM
    extract_prompt: PromptTemplate
    parse_fn: Callable
    num_workers: int
    max_paths_per_chunk: int
    
    def __init__(
        self,
        llm: Optional[LLM] = None,
        extract_prompt: Optional[Union[str, PromptTemplate]] = None,
        parse_fn: Callable = parse_fn,
        num_workers: int = 4,
        max_paths_per_chunk: int = 10
    ) -> None:
        from llama_index.core import Settings
        
        if isinstance(extract_prompt, str):
            extract_prompt = PromptTemplate(extract_prompt)
        
        super().__init__(
            llm=llm or Settings.llm,
            extract_prompt=extract_prompt or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT,
            parse_fn=parse_fn,
            num_workers=num_workers,
            max_paths_per_chunk=max_paths_per_chunk
        )
    
    @classmethod
    def class_name(cls) -> str:
        return "GraphExtractor"
    
    def __call__(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        # extract triplets from nodes
        return asyncio.run(
            self.acall(nodes, show_progress=show_progress, **kwargs)
        )
    
    async def _aextract(self, node: BaseNode) -> BaseNode:
        # extract triplets from single node
        assert hasattr(node, "text")
        
        text = node.get_content(metadata_mode="llm")
        try:
            llm_response = await self.llm.apredict(
                self.extract_prompt,
                text=text,
                max_knowledge_triplets=self.max_paths_per_chunk,
            )
            entities, entities_relationship = self.parse_fn(llm_response)
        except ValueError:
            entities = []
            entities_relationship = []
        
        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        metadata = node.metadata.copy()
        for entity, entity_type, description in entities:
            metadata[
                "entity_description"
            ] = description
            entity_node = EntityNode(
                name=entity, label=entity_type, properties=metadata
            )
            existing_nodes.append(entity_node)
        
        metadata = node.metadata.copy()
        for triple in entities_relationship:
            subj, rel, obj, desc = triple
            subj_node = EntityNode(name=subj, properties=metadata)
            obj_node = EntityNode(name=obj, properties=metadata)
            metadata["relationship_description"] = desc
            rel_node = Relation(
                label=rel,
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties=metadata
            )
            
            existing_nodes.extend([subj_node, obj_node])
            existing_relations.append(rel_node)
        
        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node

    async def acall(
        self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        # extract triples from nodes async
        jobs = []
        for node in nodes:
            jobs.append(self._aextract(node))
        
        return await run_jobs(
            jobs,
            workers=self.num_workers,
            show_progress=show_progress,
            desc="Extracting paths from text"
        )