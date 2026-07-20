import torch
from typing import Any, List, Tuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from llama_index.core.graph_stores.types import EntityNode, KG_NODES_KEY, KG_RELATIONS_KEY, Relation
from llama_index.core.schema import TransformComponent, BaseNode

def parse_rebel_output(text: str) -> List[Tuple[str, str, str]]:
    # parse REBEL's tagged output into (subject, relation, object) triplets
    triplets = []
    text = text.strip()
    current = "x"
    subject, relation, obj = "", "", ""
    
    for token in text.replace("<s>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = "t"
            if subject and relation and obj:
                triplets.append((subject.strip(), relation.strip(), obj.strip()))
            subject, relation, obj = "", "", ""
        elif token == "<subj>":
            current = "s"
            if subject and relation and obj:
                triplets.append((subject.strip(), relation.strip(), obj.strip()))
            relation = ""
        elif token == "<obj>":
            current = "o"
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                obj += " " + token
            elif current == "o":
                relation += " " + token

    if subject and relation and obj:
        triplets.append((subject.strip(), relation.strip(), obj.strip()))

    return triplets


class RebelExtractor(TransformComponent):
    # extract entity-relation triplets from text using REBEL, without an LLM

    model_name: str = "Babelscape/rebel-large"
    max_length: int = 256
    num_workers: int = 4

    def __init__(self, model_name: str = "Babelscape/rebel-large", max_length: int = 256, num_workers: int = 4):
        super().__init__(model_name=model_name, max_length=max_length, num_workers=num_workers)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.eval()

    @classmethod
    def class_name(cls) -> str:
        return "RebelExtractor"

    def __call__(self, nodes: List[BaseNode], show_progress: bool = False, **kwargs: Any) -> List[BaseNode]:
        for node in nodes:
            self._extract(node)
        return nodes

    def _generate(self, text: str) -> str:
        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=self.max_length)
        with torch.no_grad():
            output_ids = self._model.generate(**inputs, max_length=self.max_length, num_beams=3)
        return self._tokenizer.decode(output_ids[0], skip_special_tokens=False)

    def _extract(self, node: BaseNode) -> BaseNode:
        text = node.get_content(metadata_mode="llm")
        raw_output = self._generate(text)
        triplets = parse_rebel_output(raw_output)

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_relations = node.metadata.pop(KG_RELATIONS_KEY, [])
        metadata = node.metadata.copy()

        for subj, rel, obj in triplets:
            subj_node = EntityNode(name=subj, properties=metadata)
            obj_node = EntityNode(name=obj, properties=metadata)
            metadata["relationship_description"] = rel
            rel_node = Relation(
                label=rel,
                source_id=subj_node.id,
                target_id=obj_node.id,
                properties=metadata,
            )
            existing_nodes.extend([subj_node, obj_node])
            existing_relations.append(rel_node)

        node.metadata[KG_NODES_KEY] = existing_nodes
        node.metadata[KG_RELATIONS_KEY] = existing_relations
        return node