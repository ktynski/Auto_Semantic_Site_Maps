import streamlit as st
import pandas as pd
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain_anthropic import ChatAnthropic
import os
import networkx as nx
from community import community_louvain
import anthropic
from typing import Dict, Set


# Set up Anthropic API key
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]


# Define models
Opus = "claude-3-opus-20240229"
Sonnet = "claude-3-sonnet-20240229"
Haiku = "claude-3-haiku-20240307"

# Initialize LLM
llm = ChatAnthropic(temperature=0.2, model_name=Haiku, max_tokens=4000)


class EntityGenerator:

    def __init__(self, llm):

        self.llm = llm

        self.entity_id_counter = 0

    def generate_entities(self, topic: str, existing_entities: Dict[str, str], num_new_entities: int, temperature: float) -> Dict[str, str]:

        prompt = PromptTemplate(

            input_variables=["topic", "existing_entities", "num_new_entities"],

            template="""Given the topic '{topic}' and the existing entities:\n\n{existing_entities}\n\n

            Your task is to suggest {num_new_entities} new entities that are semantically related to the topic and existing entities, but not already present in the existing entities.

            Use ontologies, word embeddings, and similarity measures to expand the entities while narrowing the focus based on the existing entities. Employ a simulated Monte Carlo Tree search as your mental model for coming up with this list. The goal is complete comprehensive semantic depth and breadth for the topic.

            Example output:

            machine learning, deep learning, neural networks, computer vision, natural language processing

            Please provide the output as a comma-separated list of new entities, without any other text or explanations. Your suggestions will contribute to building a comprehensive and insightful semantic map, so aim for high-quality and relevant entities.""",

        )

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        new_entities_response = llm_chain.run(

            topic=topic,

            existing_entities=", ".join([entity for entity in existing_entities.values()]),

            num_new_entities=num_new_entities,

        )

        new_entities = {}

        for entity in new_entities_response.split(","):

            entity = entity.strip()

            if entity and entity not in existing_entities.values():

                entity_id = f"e{self.entity_id_counter}"

                new_entities[entity_id] = entity

                self.entity_id_counter += 1

        return new_entities

class RelationshipGenerator:

    def __init__(self, llm):

        self.llm = llm

    def generate_batch_relationships(self, topic: str, batch_entities: Dict[str, str], other_entities: Dict[str, str], existing_relationships: Set[tuple]) -> Set[tuple]:

        prompt = PromptTemplate(

            input_variables=["topic", "batch_entities", "other_entities", "existing_relationships"],

            template="""Given the topic '{topic}' and the following entities:

            {batch_entities}

            Consider the other entities:

            {other_entities}

            and the existing relationships:

            {existing_relationships}

            Your task is to identify relevant relationships between the given entities and the other entities in the context of the topic.

            Use domain knowledge to prioritize important connections and provide meaningful edge labels. You must give each entity no less than 4 relationships and no more than 20 relationships for any individual entity. You must return all requested entity relationships.

            Example output:

            source_id1,target_id1,edge_label1

            source_id2,target_id2,edge_label2

            source_id3,target_id3,edge_label3

            Please provide the output as a list of relationships and their labels, in the format 'source_id,target_id,edge_label', without any other text or explanations.

            Focus on identifying the most significant and impactful relationships.""",

        )

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        batch_entity_ids = list(batch_entities.keys())

        existing_batch_relationships = [f"{rel[0]},{rel[1]},{rel[2]}" for rel in existing_relationships if rel[0] in batch_entity_ids]

        new_relationships_response = llm_chain.run(

            topic=topic,

            batch_entities=", ".join([f"{id}: {entity}" for id, entity in batch_entities.items()]),

            other_entities=", ".join([f"{id}: {entity}" for id, entity in other_entities.items()]),

            existing_relationships=", ".join(existing_batch_relationships),

        )

        new_relationships = set()

        for rel in new_relationships_response.split("\n"):

            rel = rel.strip()

            if "," in rel:

                parts = rel.split(",")

                if len(parts) >= 2:

                    source_id, target_id = parts[:2]

                    source_id = source_id.strip()

                    target_id = target_id.strip()

                    edge_label = parts[2].strip() if len(parts) > 2 else ""

                    if source_id in batch_entity_ids and target_id in other_entities and (source_id, target_id, edge_label) not in existing_relationships:

                        new_relationships.add((source_id, target_id, edge_label))

        return new_relationships

    def generate_relationships(self, topic: str, entities: Dict[str, str], existing_relationships: Set[tuple], batch_size: int, num_parallel_runs: int) -> Set[tuple]:

        new_relationships = set()

        entity_ids = list(entities.keys())

        batches = [entity_ids[i:i+batch_size] for i in range(0, len(entity_ids), batch_size)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:

            futures = []

            for batch_entity_ids in batches:

                batch_entities = {id: entities[id] for id in batch_entity_ids}

                other_entities = {id: entities[id] for id in entities if id not in batch_entity_ids}

                for _ in range(num_parallel_runs):

                    future = executor.submit(self.generate_batch_relationships, topic, batch_entities, other_entities, existing_relationships)

                    futures.append(future)

            for future in concurrent.futures.as_completed(futures):

                new_relationships.update(future.result())

        return new_relationships

class SemanticMapGenerator:

    def __init__(self, entity_generator: EntityGenerator, relationship_generator: RelationshipGenerator):

        self.entity_generator = entity_generator

        self.relationship_generator = relationship_generator

    def generate_semantic_map(self, topic: str, num_iterations: int, num_parallel_runs: int, num_entities_per_run: int, temperature: float, relationship_batch_size: int) -> Dict[str, Set]:

        entities = {}

        relationships = set()

        for iteration in range(num_iterations):

            print(f"Iteration {iteration + 1}")

            # Parallel entity generation

            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:

                futures = []

                for _ in range(num_parallel_runs):

                    future = executor.submit(self.entity_generator.generate_entities, topic, entities, num_entities_per_run, temperature)

                    futures.append(future)

                new_entities = {}

                for future in concurrent.futures.as_completed(futures):

                    new_entities.update(future.result())

            # Deduplicate entities

            entities.update(new_entities)

            print(f"Total entities: {len(entities)}")

            # Parallel relationship generation

            new_relationships = self.relationship_generator.generate_relationships(topic, entities, relationships, relationship_batch_size, num_parallel_runs)

            relationships.update(new_relationships)

            print(f"Total relationships: {len(relationships)}")

        return {"entities": entities, "relationships": relationships}

def save_semantic_map_to_csv(semantic_map: Dict[str, Set], topic: str):

    entities_file = f"{topic}_entities.csv"

    with open(entities_file, "w") as f:

        f.write("Id,Label\n")

        for id, entity in semantic_map["entities"].items():

            f.write(f"{id},{entity}\n")

    relationships_file = f"{topic}_relationships.csv"

    with open(relationships_file, "w") as f:

        f.write("Source,Target,Type\n")

        for relationship in semantic_map["relationships"]:

            f.write(f"{relationship[0]},{relationship[1]},{relationship[2]}\n")

# Streamlit app
def main():
    st.set_page_config(page_title="Auto Semantic SiteMaps", layout="wide")
    st.title("Auto Semantic SiteMaps")

    # Sidebar
    st.sidebar.title("Settings")
    topic = st.sidebar.text_input("Topic", value="Stochastic Terrorism of the Far Right in the USA Against LGBT_relationships")
    num_iterations = st.sidebar.number_input("Number of Iterations", min_value=1, value=1)
    num_parallel_runs = st.sidebar.number_input("Number of Parallel Runs", min_value=1, value=10)
    num_entities_per_run = st.sidebar.number_input("Number of Entities per Run", min_value=1, value=30)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=1.2, step=0.1)
    relationship_batch_size = st.sidebar.number_input("Relationship Batch Size", min_value=1, value=30)

    if st.sidebar.button("Generate Semantic Map"):
        # Generate semantic map
        entity_generator = EntityGenerator(llm)
        relationship_generator = RelationshipGenerator(llm)
        semantic_map_generator = SemanticMapGenerator(entity_generator, relationship_generator)
        semantic_map = semantic_map_generator.generate_semantic_map(topic, num_iterations, num_parallel_runs, num_entities_per_run, temperature, relationship_batch_size)

        # Save semantic map to CSV
        save_semantic_map_to_csv(semantic_map, topic)

        # Load the CSV files into DataFrames
        nodes_df = pd.read_csv(f"{topic}_entities.csv")
        edges_df = pd.read_csv(f"{topic}_relationships.csv")

        # Create a directed graph using NetworkX
        G = nx.DiGraph()

        # Add nodes to the graph
        for _, row in nodes_df.iterrows():
            G.add_node(row['Id'], label=row['Label'])

        # Add edges to the graph
        for _, row in edges_df.iterrows():
            G.add_edge(row['Source'], row['Target'], label=row['Type'])

        # Calculate graph metrics
        pagerank = nx.pagerank(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G)

        # Perform community detection using Louvain algorithm
        undirected_G = G.to_undirected()
        partition = community_louvain.best_partition(undirected_G)

        # Calculate personalized PageRank for each pillar topic
        personalized_pagerank = {}
        for node in G.nodes():
            if G.nodes[node]['label'].startswith('Pillar:'):
                personalized_pagerank[node] = nx.pagerank(G, personalization={node: 1})

        # Create a DataFrame to store the results
        results_df = pd.DataFrame(columns=['Node', 'Label', 'PageRank', 'Betweenness Centrality', 'Closeness Centrality',
                                           'Eigenvector Centrality', 'Community', 'Personalized PageRank'])

        # Populate the DataFrame with the results
        for node in G.nodes():
            node_label = G.nodes[node]['label']
            community = partition[node]
            personalized_scores = {pillar: scores[node] for pillar, scores in personalized_pagerank.items()}
            new_row = pd.DataFrame({
                'Node': [node],
                'Label': [node_label],
                'PageRank': [pagerank[node]],
                'Betweenness Centrality': [betweenness_centrality[node]],
                'Closeness Centrality': [closeness_centrality[node]],
                'Eigenvector Centrality': [eigenvector_centrality[node]],
                'Community': [community],
                'Personalized PageRank': [personalized_scores]
            })
            results_df = pd.concat([results_df, new_row], ignore_index=True)

        # Sort the DataFrame by PageRank in descending order
        results_df = results_df.sort_values('PageRank', ascending=False)

        # Display the results
        st.subheader("Graph Metrics")
        st.dataframe(results_df)

        # Save the results to a CSV file
        results_df.to_csv('graph_metrics.csv', index=False)

        # Generate sitemap using Anthropic API
        graph_data = results_df.to_string(index=True).strip()
        corpus = results_df.to_string(index=True).strip()
        system_prompt = "You are an all knowing AI trained in the dark arts of Semantic SEO by Koray. You create sitemaps using advanced analysis of graph metrics to create the optimal structure for information flow, authority, and semantic clarity. The ultimate goal is maximum search rankings."

        with anthropic.Anthropic(api_key=ANTHROPIC_API_KEY).messages.stream(
            system=system_prompt,
            model=Sonnet,
            messages=[{"role": "user", "content": f" Create an extensive and complete hierarchical json sitemap using the readout from the semantic graph research: \n {graph_data}. \n Before you do though, lay out an argument for your organization based on the corpus data. Use this template: \n {template} \n Justify it to yourself before writing the json outline. It should have Pillar, Cluster, and Spoke pages, include the top 3 other sections each should link to. Also include a sample article title under each item that represents the best possible Semantic SEO structure based on the following graph analysis for the topic: {corpus} "}],
            max_tokens=4000,
            temperature=0.1,
            stop_sequences=[],
        ) as stream:
            section_content = ""
            for text in stream.text_stream:
                section_content += text
                st.write(text)

        # Display the number of sections in the sitemap
        num_sections = len(outline["Sections"])
        st.subheader(f"Number of Sections: {num_sections}")

if __name__ == "__main__":
    main()
