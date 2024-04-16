import streamlit as st
import pandas as pd
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain_anthropic import ChatAnthropic
import os
import networkx as nx
import community as community_louvain
import anthropic
from typing import Dict, Set
import concurrent.futures
import json
from streamlit import experimental_rerun



# Define models
Opus = "claude-3-opus-20240229"
Sonnet = "claude-3-sonnet-20240229"
Haiku = "claude-3-haiku-20240307"

progress_bar = None

template = {
    "Pillars": [
        {
            "name": "Pillar 1",
            "justification": "This pillar covers the core issues and concepts that are central to the overall topic, as determined by their high PageRank scores.",
            "sample_article": "A Comprehensive Overview of [Topic]: Key Concepts, Issues, and Perspectives"
        },
        {
            "name": "Pillar 2",
            "justification": "This pillar focuses on the fundamental aspects and subtopics that are essential for understanding the main topic, based on their significant PageRank scores.",
            "sample_article": "Exploring the Foundations of [Topic]: A Deep Dive into Core Principles and Theories"
        },
        {
            "name": "Pillar 3",
            "justification": "This pillar examines the critical components and themes that shape the overall discourse surrounding the main topic, as indicated by their notable PageRank scores.",
            "sample_article": "Navigating the Landscape of [Topic]: Essential Elements and Influential Factors"
        }
    ],
    "Clusters": [
        {
            "name": "Cluster 1",
            "pillar": "Pillar 1",
            "justification": "This cluster focuses on the closely related subtopics and themes that are crucial for comprehending the main pillar, as determined by their high betweenness centrality scores.",
            "sample_article": "Unraveling the Intricacies of [Subtopic]: A Comprehensive Analysis",
            "related_pillars": ["Pillar 2", "Pillar 3"]
        },
        {
            "name": "Cluster 2",
            "pillar": "Pillar 1",
            "justification": "This cluster explores the interconnected concepts and ideas that bridge various aspects of the main pillar, based on their significant betweenness centrality scores.",
            "sample_article": "Bridging the Gap: Examining the Interconnectedness of [Subtopic] within [Topic]",
            "related_pillars": ["Pillar 2", "Pillar 3"]
        },
        {
            "name": "Cluster 3",
            "pillar": "Pillar 2",
            "justification": "This cluster delves into the key issues and challenges associated with the main pillar, as indicated by their high betweenness centrality scores.",
            "sample_article": "Confronting the Challenges of [Subtopic]: Strategies and Solutions",
            "related_pillars": ["Pillar 1", "Pillar 3"]
        },
        {
            "name": "Cluster 4",
            "pillar": "Pillar 2",
            "justification": "This cluster investigates the fundamental principles and theories that underpin the main pillar, based on their significant betweenness centrality scores.",
            "sample_article": "Unveiling the Foundations: A Deep Dive into [Subtopic] Principles and Theories",
            "related_pillars": ["Pillar 1", "Pillar 3"]
        },
        {
            "name": "Cluster 5",
            "pillar": "Pillar 3",
            "justification": "This cluster examines the emerging trends and developments within the main pillar, as determined by their high betweenness centrality scores.",
            "sample_article": "On the Horizon: Exploring Emerging Trends and Innovations in [Subtopic]",
            "related_pillars": ["Pillar 1", "Pillar 2"]
        },
        {
            "name": "Cluster 6",
            "pillar": "Pillar 3",
            "justification": "This cluster analyzes the impact and implications of the main pillar on various aspects of society and industry, based on their significant betweenness centrality scores.",
            "sample_article": "The Ripple Effect: Examining the Impact of [Subtopic] on Society and Industry",
            "related_pillars": ["Pillar 1", "Pillar 2"]
        }
    ],
    "Spokes": [
        {
            "name": "Spoke 1",
            "cluster": "Cluster 1",
            "justification": "This spoke focuses on a specific aspect or application of the cluster, as determined by its high closeness centrality score.",
            "sample_article": "Diving Deep: A Comprehensive Look at [Specific Aspect] within [Subtopic]"
        },
        {
            "name": "Spoke 2",
            "cluster": "Cluster 1",
            "justification": "This spoke explores a particular case study or real-world example related to the cluster, based on its significant closeness centrality score.",
            "sample_article": "From Theory to Practice: A Case Study on Implementing [Specific Aspect] in [Industry/Context]"
        },
        {
            "name": "Spoke 3",
            "cluster": "Cluster 2",
            "justification": "This spoke examines a specific challenge or obstacle associated with the cluster, as indicated by its high closeness centrality score.",
            "sample_article": "Overcoming Hurdles: Strategies for Addressing [Specific Challenge] in [Subtopic]"
        },
        {
            "name": "Spoke 4",
            "cluster": "Cluster 2",
            "justification": "This spoke investigates a particular solution or approach related to the cluster, based on its significant closeness centrality score.",
            "sample_article": "Innovative Solutions: Exploring [Specific Approach] for Tackling [Subtopic] Issues"
        },
        {
            "name": "Spoke 5",
            "cluster": "Cluster 3",
            "justification": "This spoke analyzes a specific trend or pattern within the cluster, as determined by its high closeness centrality score.",
            "sample_article": "Spotting the Trends: An In-Depth Analysis of [Specific Trend] in [Subtopic]"
        },
        {
            "name": "Spoke 6",
            "cluster": "Cluster 3",
            "justification": "This spoke explores a particular methodology or framework related to the cluster, based on its significant closeness centrality score.",
            "sample_article": "Frameworks for Success: Applying [Specific Methodology] in [Subtopic] Contexts"
        },
        {
            "name": "Spoke 7",
            "cluster": "Cluster 4",
            "justification": "This spoke examines a specific application or use case associated with the cluster, as indicated by its high closeness centrality score.",
            "sample_article": "From Concept to Application: Exploring [Specific Use Case] in [Subtopic]"
        },
        {
            "name": "Spoke 8",
            "cluster": "Cluster 4",
            "justification": "This spoke investigates a particular best practice or guideline related to the cluster, based on its significant closeness centrality score.",
            "sample_article": "Setting the Standard: Best Practices for Implementing [Specific Guideline] in [Subtopic]"
        },
        {
            "name": "Spoke 9",
            "cluster": "Cluster 5",
            "justification": "This spoke analyzes a specific impact or consequence associated with the cluster, as determined by its high closeness centrality score.",
            "sample_article": "The Domino Effect: Examining the Impact of [Specific Consequence] in [Subtopic]"
        },
        {
            "name": "Spoke 10",
            "cluster": "Cluster 5",
            "justification": "This spoke explores a particular opportunity or potential related to the cluster, based on its significant closeness centrality score.",
            "sample_article": "Unlocking Potential: Exploring [Specific Opportunity] in [Subtopic]"
        },
        {
            "name": "Spoke 11",
            "cluster": "Cluster 6",
            "justification": "This spoke examines a specific case study or real-world example associated with the cluster, as indicated by its high closeness centrality score.",
            "sample_article": "Lessons Learned: A Case Study on [Specific Example] in [Subtopic]"
        },
        {
            "name": "Spoke 12",
            "cluster": "Cluster 6",
            "justification": "This spoke investigates a particular future direction or possibility related to the cluster, based on its significant closeness centrality score.",
            "sample_article": "Envisioning the Future: Exploring [Specific Possibility] in [Subtopic]"
        }
    ]
}

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
        self.entities = {}
        self.relationships = set()

    def generate_semantic_map(self, topic: str, num_iterations: int, num_parallel_runs: int, num_entities_per_run: int, temperature: float, relationship_batch_size: int) -> Dict[str, Set]:
        entities_count = 0
        relationships_count = 0
        entities_placeholder = st.empty()  # Initialize entities_placeholder
        relationships_placeholder = st.empty()  # Initialize relationships_placeholder

        for iteration in range(num_iterations):
            # Parallel entity generation
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for _ in range(num_parallel_runs):
                    future = executor.submit(self.entity_generator.generate_entities, topic, self.entities, num_entities_per_run, temperature)
                    futures.append(future)
                new_entities = {}
                for future in concurrent.futures.as_completed(futures):
                    new_entities.update(future.result())

            # Deduplicate entities
            self.entities.update(new_entities)
            entities_count += len(new_entities)
            entities_placeholder.metric("Total Entities", entities_count)

            # Parallel relationship generation
            new_relationships = self.relationship_generator.generate_relationships(topic, self.entities, self.relationships, relationship_batch_size, num_parallel_runs)
            self.relationships.update(new_relationships)
            relationships_count += len(new_relationships)
            relationships_placeholder.metric("Total Relationships", relationships_count)

            progress = (iteration + 1) / num_iterations
            progress_bar.progress(progress)
            experimental_rerun()

        return {"entities": self.entities, "relationships": self.relationships}

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


import Levenshtein

def merge_similar_nodes(G, similarity_threshold=0.8):
    merged_nodes = set()
    for node1 in G.nodes():
        if node1 not in merged_nodes:
            for node2 in G.nodes():
                if node1 != node2 and node2 not in merged_nodes:
                    label1 = G.nodes[node1]['label']
                    label2 = G.nodes[node2]['label']
                    similarity = Levenshtein.ratio(label1, label2)
                    if similarity >= similarity_threshold:
                        # Merge nodes
                        G = nx.contracted_nodes(G, node1, node2, self_loops=False)
                        merged_nodes.add(node2)
                        break
    return G


# Streamlit app
def main():
    st.set_page_config(page_title="Generating Semantically Complete Sitemaps with Large Language Models and Graph Analysis", layout="wide")
    st.title("Semantically Complete SiteMaps with LLMs and Graph Analysis")
    description = """
    ## What is this and how does it work?

        This Agenti AI tool leverages multiple advanced techniques to generate highly sophisticated, extremely comprehensive, SEO-optimized semantic sitemaps:
        
        - 🌐 **Graph analysis metrics** (PageRank, betweenness centrality) to identify crucial nodes
        - 🧩 **Community detection algorithms** to uncover closely related topic clusters
        - 🤖 **Large language models** to process extensive data and generate comprehensive semantic maps
        
        The tool automates the creation of semantically rich, well-organized, and search engine-optimized sitemaps that would be incredibly difficult and time-consuming to build manually. 
        
        > The resulting output is a hierarchical JSON sitemap that serves as a blueprint for constructing a website that is semantically rich, well-organized, and highly optimized for search engines.
        
        Key features:
        - Generates up to 5,000 entities and 25,000 relationships for a given topic
        - Provides concrete recommendations for optimizing website structure and content
        - Automatically generates a Graphviz chart for easy visualization of the sitemap structure
        
        Benefits:
        - 🚀 **Saves significant time and effort** compared to manual semantic mapping
        - 📈 **Enhances search rankings** through semantically coherent content organization
        - 🧠 **Leverages advanced data-driven techniques** that are hard to replicate through human analysis  """
    st.markdown(description)
    # Sidebar
    st.sidebar.title("Settings")
    topic = st.sidebar.text_input("Topic", value="Enter Your Topic Here")
    ANTHROPIC_API_KEY = st.sidebar.text_input("Anthropic API Key", type="password")
    num_iterations = st.sidebar.number_input("Number of Iterations", min_value=1, max_value=3, value=1)
    num_parallel_runs = st.sidebar.number_input("Number of Parallel Runs", min_value=1, max_value=10, value=5)
    num_entities_per_run = st.sidebar.number_input("Number of Entities per Run", min_value=1, max_value=20, value=10)
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    relationship_batch_size = st.sidebar.number_input("Relationship Batch Size", min_value=1, max_value=20, value=10)
    model_name = st.sidebar.selectbox("Claude Model", [Opus, Sonnet, Haiku], index=2)
    # Initialize LLM
    llm = ChatAnthropic(temperature=0.2, model_name=model_name, max_tokens=4000, api_key=ANTHROPIC_API_KEY)
    global progress_bar
    progress_bar = st.progress(0)
    if st.sidebar.button("Generate Semantic Map"):
        if not ANTHROPIC_API_KEY:
            st.error("Please enter a valid Anthropic API key.")
        else:
            status_text = st.empty()

            # Generate semantic map
            entity_generator = EntityGenerator(llm)
            relationship_generator = RelationshipGenerator(llm)
            semantic_map_generator = SemanticMapGenerator(entity_generator, relationship_generator)
            with st.spinner("Generating semantic map..."):
                entities_placeholder = st.empty()
                relationships_placeholder = st.empty()
                entities_count = 0
                relationships_count = 0

                for iteration in range(num_iterations):
                    # Parallel entity generation
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        futures = []
                        for _ in range(num_parallel_runs):
                            future = executor.submit(entity_generator.generate_entities, topic, semantic_map_generator.entities, num_entities_per_run, temperature)
                            futures.append(future)
                        new_entities = {}
                        for future in concurrent.futures.as_completed(futures):
                            new_entities.update(future.result())

                    # Deduplicate entities
                    semantic_map_generator.entities.update(new_entities)
                    entities_count += len(new_entities)
                    entities_placeholder.metric("Total Entities", entities_count)

                    # Parallel relationship generation
                    new_relationships = relationship_generator.generate_relationships(topic, semantic_map_generator.entities, semantic_map_generator.relationships, relationship_batch_size, num_parallel_runs)
                    semantic_map_generator.relationships.update(new_relationships)
                    relationships_count += len(new_relationships)
                    relationships_placeholder.metric("Total Relationships", relationships_count)

                    progress_bar.progress((iteration + 1) / num_iterations)

            status_text.text("Semantic map generated.")

        # Save semantic map to CSV
        save_semantic_map_to_csv({"entities": semantic_map_generator.entities, "relationships": semantic_map_generator.relationships}, topic)
        progress_bar.progress(0.4)
        status_text.text("Semantic map saved to CSV.")
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

        # Merge similar nodes
        G = merge_similar_nodes(G, similarity_threshold=0.8)

        # Calculate graph metrics
        with st.spinner("Calculating graph metrics..."):
            pagerank = nx.pagerank(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
            eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
            progress_bar.progress(0.6)
            status_text.text("Graph metrics calculated.")
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
        progress_bar.progress(0.8)
        status_text.text("Results DataFrame created.")
        # Display the results
        with st.expander("Graph Metrics"):
            st.dataframe(results_df)
            st.subheader("DataFrame Summary")
            st.write(results_df.describe())
        # Save the results to a CSV file
        results_df.to_csv('graph_metrics.csv', index=False)
        
        # Generate sitemap using Anthropic API
        graph_data = results_df.to_string(index=True).strip()
        corpus = results_df.to_string(index=True).strip()
        system_prompt = "You are an all knowing AI trained in the dark arts of Semantic SEO by Koray. You create sitemaps using advanced analysis of graph metrics to create the optimal structure for information flow, authority, and semantic clarity. The ultimate goal is maximum search rankings."
        with st.spinner("Generating sitemap..."):
            sitemap_response = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY).messages.create(
                system = system_prompt,
                messages = [{"role": "user", "content": f"Create an extensive and complete hierarchical json sitemap using the readout from the semantic graph research: \n {graph_data}. \n Before you do though, lay out an argument for your organization based on the corpus data. Use this template: \n {template} \n Justify it to yourself before writing the json outline. It should have Pillar, Cluster, and Spoke pages, include the top 3 other sections each should link to. Also include a sample article title under each item that represents the best possible Semantic SEO structure based on the following graph analysis for the topic: {corpus}"}],
                model=model_name,
                max_tokens=4000,
                temperature=0.1,
                stop_sequences=[],
            )
            sitemap_json = sitemap_response.content[0].text
            progress_bar.progress(1.0)
            status_text.text("Sitemap generated.")
        st.code(sitemap_json, language="json")
        # Generate additional commentary and recommendations using Anthropic API
        with st.spinner("Generating additional commentary and recommendations..."):
            commentary_response = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY).messages.create(
                system = system_prompt,
                messages = [{"role": "user", "content": f"Based on the generated semantic sitemap and graph analysis, provide a few paragraphs of additional commentary and concrete recommendations that are highly specific for the given topic and provided sitemap for optimizing the website structure and content for semantic SEO. Consider factors such as internal linking anchortext, content depth and breadth, and user experience. Here is the graph you generated: {sitemap_json} and the underlying graph data research: {graph_data}"}],
                model=model_name,
                max_tokens=1000,
                temperature=0.2,
                stop_sequences=[],
            )
            commentary = commentary_response.content[0].text
        st.markdown(commentary)
        # Generate Mermaid chart using Anthropic API
        with st.spinner("Generating Mermaid chart..."):
            mermaid_prompt = f"""
                Generate a Graphviz DOT representation of the hierarchical structure of the semantic sitemap. Use the following JSON sitemap as input:
                {sitemap_json}
                Example Graphviz DOT:
                
                digraph {{
                    rankdir=LR;
                    "Pillar 1" -> "Cluster 1";
                    "Pillar 1" -> "Cluster 2";
                    "Cluster 1" -> "Spoke 1";
                    "Cluster 1" -> "Spoke 2";
                    "Cluster 2" -> "Spoke 3";
                    "Cluster 2" -> "Spoke 4";
                }}
                
                DO NOT return any commentary, preamble, postamble, or meta commentary on the task or its completion. Return ONLY the digraph. Your response should start with digraph and then a bracket."""
            mermaid_response = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY).messages.create(
                system = "You are an AI that ouputs digraph charts in a specific format without any other text, preamble, postamble or meta commentary. You do this accurately every time. Failing results in me losing my job.",
                messages=[{"role": "user", "content": f"{mermaid_prompt}"}],
                model=model_name,
                max_tokens=4000,
                temperature=0.1,
                stop_sequences=[],
            )
            mermaid_chart = mermaid_response.content[0].text
            print(mermaid_chart)
        st.markdown("## Site Map Visualization")
        st.graphviz_chart(mermaid_chart, use_container_width=True)


if __name__ == "__main__":
    main()
