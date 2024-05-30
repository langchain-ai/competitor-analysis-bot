"""### UX:

- Input:
    - Expected: URL for a  product
        - Or maybe a specific product to scope it down
    - Alternatives:
        - company name, “idea” etc.
- Output:
    - List of competing products
        - Name,
        - summary details
        - How they compare to base company
        - List of customers they are marketing towards
    - Recommendations for potential experiments that could be conducted, such as those related to product discovery, positioning experiments, or user experiments, based on the gathered results.
- Flow (kinda similar to perplexity)
    - User provides a url for a product
    - Bot searches and generates a list of features with detailed information (problem it’s trying to solve, who it’s solving this for (customer segments), etc.) for that product
    - **Confirms with user which features/product to analyze**
        - Think we should avoid multiple products rn bcs it’s kinda hard to do like “find competitors for nvidia” vs. “find competitors of CUDA”
    - (branching) For each product, bot does a web search to find competitors for each product
        - MVP: just query expansion from the product description then google/tavily searches or whatever based on key words
    - **(?)** Maybe could include user confirmation on competitors before proceeding here
    - (branching) for each competitor, populate a similar product card with product name, summary, primary target audience, list of customers/potential customers
    - For each card, write 1-3 experiments you could do
    - (merge) For all the competing products/customers rerank to prioritize experiments

### Graph Structure

Nodes (verbs):

- research_company
    - given the url and optionally other description, scrape the website to populate N product cards
    - **Could probably just be a 1-step extraction** for now
- find_competitors
    - Given the product card of the base company, search the web and generate a list of competing products
    - **Could probably be a simple zero-shot agent**
- research_competitor
    - Given the url and description of a competitor, research competing product + generate competitor card:
        - same as product card +
            - Strengths/benefits vs. the base company
            - Weaknesses vs. base company
            - Recommended experiment(s) to run to gain more information about this competitor
            - Extension: (?) Perceived level of competitiveness - llm probably cant do this well/at all, but some way of scoring this based on similarity to base company and perceived quality of service
- prioritize_experiments
    - Given the list of competitor cards and experiments,

**Extensions (wont impelment)**:

- Start mid-way (we already have our product card, we already have competitor information, etc)
- Expand search (for each competing product, do the same search so you can include competitors of competitors in your list)."""

import asyncio
from typing import Annotated, Dict, List, Optional

from bs4 import BeautifulSoup as Soup
from langchain_anthropic import ChatAnthropic
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import AnyUrl, BaseModel, Field, validator
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.checkpoint import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class ProductInfo(BaseModel):
    company_name: str = Field(
        ..., description="The name of the company that produces the product"
    )
    product_name: str = Field(..., description="The name of the product")
    website: AnyUrl = Field(..., description="The website of the product")
    problem_solved: str = Field(
        ..., description="The primary problem solved by the product."
    )
    key_features: List[str] = Field(
        ...,
        description="The key features + benefits of the product."
        "Each feature should be a sentence descriping the job it does without using any adjectives or adverbs.",
    )
    target_customers: List[str] = Field(
        ...,
        description="The target customers / personas / segments of the product."
        "Each customer segment should be a sentence describing the type of customer.",
    )
    search_terms: List[str] = Field(
        ...,
        description="5-10 search terms used to find competing products."
        "These can be keywords, phrases, adwords, needs, or questions that are relevant to the product.",
    )


class CompetingProductInfo(ProductInfo):
    relative_strengths: List[str] = Field(
        ...,
        description="The relative strengths or benefits of the competitor compared to the base company."
        " List as many as you can find or infer."
        " These can be explicit (e.g., 'faster processing speed', features that aren't implemented by the base company, etc )"
        " or implicit (e.g., larger company has distribution advantage, smaller company has moved faster, more existing relationships, etc.).",
    )
    relative_weaknesses: List[str] = Field(
        ...,
        description="The relative weaknesses or disadvantages of the competitor compared to the base company."
        " List as many as you can find or infer."
        " These can be explicit (e.g., 'slower processing speed', "
        "features that are implemented by the base company but not found in the competitor, etc )"
        " or implicit (e.g., smaller company has worse distribution, fewer existing relationships, etc.).",
    )
    opportunities: List[str] = Field(
        ...,
        description="Opportunities for the base company to exploit the competitor's weaknesses."
        " These can be features that the competitor lacks, customer segments that the competitor is not targeting, etc.",
    )
    threats: List[str] = Field(
        ...,
        description="Threats that the competitor poses to the base company."
        " These can be features that the competitor has that the base company lacks, customer segments that the competitor is targeting, etc.",
    )
    recommended_experiments: List[str] = Field(
        ...,
        description="1-3 experiments to run to gain more information about"
        " how the base company stacks up against this competitor.",
    )


class Competitor(BaseModel):
    company_name: str = Field(
        ..., description="The name of the company that produces the product"
    )
    product_name: Optional[str] = Field(
        None, description="The name of the product to find competitors for."
    )
    url: Optional[AnyUrl] = Field(
        None, description="The website of the product to find competitors for."
    )


class Competitors(BaseModel):
    """List of all competing products found in the research process."""

    competitors: List[Competitor] = Field(
        ...,
        description="The deduplicated list of ALL competitors found during the research process.",
    )

    @validator("competitors")
    def deduplicate_competitors(cls, competitors):
        seen = set()
        deduplicated = []
        for competitor in competitors:
            key = (competitor.company_name, competitor.product_name)
            if key not in seen:
                seen.add(key)
                deduplicated.append(competitor)
        return deduplicated


class KeySuccessFactor(BaseModel):
    name: str = Field(..., description="The name of the key success factor.")
    weight: float = Field(
        ...,
        description="The weight of the key success factor, should sum to 1.0 across all factors.",
    )
    description: str = Field(
        ..., description="A brief description of the key success factor."
    )


class CPMScore(BaseModel):
    factor: str = Field(..., description="The key success factor.")
    rating: int = Field(
        ..., description="The rating for this factor, on a scale from 1 to 4."
    )


class CPMRow(BaseModel):
    company_name: str = Field(..., description="The name of the company.")
    product_name: str = Field(..., description="The name of the product.")
    scores: List[CPMScore] = Field(
        ..., description="List of scores for each key success factor."
    )

    def get_weighted_score(self, ksfs: List[KeySuccessFactor]):
        total = 0
        for ksf in ksfs:
            factor_score = next(
                (score.rating for score in self.scores if score.factor == ksf.name), 0
            )
            total += factor_score * ksf.weight
        return total


class CPM(BaseModel):
    key_success_factors: List[KeySuccessFactor] = Field(
        ..., description="List of key success factors."
    )
    base_product: CPMRow = Field(..., description="The base product's CPM row.")
    competitors: List[CPMRow] = Field(..., description="List of competitor CPM rows.")

    def get_ranked_competitors(self):
        competitors = self.competitors.copy()
        competitors.sort(
            key=lambda x: x.get_weighted_score(self.key_success_factors), reverse=True
        )
        return competitors

    def as_markdown(self) -> str:
        """Convert this matrix to a markdown table."""
        header = "|".join(
            ["Company", "Product"]
            + [factor.name for factor in self.key_success_factors]
        )
        separator = "|".join(["---", "---"] + ["---" for _ in self.key_success_factors])
        rows = [
            "|".join(
                [row.company_name, row.product_name]
                + [str(score.rating) for score in row.scores]
            )
            for row in self.competitors
        ]
        return "\n".join([header, separator] + rows)


class ResearchState(TypedDict, total=False):
    base_product: Annotated[str | ProductInfo, lambda x, y: y if y else x]
    # First just the competitor extracted name, only later is it a full card
    competitors: Annotated[List[Competitor | ProductInfo], lambda x, y: y if y else x]
    suggested_experiments: List[str]
    cpm: CPM
    # messages: Annotated[List[BaseMessage], add_messages]


def _format_doc(doc: Document) -> str:
    xml_metadata = " ".join(f'{key}="{value}"' for key, value in doc.metadata.items())
    return f"<doc {xml_metadata.strip()}>\n{doc.page_content}\n</doc>"


def _format_docs(docs: List[Document]) -> str:
    docs = "\n".join(_format_doc(doc) for doc in docs)
    return f"<docs>\n{docs}\n</docs>"


async def load_docs(url: str, config: RunnableConfig) -> str:
    loader = RecursiveUrlLoader(
        url=url, max_depth=2, extractor=lambda x: Soup(x, "html.parser").text
    )
    docs = await loader.aload()
    return _format_docs(docs)


async def research_company(state: ResearchState, config: RunnableConfig):
    url = (
        state["base_product"]
        if isinstance(state["base_product"], str)
        else state["base_product"]["website"]
    )
    docstring = await load_docs(url, config)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a competitive analysis bot. Please generate a"
                " ProductInfo cards for the product described in the following documents.\n\n"
                "{docstring}\n\n"
                "First, brainstorm within <reflect></reflect> tags to distill the core problems the product is trying to solve, who it are solving it for, and the key features of the product(s).",
            ),
            (
                "user",
                "Reflect on the provided docs and fill out ProductInfo cards using the provided functions."
                "Be as specific as possible. Avoid adjectives and adverbs. Focus on verifiable/testable features and benefits."
                " Do not forget to provide a list of target customers and key features.",
            ),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o")

    chain = prompt | llm.with_structured_output(ProductInfo)
    product_info = await chain.ainvoke({"docstring": docstring}, config)

    return {"base_product": product_info}


async def get_competitors(state: ResearchState, config: RunnableConfig):
    base_product = state["base_product"]
    if not isinstance(base_product, ProductInfo):
        raise ValueError(
            f"Base product must be a ProductInfo object. Got: {base_product}"
        )

    search_engine = TavilySearchResults(max_results=5)
    search_results = await search_engine.abatch(
        base_product.search_terms, config, return_exceptions=True
    )
    flattened = [
        result
        for sublist in search_results
        for result in (sublist if not isinstance(sublist, Exception) else [])
    ]
    formatted_flattened = "\n".join(f"<doc>{doc}</doc>" for doc in flattened)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a competitive analysis bot tasked with extracting competitor information from search results. "
                "The information will be used to fill out competitor cards for analysis. "
                "Your task is to identify and list competitors, including their company name, product name, and URL if available. "
                "You are researching competitors to the following product:\n\n"
                "{base_product}\n\n"
                "Using this base product information, extract relevant competitor details from the search results provided.",
            ),
            (
                "user",
                "Here are the search results:\n\n"
                "{search_results}\n\n"
                "Please extract the following details for each competitor:\n"
                "1. Company name\n"
                "2. Product name (if available)\n"
                "3. URL of the competitor's website (if available)\n\n"
                "Ensure that the extracted details are accurate and complete. "
                " Generate ALL competitors found in the search results, but only include competitors with"
                " products that actually compete against the base product. Do not include irrelevant products.",
            ),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o")
    chain = prompt | llm.with_structured_output(Competitors)
    extracted: Competitors = await chain.ainvoke(
        {"search_results": formatted_flattened, "base_product": base_product.dict()},
        config,
    )
    return {"competitors": extracted.competitors}


async def _research_competitor(
    inputs: dict, config: RunnableConfig
) -> CompetingProductInfo:
    competitor: Competitor = inputs["competitor"]
    base_product: ProductInfo = inputs["base_product"]
    url = competitor.url
    if not url:
        raise ValueError(f"Competitor {competitor.company_name} has no URL.")
    docstring = await load_docs(url, config)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a competitive analysis bot. Your task is to generate a CompetingProductInfo card "
                "for the competitor described in the following documents. Use the base product information "
                "provided to compare and contrast the competitor's product.\n\n"
                "Base Product Information:\n{base_product}\n\n"
                "Competitor Documents:\n{docstring}\n\n"
                "First, brainstorm within <reflect></reflect> tags to distill the core problems the product is trying to solve, "
                "who it is solving it for, and the key features of the product(s).",
            ),
            (
                "user",
                "Reflect on the provided documents and fill out the CompetingProductInfo card using the provided functions. "
                "Be as specific as possible. Avoid adjectives and adverbs. Focus on verifiable/testable features and benefits. "
                "Ensure to provide a list of target customers, key features, relative strengths and weaknesses compared to the base product, "
                "recommended experiments to complete a full SWOT analysis of strengths, weaknesses, opportunities, and threats.",
            ),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o")
    chain = prompt | llm.with_structured_output(CompetingProductInfo)
    competing_product_info = await chain.ainvoke(
        {"docstring": docstring, "base_product": base_product.dict()}, config
    )
    return competing_product_info


async def research_competitor(state: ResearchState, config: RunnableConfig):
    competitors: List[Competitor] = state["competitors"]
    base_product = state["base_product"]
    results = await RunnableLambda(_research_competitor).abatch(
        [
            {"base_product": base_product, "competitor": competitor}
            for competitor in competitors
        ],
        config,
        return_exceptions=True,
    )
    return {"competitors": results}


class SuggestedExperiments(BaseModel):
    suggested_experiments: List[str] = Field(
        ...,
        description="The list of experiments to prioritize, ranked in order of priority.",
    )


async def prioritize_experiments(state: ResearchState, config: RunnableConfig):
    base_product = state["base_product"]
    competitors = state["competitors"]
    experiments = [
        experiment
        for competitor in competitors
        if isinstance(competitor, CompetingProductInfo)
        for experiment in competitor.recommended_experiments
    ]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an experiment prioritization bot. Your task is to rank the provided experiments based on their "
                "potential to maximize information gain for the user and maximize competitive advantage. "
                "Consider the base product and the competitor information when ranking the experiments.\n\n"
                "Base Product Information:\n{base_product}\n\n"
                "Competitor Experiments:\n{experiments}\n\n",
            ),
            (
                "user",
                "Please rank the experiments in order of priority, with the first experiment being the highest priority. "
                "Provide a brief explanation for each experiment's ranking.",
            ),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o")
    chain = prompt | llm.with_structured_output(SuggestedExperiments)
    result: SuggestedExperiments = await chain.ainvoke(
        {"base_product": base_product.dict(), "experiments": experiments}, config
    )
    return {"suggested_experiments": result.suggested_experiments}


async def generate_cpm(state: ResearchState, config: RunnableConfig):
    base_product = state["base_product"]
    competitors = state["competitors"]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a competitive analysis bot. Your task is to generate a competitive profile matrix (CPM) "
                "comparing the base product with its competitors. The CPM should include relevant features and "
                "a comparison of each competitor's product against the base product for each feature. "
                "If information is not available for a particular feature, use 'N/A'.\n\n"
                "Base Product Information:\n{base_product}\n\n"
                "Competitor Information:\n{competitors}\n\n",
            ),
            (
                "user",
                "Please generate the CPM using the provided function. "
                "The CPM should include the base product and all competitors, with a comparison of each product against the base product "
                "for each key success factor. Use a scale of 1-4 to rate each product for each factor, with 1 being the lowest and 4 being the highest. "
                "Ensure that the CPM is complete and accurate.",
            ),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o")
    chain = prompt | llm.with_structured_output(CPM)
    result = await chain.ainvoke(
        {
            "base_product": base_product.dict(),
            "competitors": [
                (
                    competitor.dict()
                    if isinstance(competitor, CompetingProductInfo)
                    else competitor.dict()
                )
                for competitor in competitors
            ],
        },
        config,
    )
    return {"cpm": result}


graph_builder = StateGraph(ResearchState)
graph_builder.add_node("research_company", research_company)
graph_builder.add_node("get_competitors", get_competitors)
graph_builder.add_node("research_competitor", research_competitor)
graph_builder.add_node("prioritize_experiments", prioritize_experiments)
graph_builder.add_node("generate_cpm", generate_cpm)

# Yoho through no loops we go
graph_builder.add_edge(START, "research_company")
graph_builder.add_edge("research_company", "get_competitors")
graph_builder.add_edge("get_competitors", "research_competitor")

# Execute these in parallel
graph_builder.add_edge("research_competitor", "prioritize_experiments")
graph_builder.add_edge("research_competitor", "generate_cpm")
graph_builder.add_edge("prioritize_experiments", END)
graph_builder.add_edge("generate_cpm", END)

graph = graph_builder.compile(
    checkpointer=MemorySaver(),  # interrupt_after=["get_competitors"]
)


# Initialize the research process
async def main():
    config = {"configurable": {"thread_id": "1"}, "max_concurrency": 3}
    events = graph.astream(
        {"base_product": "https://docs.smith.langchain.com/"},
        config,
        stream_mode="values",
    )
    async for event in events:
        print(event)
    snapshot = await graph.aget_state(config)
    final_state: ResearchState = snapshot.values
    print("\n\nDone!")
    print("\n\n")
    print(final_state["suggested_experiments"])
    print(final_state["cpm"].as_markdown())


asyncio.run(main())
