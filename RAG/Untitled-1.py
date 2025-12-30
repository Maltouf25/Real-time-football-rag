# %%
# {"id": "MOR_MLI_2025_10_001", 
# "match_id": "MOR_MLI_2025", 
# "minute": 10, 
# "text": "Lassana Coulibaly (Mali) wins a free kick in the defensive half.", 
# "event_type": "free kick won", 
# "team": "Mali", 
# "timestamp": null}


# %%
!pip install qdrant-client sentence-transformers


# %%
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# %%
from qdrant_client import QdrantClient

qdrant = QdrantClient(
    url="https://c140de78-7617-4888-b2ac-c2cf101a4bdc.europe-west3-0.gcp.cloud.qdrant.io",  # or cloud URL
    check_compatibility=False,
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.7yZMCtz3hO8hwL5TMFjIRzCVhOWYOsevgzNM0bVMwHs"
)

COLLECTION = "play_by_play_collection"

# %%
import uuid

def store_event(event: dict):
    vector = embedder.encode(event["text"]).tolist()

    point = {
        "id": str(uuid.uuid4()),
        "vector": vector,
        "payload": {
            "match_id": event["match_id"],
            "minute": event["minute"],
            "event_type": event["event_type"],
            "team": event["team"],
            "text": event["text"]  
        }
    }

    qdrant.upsert(
        collection_name=COLLECTION,
        points=[point]
    )


# %%
event = {"id": "MOR_MLI_2025_10_001", 
"match_id": "MOR_MLI_2025", 
"minute": 10, 
"text": "Lassana Coulibaly (Mali) wins a free kick in the defensive half.", 
"event_type": "free kick won", 
"team": "Mali", 
}

# %%
store_event(event)

# %%
import json
with open("play_by_play.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 10:
            break

        event = json.loads(line)

        vector = embedder.encode(event["text"]).tolist()

        point = {
            "id": str(uuid.uuid4()),
            "vector": vector,
            "payload": {
                "match_id": event["match_id"],
                "minute": event["minute"],
                "event_type": event["event_type"],
                "team": event["team"],
                "text": event["text"]
            }
        }

        qdrant.upsert(
            collection_name=COLLECTION,
            points=[point]
        )

# %%


# %%
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

def retrieve_events(user_query,match_id,
        top_k):
    query_vector = embedder.encode(user_query).tolist()
    results = qdrant.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="match_id",
                    match=MatchValue(value=match_id)
                ),
                # Optional time filter:
                FieldCondition(
                    key="minute",
                    range=Range(gte=5, lte=10)
                )
            ]
        )
    )

    contexts = [hit.payload["text"] for hit in results]
    return contexts


# %%
def build_prompt(user_query,contexts):    
    prompt = f"""SYSTEM:
    You are a live football match assistant.
    Use ONLY the context to answer.

    CONTEXT:
    {chr(10).join(contexts)}

    QUESTION:
    {user_query}

    ANSWER:
    """
    return prompt

# %%
!pip install mistralai


# %%
from mistralai import Mistral

client = Mistral(api_key="R7VWsGtxnEFQBOP2WOgzmU8GbB2xjxgs")

def generate_answer(prompt: str):
    response = client.chat.complete(
        model="mistral-7b-instruct",
        messages=[
            {"role": "system", "content": "You are a live football match assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.2
    )
    return response.choices[0].message.content

# %%
def answer_question_rag(
    question: str,
    match_id: str,
    top_k: int = 5):
    # 1. Embed question
    query_vector = embedder.encode(question).tolist()

    # 2. Retrieve relevant events
    contexts = retrieve_events(
        user_query=query_vector,
        match_id=match_id,
        top_k=top_k
    )

    # 3. Build prompt
    prompt = build_prompt(contexts, question)

    # 4. Generate answer
    return generate_answer(prompt)


# %%
answer = answer_question_rag(
    question="Who won the last foul",
    match_id="MOR_MLI_2025"
)

print(answer)



