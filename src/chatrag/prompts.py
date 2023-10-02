MOVIE_RETRIEVER_TEMPLATE = """Use the following movies data to find the best matches for the user request in the question overview topic. Rules:
- You can return more than one movie if they are a good match.
- Answer with movie names of the medias and a short text justifying the choice.
- The justification must take into account overview topic matching and vote average score (higher=better).

Media data:
{context}

Question overview topic:
{question}

Example answer:
1. Title: [selected movie title]
- Justification: [Given justification]
- Score: [vote_average]

Movies attending to rules and ordered from best to worst:
""" # noqa