from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from torch import tensor, matmul, argsort
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

# ref: https://huggingface.co/nomic-ai/modernbert-embed-base
model = SentenceTransformer("nomic-ai/modernbert-embed-base")

@app.route('/compute_similarity', methods=['POST'])
def compute_similarity():
    """
    Compute the similarity between a query and multiple documents.
    ---
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            query:
              type: string
              description: The query string.
            documents:
              type: array
              items:
                type: string
              description: A list of documents to compare against the query.
    responses:
      200:
        description: A list of documents ranked by similarity.
        schema:
          type: object
          properties:
            best_matches:
              type: array
              items:
                type: object
                properties:
                  document:
                    type: string
                  similarity:
                    type: number
      400:
        description: Error if required fields are missing.
      500:
        description: Internal server error.
    """
    try:
        data = request.get_json()
        query = data.get('query', None)
        documents = data.get('documents', [])

        if not query or not documents:
            return jsonify({"error": "Both 'query' and 'documents' fields are required and cannot be empty."}), 400

        query_embedding = model.encode([query])
        doc_embeddings = model.encode(documents)

        query_tensor = tensor(query_embedding)
        doc_tensor = tensor(doc_embeddings)

        similarities = matmul(query_tensor, doc_tensor.T).squeeze(0)

        best_indices = argsort(similarities, descending=True)
        best_matches = [{"document": documents[i], "similarity": similarities[i].item()} for i in best_indices]

        response = {
            "best_matches": best_matches
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
