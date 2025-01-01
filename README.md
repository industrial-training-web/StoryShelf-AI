# StoryShelf-AI

 - http://localhost:5000/
 - http://localhost:5000/apidocs/

## Example
```curl
curl -X POST http://127.0.0.1:5000/compute_similarity \
-H "Content-Type: application/json" \
-d '{
    "query": "search_query: maifee is a good boy",
    "documents": [
        "search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten",
        "search_document: Principal Component Analysis (PCA) is another dimensionality reduction technique"
    ]
}'
```