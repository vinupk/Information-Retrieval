# Text to Image Search engine
This project done as part of carriculm project to demonstrate comprehensive understanding and practical application of various concepts and techniques related to information retrieval.

## Objective 
Objective is to building an efficient and accurate Text to image search engine system. I explored ANNOTATION, INDEXING, RETRIEVAL, EVALUATION.

## Important Process 
- `Data Gathering and Textual Surrogates`: web crawler navigates through web pages, identifies images, and downloads them to build a diverse image collection.For each image downloaded, we extract relevant
information from the HTML content, such as alt tags, captions, and surrounding text, to create textual surrogates. These textual surrogates serve as the basis for indexing and retrieval, allowing users to search for images using textual queries. For this project pre-crawled images were used.

- `Annotation`: Image annotation is the process of adding metadata or labels to images to provide additional information about the objects or regions of interest within the image. This information helps in categorizing and understanding the content of the images, making them more meaningful and usable for various applications.

- `Bounding Boxes`: Drawing rectangles around objects of interest, indicating their location within the image.
- `Segmentation Masks`: Creating pixel-level masks that identify the exact boundaries of objects or regions.
- `Key Points`: Marking specific points of interest within an image, such as facial landmarks or keypoints in object detection tasks.
- `Text Annotations`: Adding textual labels or descriptions to images.
- `Indexing`: Indexing is a crucial step in the image search engine, as it enables efficient retrieval of relevant images in response to user queries..
- `Implement ranking models`: Understand the concepts behind the *Vector Space Model, BM25, and the Language Model with Dirichlet Smoothing*, and successfully implement them to rank documents based on their relevance to queries(cran.qry.xml).
- `Evaluate information retrieval system`: Use TREC evaluation metrics like *MAP, P@5, and NDCG@5* to quantitatively assess the performance of the ranking models and make informed decisions about their effectiveness.
- `Analyze and discuss results`: Interpret the evaluation results and provide insightful discussions on the strengths and weaknesses of each model, identifying the most suitable approach for different retrieval scenarios.
## Retrieval
This assignment the retrieval process of finding and presenting relevant information from a large collection of data in response to a user’s query. I used basic HTML page calling python file which uses FLASK framework. using this framework, Java Script invoke python GET method to retrieve relevant images.
