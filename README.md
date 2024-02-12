# Welcome to Movie Wiz!
Movie Wiz is a web-based large langauge model (LLM) that recommends a movie to a user after a series of generated questions. With LangChain framework, the app utilizes 2 OpenAI LLM's: an initial model to generate questions to ask the user and a second model that works in conjunction with a movie database to provide a final movie recomendation to the user based on the conversation history. 

## Preview Image
<img width="1511" alt="Screenshot 2024-01-09 at 5 18 42 PM" src="https://github.com/mlynch019/movie-genie/assets/113787390/e6e670be-61e2-4a57-b993-bba4f440cfdc">

## Set-Up
Clone this repository into VSCode. Execute relevent pip installations. Create 2 txt files named 'api.txt' and 'tmdb_api.txt'. Create accounts for OpenAi and TMDB Developer and generate API codes on free plan. Copy codes into respective txt files. Run app.py and visit http://127.0.0.1:5000 on your local browser. 


### Image Sources, API Usage, and Referenced Documentation
https://www.tickpick.com/blog/hollywood-pantages-theatre-seating-chart-with-seat-numbers/

https://developer.themoviedb.org

https://openai.com/blog/openai-api

https://python.langchain.com/docs/get_started/introduction 
