import os
from flask import Flask, render_template, request, jsonify
from langchain.prompts.chat import HumanMessage
from langchain.agents import load_tools, initialize_agent
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType
from pydantic import BaseModel
from pydantic import Field
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler;
from langchain.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.tools import YouTubeSearchTool
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
import requests
import json

#reading txt files for API keys (OPENAI and SerpAPI). Gitignore to protect leak
with open("api.txt",'r') as token_file:
    open_ai_key = token_file.read()
with open("serpapi.txt",'r') as token_file:
    serp_ai_key = token_file.read()


os.environ['OPENAI_API_KEY'] = open_ai_key
os.environ['SERPAPI_API_KEY'] = serp_ai_key

#initialize first chat model. Will ask user questions and gather information about what movie they are interested in.
#Higher temp for greater 'creativity' when generating questions to ask the user
chat = OpenAI(temperature=0.6)

#initialize second chat model. Will take input of conversation (history) and movie API to make recommendation. 
#Lower temp because should be strict based on prompt and given conversation history
chat2 = OpenAI(temperature=0.1)

#loading google search into first chat. Might not require
tools = load_tools(["serpapi"],llm=chat)

#converstation history
memory = ConversationBufferMemory(memory_key="chat_history")

#First chain prompt/template
template1 = """
    You are part of a movie recommending computer program called "Movie Wiz". 
    Your job is to ask a question to gather information about what kind of movie the user likes.
    Ensure your questions are completely different from previously asked questions in the chat history.
    Do not ask a question that was already asked based on the chat history.
    If no chat history exists yet, you should make a question that is in a similar style to these questions:
        Examples:
        -What genre of movies do you enjoy watching?
        -What genre of movies do you enjoy watching?
        -Are you looking for something light-hearted and funny, or a more serious and thought-provoking film?
        -What actors do you enjoy watching?
        -Do you prefer contemporary movies or classics?
        - Do you like movies about the ocean?
        -Are you open to foreign language films, or do you prefer movies in your native language?
        - Do you have a favorite director?
        -How long of a movie are you in the mood for? Do you want a shorter film or are you open to longer ones?
        -What genre of movies do you enjoy watching?
        -Would you like a movie with a strong plot and twists, or one that focuses more on character development?

    History (if there is any):
    {chat_history}
    Your Question:
    """

#load template to chain
chain1_sm_prompt = SystemMessagePromptTemplate.from_template(template1)

#Chain 1 overall prompt (combining the human and the system message)
chain1_prompt = ChatPromptTemplate.from_messages([chain1_sm_prompt])

#Chain 1 
#put the prompt into the chat model and stores the interaction in memory when called
chain1a = LLMChain(llm=chat, prompt=chain1_prompt, memory=memory) 


#initialize flask and enable access to html files and images
app = Flask(__name__, template_folder="public", static_folder = "images")

@app.route('/', methods = ['GET', 'POST'])
def index():
    #initializing variable to count number of questions asked 
    global index1
    index1 = 0

    #running first chain: greets user and asks initial question
    first_message = chain1a.run({"chat_history": "Hello", "input": "_"}) 

    #sending response to html file to display
    return render_template('index.html', first_message = first_message)

@app.route('/showMovie')
def showMovie():
    
    #Make call to TMDB database based on answers

    # for reference: base_url = 'https://api.themoviedb.org/3/'
    #                base_photo_url = 'https://image.tmdb.org/t/p/original/'
    #                testing_url = "https://api.themoviedb.org/3/discover/movie?include_adult=false&include_video=false&language=en-US&page=1&sort_by=popularity.desc"
    
    
    
    #Second chain template: gives model chat history and the movie API documentation. 
    #Model should generate list of URL's of movies based on the user provided information in the chat_history
    chain2_template = """
    You are given the below API Documentation and a previous conversation between a user and AI:

    API documentation:
    Endpoint: https://api.themoviedb.org/3
    GET /discover/movie

    This API is for searching movies.

    Example call-
    https://api.themoviedb.org/3/discover/movie?include_adult=true&include_video=true&language=en-US&page=1&region=US&sort_by=popularity.desc&with_genres=12

    Parameters table:
    include_adult | boolean | Choose whether to include adult content in the results (not child friendly). default | optional
    include_video | boolean | always say true | required
    language | string | always set to "en-US" | required
    primary_release_date.gte | int |  Filter and only include movies that have a primary release date that is greater or equal to the specified value. | optional
    watch_region | string | Chooses the region the user is watching from. Always use "US" | required
    with_genres | string | Comma separated list of genre ids that you want to include in the results. These are the id's to choose from: id: 28 name: Action - id: 12 name: Adventure - id: 16 name: Animation - id: 35 name: Comedy - id: 80 name: Crime - id: 99 name: Documentary - id: 18 name: Drama - id: 10751 name: Family - id: 14 name: Fantasy - id: 36 name: History - id: 27 name: Horror - id: 10402 name: Music - id: 9648 name: Mystery - id: 10749 name: Romance - id: 878 name: Science Fiction - id: 10770 name: TV Movie - id: 53 name: Thriller - id: 10752 name: War - id: 37 name: Western | required
    with_runtime.lte | int | Shows movies with a runtime less then the specified number of minutes | optional
    with_watch_providers | pipe separated list of movie provider ids that will have the movies the user wants. These are the id's to choose from: provider_name: "Amazon Prime Video", provider_id: 119 - provider_name: "Netflix", provider_id: 8 - provider_name: "Disney Plus", provider_id: 337 - provider_name: "Apple TV Plus", provider_id: 350 - provider_name: "YouTube", provider_id: 192 - provider_name: "Paramount Plus", provider_id: 531 | optional

    Previous conversation:
    {chat_history}

    Using this documentation and the user history, generate the full API url to call the API for a movie the user would like based on their chat_history.
    You should build the API url in order to get a response that includes as many aspects of a movie the user would like. 
    Please output only the reqired URL.
    """

    #formating the template into a prompt (so it can be passed into other langchain components)
    chain2_sm_prompt = SystemMessagePromptTemplate.from_template(chain2_template) 
    #Turning that prompt from a system message prompt to an overall prompt (Because we are using a chat model instead of llm)
    chain2_prompt = ChatPromptTemplate.from_messages([chain2_sm_prompt])  
    #initializing chain
    chain2 = LLMChain(llm = chat2, prompt=chain2_prompt, verbose = True) 

    #running chain for url
    url = chain2.run(chat_history = memory) 
   

    #A different format of API key for the TMDB
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJmYjczZWUwZmZhN2QxYmZkODMzOWI4NzAwZWYzZTZmOSIsInN1YiI6IjY0N2M4NmU3ZTMyM2YzMDE0ODE3MjQyNiIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.0_jt4N-jBbqM7yESJ-N3tiu9EXlpsHio3BiCFhm9iYo"
    }

    #making the request and turning the string into a json  
    request = (requests.get(url, headers=headers)).json() 
    #navigating through the json tree
    movies = request['results'] 

    #the movie_list we will pass on
    filtered_movie_list = []
    #limiting it to the first 15
    for movie in movies[:15]:
        movie_data = {
            'title': movie['title'],
            'id': movie['id'],
            'overview': movie['overview'],
            'poster_path': movie['poster_path'],
            'backdrop_path': movie['backdrop_path'],
            'vote_average': movie['vote_average']
        }
        filtered_movie_list.append(movie_data)



    #____________________________MAKING SUGGESTION FROM THE LIST____________________________________

    #Used movie "list" so they aren't repeated in the suggetions.
    used_movie_list = " "

    #Instuctions to the llm. Will be formated into the prompt
    chain3_template = """
    As a movie recommending chat bot, your task is to find a movie from the unsorted movie list that matches the user's preferences based on the previous conversation. You should only consider the movie overviews and how they align with the user's preferences. Output the information of the selected movie in JSON format as shown below.

    To accomplish this, follow these steps:

    1. Review the previous conversation with the user to understand their preferences.
    2. Examine the unsorted movie list provided. 
    3. Analyze the overviews of the movies in the list and compare them with the user's preferences. Search for keywords, genres, and features.
    4. Select the movie that best matches the user's preferences based on the overviews.
    5. Output the following information of the recommended movie in JSON format:

    Previous Conversation:
    {chat_history}

    Unsorted Movie List:
    {movie_list}

    Output Instructions:
    {format_instructions}

    Note: The movie should not be from the list of used movies.
    If there are any movies you shouldn't mention they will be listed bellow:
    {used_movies}
    """
    #template for llm ouput
    class movie_json(BaseModel):
        title: str = Field(description="title of the movie")
        id: str = Field(description="id of the movie")
        poster_path: str = Field(description="url to the poster")
        backdrop_path: str = Field(description="url to the backdrop")
        reasoning: str = Field(description="A short reasoning as to why you choose this movie")
    parser = PydanticOutputParser(pydantic_object=movie_json)


    #formatting prompt to include parser
    chain3_prompt = PromptTemplate(
        template=chain3_template,
        input_variables=["movie_list","chat_history","used_movies"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    #_____________________________________________________________________________________________________________________
  
    #formated prompt
    formated_chain3_prompt = chain3_prompt.format_prompt(movie_list = filtered_movie_list, chat_history = memory, used_movies = used_movie_list)

    #parser only works with a normal llm instead of a chat one. Intitialized here
    llm = OpenAI(temperature=0)

    #call to the llm
    output = llm(formated_chain3_prompt.to_string())
    #correcting output (if necessary) to correct format
    movie_rec = parser.parse(output)

    # Serialize the movie_rec object to a JSON string
    movie_rec_json_str = json.dumps(movie_rec.dict())
    # Parse the JSON string into a JSON object (better navigation)
    movie_rec_json = json.loads(movie_rec_json_str)

    #Getting the trailer from youtube. 
    tool = YouTubeSearchTool()
    videos = tool.run(str(movie_rec_json['title']) + "offical trailor")
    videos_list = videos.split(',')  
    youtube_link = "youtube.com" + videos_list[0].strip("[]") .strip("'")

    #adding the movie into a list so it isn't repeated again if the user doesn't like the choice. 
    used_movie_list += " , "
    used_movie_list += str(movie_rec_json['title'])

    movie = str(movie_rec_json['title'])
    poster_link = 'https://image.tmdb.org/t/p/original/' + str(movie_rec_json['poster_path'])
    #pass necessary info to html file
    return render_template('showMovie.html', movie = movie, youtube_link = youtube_link, poster_link = poster_link)

@app.route('/process_message', methods=['POST'])
def process_message():

    #allowing model to ask 4 questions then initializing movie search process. 
    global index1
    if(index1 < 4):
        user_message = request.form['message']
        bot_response = chain1a.run({"chat_history": "Hello", "input" : user_message})
        index1 += 1
    if(index1 == 5):
        return jsonify(message="You have reached the max number of questions. A movie will be recommended shortly.")
    
    return jsonify(message=bot_response)


if __name__ == '__main__':
    app.run()
