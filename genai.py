import os
import openai
import json
import pandas as pd
import requests
import time
import re
import openai



class GenAI:
    """
    A class for interacting with the OpenAI API to generate text, images, video descriptions,
    perform speech recognition, and handle basic document processing tasks.

    Attributes:
    ----------
    client : openai.Client
        An instance of the OpenAI client initialized with the API key.
    """
    def __init__(self, openai_api_key):
        """
        Initializes the GenAI class with the provided OpenAI API key.

        Parameters:
        ----------
        openai_api_key : str
            The API key for accessing OpenAI's services.
        """
        self.client = openai.Client(api_key=openai_api_key)
        self.openai_api_key = openai_api_key

    def generate_text(self, prompt, instructions='You are a helpful AI named Jarvis', model="gpt-4o-mini", output_type='text', temperature =1):
        """
        Generates a text completion using the OpenAI API.

        This function sends a prompt to the OpenAI API with optional instructions to guide the AI's behavior. 
        It supports specifying the model and output format, and returns the generated text response.

        Parameters:
        ----------
        prompt : str
            The user input or query that you want the AI to respond to.
        
        instructions : str, optional (default='You are a helpful AI named Jarvis')
            System-level instructions to define the AI's behavior, tone, or style in the response.
        
        model : str, optional (default='gpt-4o-mini')
            The OpenAI model to use for generating the response. You can specify different models like 'gpt-4', 'gpt-3.5-turbo', etc.
        
        output_type : str, optional (default='text')
            The format of the output. Typically 'text', but can be customized for models that support different response formats.

        Returns:
        -------
        str
            The AI-generated response as a string based on the provided prompt and instructions.

        Example:
        -------
        >>> response = generate_text("What's the weather like today?")
        >>> print(response)
        "The weather today is sunny with a high of 75°F."
        """
        completion = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": output_type},
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt}
            ]
        )
        response = completion.choices[0].message.content
        response = response.replace("```html", "")
        response = response.replace("```", "")
        return response


 

    def remove_urls(self, text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def display_tweet(self,text='life is good', screen_name='AI Persona'):
        display_html = f'''
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                .tweet {{
                    background-color: white;
                    color: black;
                    border: 1px solid #e1e8ed;
                    border-radius: 10px;
                    padding: 20px;
                    max-width: 500px;
                    margin: 20px auto;
                    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
                    box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
                }}
                .user strong {{
                    color: #1da1f2;
                }}
                .tweet-text p {{
                    margin: 0;
                    line-height: 1.5;
                }}
            </style>
        </head>
        <body>
            <div class="tweet">
                <div class="user">
                    <strong>@{screen_name}</strong>
                </div>
                <div class="tweet-text">
                    <p>{text}</p>
                </div>
            </div>
        </body>
        </html>
        '''
       
        return display_html
