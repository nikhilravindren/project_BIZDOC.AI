from django.shortcuts import render, redirect,get_object_or_404
from django.contrib.auth import login, authenticate,logout
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from .models import company, sentiment
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import warnings
import openai
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.db.models import Sum,Count,Avg
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password

# Suppress specific warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="This tokenizer was incorrectly instantiated")


# Create your views here.

def user_user_login(request):
    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        if user is not None and user.is_active:
            if user.is_superuser == False and user.is_staff == False:
                login(request, user)
                return redirect('home')
            elif user.is_superuser == True and user.is_staff == True:
                login(request, user)
                return redirect('home')
            else:
                msg = "You are not autherized for the access!"
                return render(request, 'login.html', {'msg': msg})
        else:
            msg = "Wrong credentials"
            return render(request, 'login.html', {"msg": msg})
    return render(request, 'login.html')

def user_logout(request):
    logout(request)
    return redirect('home')


def create_ac(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        cpassword = request.POST.get('cpassword')
        email = request.POST.get('email')
        
        if User.objects.filter(username=username):
            msg = 'This username already exists please login to continue!'
            return render(request, 'login.html', {'msg': msg})
        
        if password != cpassword:
            msg = "Passwords do not match!"
            return render(request, 'login.html', {'msg': msg})
        
        else:
            hashed_password = make_password(password)
            User.objects.create(username=username, password=hashed_password, email=email)
            msg = "user created successfully ,please login to continue!"
            return render(request, 'login.html', {'msg': msg})
    
    return render(request, 'login.html')



def home(request):
    return render(request, 'home.html')

def dashboard(request):
    sentiments = sentiment.objects.values('sentiment').annotate(count=Count('id'))
    sector = company.objects.values('sector').annotate(count=Count('name'))
    ellam = sentiment.objects.filter(user=request.user)
    return render(request,'dashboard.html',{'sentiment':sentiments,'sector':sector,'all':ellam})


def chat(request):
    return render(request,'chat.html')

def form_submit(request):
    if request.method == "POST":
        company_name = request.POST.get('company')
        sector = request.POST.get('sector')
        print(company)
        if analyze_news_sentiment(request, company_name, sector):
            return render(request,'home.html',{"msg":analyze_news_sentiment(request, company_name, sector)})
    
    return redirect('home')

# Fetch real-time news about a particular company
def fetch_news(company_name):
    api_key = 'c25f86c2ebea425daab83b7472d22bf3'
    url = f'https://newsapi.org/v2/everything?q={company_name}&language=en&apiKey={api_key}'
    response = requests.get(url)
    
    if response.status_code != 200:
        return None
    
    news_articles = response.json().get('articles', [])
    
    results = ""
    for article in news_articles:
        title = article['title']
        description = article.get('description', '')
        article_url = article.get('url', '')
        
        if company_name.lower() in title.lower() or company_name.lower() in description.lower():
            if article_url:
                article_content = fetch_full_article(article_url)
                text = f"{title} {description} {article_content}\n"
            else:
                text = f"{title} {description}\n"
            
            results += text
    
    return results

# Scrape the entire news article content
def fetch_full_article(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        return content
    except Exception as e:
        print(f"Error fetching full article content: {e}")
        return ''


# Sentiment Analysis using FinBERT
class FinBERTSentimentAnalyzer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone", use_fast=True)
        self.model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        self.model.eval()
    
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_label = torch.argmax(probs, dim=1).item()
        sentiment = ["positive", "neutral", "negative"]
        return {"label": sentiment[pred_label], "confidence": probs[0][pred_label].item()}

# Analyze news sentiment and store data in the database
def analyze_news_sentiment(request, company_name, sector):
    news_texts = fetch_news(company_name)
    finbert_analyzer = FinBERTSentimentAnalyzer()
    news_summarizer = FastNewsSummarizer(2000)
    
    if not news_texts:
        print("Failed to fetch news or no news available.")
        return "Failed to fetch news or no news available."
    
    sentiment_data = finbert_analyzer.predict(news_texts)
    summary = news_summarizer.summarize(news_texts)
    
    # Ensure that the company exists in the database
    stock_company, created = company.objects.get_or_create(name=company_name, defaults={'sector': sector})
    
    # Save the sentiment analysis to the database
    sentiment.objects.create(
        user=request.user,
        company=stock_company,
        news=summary,
        sentiment=sentiment_data.get('label'),
        confidence=sentiment_data.get('confidence')
    )

# Fast News Summarizer using T5 model
class FastNewsSummarizer:
    def __init__(self,max_length):
        cache_dir = 'E:/'  
        os.environ['TRANSFORMERS_CACHE'] = cache_dir
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base', cache_dir=cache_dir, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained('t5-base', cache_dir=cache_dir)
        self.max_length = max_length
    
    def summarize(self, text):
        inputs = self.tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=self.max_length, truncation=True)
        
        summary_ids = self.model.generate(
            inputs,
            max_length=self.max_length, 
            min_length=100,   
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary





# Set your OpenAI API key
import openai
import logging
from django.shortcuts import render
from django.http import JsonResponse
from PyPDF2 import PdfReader

# Set your OpenAI API key
openai.api_key = ''

def chating(request):
    # Initialize conversation history in session if it doesn't exist
    if 'conversation_history' not in request.session:
        request.session['conversation_history'] = []

    response_message = ""

    if request.method == "POST":
        user_message = request.POST.get('message')
        uploaded_file = request.FILES.get('uploaded_file')

        try:
            # If the user sends a message
            if user_message:
                request.session['conversation_history'].append({'role': 'user', 'content': user_message})
                response_message = chat_with_gpt(request.session['conversation_history'], request)

            # If the user uploads a PDF
            elif uploaded_file and uploaded_file.name.endswith('.pdf'):
                pdf_text = extract_text_from_pdf(uploaded_file)
                news_summarizer = FastNewsSummarizer(30000)
                pdf_text = news_summarizer.summarize(pdf_text)
                if pdf_text:
                    request.session['conversation_history'].append({'role': 'user', 'content': pdf_text})
                    request.session['conversation_history'].append({'role': 'user', 'content': "give me a summary of this company"})
                    response_message = chat_with_gpt(request.session['conversation_history'], request)
                    print(response_message)
                else:
                    return JsonResponse({"error": "Failed to extract text from the PDF."}, status=400)

            else:
                return JsonResponse({"error": "Please provide a valid text message or upload a PDF file."}, status=400)

        except Exception as e:
            logging.error(f"Error during chat: {str(e)}")
            response_message = f"An error occurred: {str(e)}"

    return render(request, 'chat.html', {'response': response_message, 'history': request.session['conversation_history'][-1]})

def chat_with_gpt(conversation_history, request):
    # Format conversation history for API
    messages = [{"role": msg['role'], "content": msg['content']} for msg in conversation_history]

    # Call the OpenAI API with the conversation history
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages
    )
    
    # Get the assistant's response
    assistant_message = response['choices'][0]['message']['content']
    # Append assistant's response to history
    
    # Save updated history back to session
    request.session['conversation_history'] = conversation_history

    return assistant_message

import logging
from PyPDF2 import PdfReader
import camelot

def extract_text_from_pdf(file):
    try:
        # Step 1: Extract text using PyPDF2
        pdf_reader = PdfReader(file)
        extracted_text = ""

        # Iterate through each page and extract text
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"  # Append extracted text with a newline

        # Step 2: Extract tables using Camelot
        extracted_tables = ""
        try:
            # Use Camelot to extract tables from the PDF (this works well with structured PDFs)
            tables = camelot.read_pdf(file, pages='all')
            for i, table in enumerate(tables):
                extracted_tables += f"\nTable {i + 1}:\n"
                extracted_tables += table.df.to_string(index=False) + "\n"
        except Exception as table_err:
            logging.error(f"Error extracting tables: {str(table_err)}")
            extracted_tables += "\n[Error extracting tables]\n"

        # Combine text and table content
        combined_output = extracted_text.strip() + "\n" + extracted_tables.strip()

        return combined_output
    except Exception as e:
        logging.error(f"Error reading PDF: {str(e)}")
        return None



def shareholding(request):
    user_message = "give me the share holder pattern of TCS company in  square bracket, just need the list with holding percentage only no other text"
    request.session['conversation_history'].append({'role': 'user', 'content': user_message})
    response_message = chat_with_gpt(request.session['conversation_history'], request)
    

