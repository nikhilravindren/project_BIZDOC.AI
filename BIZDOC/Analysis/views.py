from django.shortcuts import render, redirect,get_object_or_404
from django.contrib.auth import login, authenticate,logout
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from .models import company, sentiment,Analysis,shareholders_pattern
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
    c = company.objects.filter(name=request.session['company']).first()
    senti = analyze_news_sentiment(request, request.session['company'], c.sector)
    print(senti)
    return render(request,'chat.html',{"msg":senti})

# Fetch real-time news about a particular company
def fetch_news(company_name):
    api_key = 'c25f86c2ebea425daab83b7472d22bf3'
    url = f'https://newsapi.org/v2/everything?q={company_name}&language=en&apiKey={api_key}'
    
    try:
        response = requests.get(url, timeout=10)  # Set a timeout to avoid hanging requests
        response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        
        news_articles = response.json().get('articles', [])
    except requests.RequestException as e:
        print(f"Error fetching news: {e}")
        return None
    
    results = ""
    for article in news_articles:
        title = article['title']
        description = article.get('description', '')
        article_url = article.get('url', '')

        if company_name.lower() in title.lower() or company_name.lower() in description.lower():
            if article_url:
                # Attempt to fetch the full article content
                article_content = fetch_full_article(article_url)
                text = f"{title}\n{description}\n{article_content}\n\n"
            else:
                text = f"{title}\n{description}\n\n"
            
            results += text
    
    return results

# Scrape the full content of a news article
def fetch_full_article(url):
    try:
        response = requests.get(url, timeout=10)  # Timeout added
        response.raise_for_status()  # Check for a successful status code

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])

        # Return only if content has some meaningful length
        if len(content) > 100:
            return content
        else:
            return "Content is too short or not available."
        
    except requests.RequestException as e:
        print(f"Error fetching full article content: {e}")
        return "Error fetching full article content."


# Sentiment Analysis using FinBERT
class FinBERTSentimentAnalyzer:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone", use_fast=True)
        self.model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        self.model.eval()
    
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512,clean_up_tokenization_spaces=True)
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
    stock_company = company.objects.filter(name=company_name, sector=sector).first()
    
    # Save the sentiment analysis to the database
    senti = sentiment.objects.create(
        user=request.user,
        company=stock_company,
        news=summary,
        sentiment=sentiment_data.get('label'),
        confidence=sentiment_data.get('confidence')
    )

    return senti

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
from django.core.files.storage import FileSystemStorage

# Set your OpenAI API key
openai.api_key = ''

def chating(request):
    # Initialize conversation history in session if it doesn't exist
    if 'conversation_history' not in request.session:
        request.session['conversation_history'] = []

    response_message = ""

    if request.method == "POST":
        user_message = request.POST.get('message')
        uploaded_file = request.FILES['uploaded_file']

        try:
            # If the user sends a message
            if user_message:
                request.session['conversation_history'].append({'role': 'user', 'content': user_message})
                response_message = chat_with_gpt(request.session['conversation_history'], request)

            # If the user uploads a PDF
            elif uploaded_file and uploaded_file.name.endswith('.pdf'):
                fs = FileSystemStorage()
                file_path = fs.save(uploaded_file.name, uploaded_file)
                request.session['file'] = file_path
                keywords = ['financial details', 'CEO message','letter from chairman','letter from ceo','board of directors' 'shareholders pattern', 'future plans', 'key business',]
                pdf_text = extract_text_from_pdf(file_path,keywords)
                if pdf_text:
                    company_name,sector = extract_company_and_sector(request.session['file'])
                    if not company.objects.filter(name=str(company_name), sector=str(sector)):
                        company.objects.create(name=company_name, sector=sector)
                    request.session['company'] = company_name
                    request.session['conversation_history'].append({'role': 'user', 'content': pdf_text})
                    request.session['conversation_history'].append({'role': 'user', 'content': "give me a summary of this company you found in this pdf text ,i need professional answer with a proper format"})
                    response_message = chat_with_gpt(request.session['conversation_history'], request)
                    Analysis.objects.create(user=request.user,company=company.objects.get(name=company_name, sector=sector),summary=response_message)
                else:
                    return JsonResponse({"error": "Failed to extract text from the PDF."}, status=400)

            else:
                return JsonResponse({"error": "Please provide a valid text message or upload a PDF file."}, status=400)

        except Exception as e:
            logging.error(f"Error during chat: {str(e)}")
            response_message = f"An error occurred: {str(e)}"

    return render(request, 'chat.html', {'response': response_message, 'history': request.session['conversation_history']})

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
import fitz  # PyMuPDF

def extract_text_from_pdf(file_path, keywords):
    extracted_text = ""

    # Open the PDF with fitz
    with fitz.open(file_path) as pdf:
        # Loop through all the pages
        for page_num in range(len(pdf)):
            page = pdf.load_page(page_num)  # Load each page
            text = page.get_text("text")  # Extract text from the page
            
            # Check if any of the keywords are in the text
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    extracted_text += f"{keyword}{text}"  # Append found keyword and text
                    break  # Stop after finding a keyword in the page

    return extracted_text


import re
import pdfplumber
from collections import Counter

def extract_company_and_sector(file_path, max_pages=10):
    company_name = None
    sector = None
    potential_names = []

    # Broader regex pattern to catch possible company names
    company_patterns = [
        r'\b[A-Z][A-Za-z]{2,}(?: [A-Z][A-Za-z]{2,})*(?: Ltd|Limited|Corporation|Inc\.|Incorporated|LLC|Company|Co\.)?\b',  # Common suffixes
        r'\b[A-Z][A-Za-z]{2,}(?: [A-Z][A-Za-z]{2,})*\b'  # General capitalized words
    ]
    
    # Sector-related keywords or patterns
    sector_keywords = [
        r'\b(Technology|Healthcare|Finance|Energy|Consumer Goods|Industrials|Materials|Utilities|Telecommunications|Real Estate|Information Technology|Financial Services)\b',
        r'\b(?:operates in the|belongs to the|part of the|is in the) (.+?) sector\b',
        r'\b(?:business area|industry) (.+?)\b'
    ]

    # Open the PDF using pdfplumber
    with pdfplumber.open(file_path) as pdf:
        # Limit to max_pages for faster processing
        for i, page in enumerate(pdf.pages[:max_pages]):
            text = page.extract_text()
            if text:
                # Search for potential company names
                for pattern in company_patterns:
                    matches = re.findall(pattern, text)
                    potential_names.extend(matches)
                
                # Search for sector keywords or patterns
                if not sector:
                    for pattern in sector_keywords:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            sector = match.group(1)  # Extract the sector from the match
                            break  # Stop once sector is found

            # If multiple names are found, use the most frequent one
            if potential_names:
                common_name = Counter(potential_names).most_common(1)
                company_name = common_name[0][0] if common_name else None

            # Stop scanning pages if both company name and sector are found
            if company_name and sector:
                break

    return company_name, sector

def shareholding(request):
    analysis = shareholders_pattern.objects.filter(company=company.objects.filter(name=request.session['company']).first()).first()
    if not analysis:
        conversation_history = request.session.get('conversation_history', [])
        print(request.session['company'])
        user_message = "give me the approximate share holder pattern of" + str(request.session['company']) +" in  square bracket, just need the list with holding percentage and holder only seperated by colon and there should not be other things in the response strictly,and the given instructions should be followed it is a wanning"
        request.session['conversation_history'].append({'role': 'user', 'content': user_message})
        s = chat_with_gpt([{'role': 'user', 'content': user_message}], request)
        print(s)
        numbers = [float(item.strip('%')) for item in re.findall("[0-9.]+%",s)]
        holder = [(item.strip(':')) for item in re.findall("[a-zA-Z(). ]+:",s)]
        print(holder,numbers)
        shareholders_pattern.objects.create(company=company.objects.filter(name=request.session['company']).first(),holder=holder,numbers=numbers)
        return render(request,'chat.html',{"holder":holder,"numbers":numbers})
    else:
        numbers = analysis.numbers
        holder = analysis.holder
        return render(request,'chat.html',{"holder":holder,"numbers":numbers})


def extract_ceo_chairman_message(file_path, max_pages=10):
    # Ensure max_pages is an integer
    if not isinstance(max_pages, int):
        try:
            max_pages = int(max_pages)
        except ValueError:
            raise TypeError("max_pages should be an integer.")
    
    ceo_message = ""
    chairman_message = ""
    
    # Regular expressions for CEO and Chairman sections
    ceo_patterns = [
        r'(CEO\s*Message|Message\s*from\s*the\s*CEO|Letter\s*from\s*the\s*CEO)',
        r'(Chief Executive Officer|Executive\s*Message)'
    ]
    chairman_patterns = [
        r'(Chairman\s*Message|Message\s*from\s*the\s*Chairman|Letter\s*from\s*the\s*Chairman)',
        r'(Chairperson\s*Message|Chairman\s*Statement)'
    ]

    # Open the PDF using pdfplumber
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages[:max_pages]):
            text = page.extract_text()
            if text:
                # Search for CEO message
                if not ceo_message:
                    for pattern in ceo_patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            ceo_message_start = match.end()
                            ceo_message = text[ceo_message_start:].strip()
                            # Append content from next page if needed
                            if i + 1 < len(pdf.pages):
                                ceo_message += pdf.pages[i + 1].extract_text().strip() if pdf.pages[i + 1].extract_text() else ""
                
                # Search for Chairman message
                if not chairman_message:
                    for pattern in chairman_patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            chairman_message_start = match.end()
                            chairman_message = text[chairman_message_start:].strip()
                            if i + 1 < len(pdf.pages):
                                chairman_message += pdf.pages[i + 1].extract_text().strip() if pdf.pages[i + 1].extract_text() else ""

            # Stop if both messages are found
            if ceo_message and chairman_message:
                break

    return ceo_message, chairman_message



def director_message(request):
    keywords = [ 'CEO message','letter from the chairman','letter from the ceo','about company','letter from directors']
    pdf_text1,pdf_text2 = extract_ceo_chairman_message(request.session['file'])
    pdf_text =  pdf_text1+pdf_text2 + "# give me the chairman and ceo message from the pdf text given above"
    request.session['conversation_history'].append({'role': 'user', 'content': pdf_text})
    s = chat_with_gpt([{'role': 'user', 'content': pdf_text}], request)
    return render(request,'chat.html',{"director_msg":s})
