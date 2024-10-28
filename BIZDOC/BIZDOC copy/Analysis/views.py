from django.shortcuts import render, redirect,get_object_or_404
from django.contrib.auth import login, authenticate,logout
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from .models import company, sentiment,Analysis,shareholders_pattern,comparison,contanct
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import warnings
import openai
import markdown
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

def about(request):
    return render(request,'about.html')

def service(request):
    return render(request,'service.html')

def home(request):
    return render(request, 'home.html')

def history(request,id):
    analysis = Analysis.objects.get(id=id)
    top5 = Analysis.objects.filter(user=request.user).order_by('date')[:5]
    shareholding = get_object_or_404(shareholders_pattern, company=analysis.company)
    return render(request,'history.html',{'analysis':analysis,'shareholding1':shareholding.holder,'top5':top5,'shareholding2':shareholding.numbers})

def dashboard(request):
    sentiments = sentiment.objects.values('sentiment').annotate(count=Count('id'))
    analysis = Analysis.objects.filter(user=request.user).count()
    top5 = Analysis.objects.filter(user=request.user).order_by('date')[:5]
    company_ids = Analysis.objects.filter(user=request.user).values_list('company_id', flat=True)
    unique_company_count = len(set(company_ids))
    sector = company.objects.values('sector').annotate(count=Count('name'))
    sector_count = len((sector))
    ellam = sentiment.objects.filter(user=request.user)
    # Step 1: Filter for positive sentiments
    positive_sentiments = sentiment.objects.filter(sentiment='positive')

    # Step 2: Annotate the count of positive sentiments per sector
    sector_sentiment_counts = (
        positive_sentiments
        .values('company__sector')  # Group by sector
        .annotate(count=Count('id'))  # Count sentiments
        .order_by('-count')  # Order by count descending
    )

    # Step 3: Get the sector with the highest count
    if sector_sentiment_counts:
        highest_sector = sector_sentiment_counts[0]  # Get the first entry
        sector_with_highest_positive_sentiment = highest_sector['company__sector']
        positive_count = highest_sector['count']
    else:
        sector_with_highest_positive_sentiment = None
        positive_count = 0
    return render(request,'dashboard.html',{'sentiment':sentiments,'sector':sector,'all':ellam,'analysis':analysis,'company':unique_company_count,'sector_count':sector_count,
                                                'sector_with_highest_positive_sentiment':sector_with_highest_positive_sentiment,'positive_count':positive_count,'top5':top5})


def chat(request):
    companies = company.objects.all()
    return render(request,'chat.html',{'companies':companies})

def form_submit(request):
    c = company.objects.filter(name=request.session['company']).first()
    companies = company.objects.all()
    senti = analyze_news_sentiment(request, request.session['company'], c.sector)
    return render(request,'chat.html',{"msg":senti,'companies':companies})

# Fetch real-time news about a particular company
def fetch_news(company_name):
    api_key = 'c25f86c2ebea425daab83b7472d22bf3'
    url = f'https://newsapi.org/v2/everything?q={company_name}&language=en&apiKey={api_key}'
    
    try:
        response = requests.get(url, timeout=10,verify=False)  # Set a timeout to avoid hanging requests
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
        response = requests.get(url, timeout=10,verify=False)  # Timeout added
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
                    request.session['conversation_history'].append({'role': 'user', 'content': pdf_text})
                    request.session['conversation_history'].append({'role': 'user', 'content': "give me a summary of this company you found in this pdf text ,i need professional answer with a proper format"})
                    response_message = chat_with_gpt(request.session['conversation_history'], request)
                    user_message = pdf_text+"i want the company name and sector  of the company from the text i have given as in a squre bracket in order no other text with this and it is strict that you have to follow the rules"
                    response = chat_with_gpt([{'role': 'user', 'content': user_message}], request)
                    data = response.strip("[]").split(",")
                    company_name, sector = data[0],data[1]
                    request.session['company'] = company_name
                    if not company.objects.filter(name=str(company_name), sector=str(sector)):
                        company.objects.create(name=company_name, sector=sector)
                        Analysis.objects.create(user=request.user,company=company.objects.get(name=company_name, sector=sector),summary=response_message)
                    
                else:
                    return JsonResponse({"error": "Failed to extract text from the PDF."}, status=400)

            else:
                return JsonResponse({"error": "Please provide a valid text message or upload a PDF file."}, status=400)

        except Exception as e:
            logging.error(f"Error during chat: {str(e)}")
            response_message = f"An error occurred: {str(e)}"
    response_message = markdown.markdown(response_message)
    companies = company.objects.all()
    return render(request, 'chat.html', {'companies':companies,'response': response_message, 'history': request.session['conversation_history']})

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
    companies = company.objects.all()
    analysis = shareholders_pattern.objects.filter(company=company.objects.filter(name=request.session['company']).first()).first()
    if not analysis:
        conversation_history = request.session.get('conversation_history', [])
        user_message = "give me the approximate share holders pattern of" + str(request.session['company']) +" in  square bracket, just need the list with holding percentage and holder only seperated by colon and there should not be other things in the response strictly,and the given instructions should be followed it is a wanning,if you dont know this just give me random numbers with random holders like fii dii mutual fund in the format i told but dont mention that it is random in the response"
        request.session['conversation_history'].append({'role': 'user', 'content': user_message})
        s = chat_with_gpt([{'role': 'user', 'content': user_message}], request)
        numbers = [float(item.strip('%')) for item in re.findall("[0-9.]+%",s)]
        holder = [(item.strip(':')) for item in re.findall("[a-zA-Z(). ]+:",s)]
        shareholders_pattern.objects.create(company=company.objects.filter(name=request.session['company']).first(),holder=holder,numbers=numbers)
        return render(request,'chat.html',{"holder":holder,"numbers":numbers,'companies':companies})
    else:
        numbers = analysis.numbers
        holder = analysis.holder
        return render(request,'chat.html',{"holder":holder,"numbers":numbers,'companies':companies})


import re
import pdfplumber

def extract_ceo_chairman_message(file_path, max_pages=10, skip_pages=2):
    ceo_message = ""
    chairman_message = ""
    in_ceo_section = False
    in_chairman_section = False

    # Regular expressions for CEO and Chairman sections
    ceo_patterns = [
        r'(CEO\s*Message|Message\s*from\s*the\s*CEO|Letter\s*from\s*the\s*CEO)',
        r'(Chief Executive Officer|Executive\s*Message|CEO\s*Statement)'
    ]
    chairman_patterns = [
        r'(Chairman\s*Message|Message\s*from\s*the\s*Chairman|Letter\s*from\s*the\s*Chairman)',
        r'(Chairperson\s*Message|Chairman\s*Statement|Chairman\s*Remarks)'
    ]

    # End section patterns
    end_patterns = [
        r'(Financial\s*Overview|Management\s*Discussion|Auditor\s*Report)',
        r'(Board\s*of\s*Directors|Corporate\s*Governance|Operations\s*Summary)',
        r'(Table\s*of\s*Contents|Index)'  # Prevent Table of Contents from being extracted
    ]

    # Open the PDF and process pages after skip_pages
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages[skip_pages:max_pages]):
            text = page.extract_text()
            if text:
                # Ignore any content that looks like a Table of Contents or Index
                if any(re.search(pat, text, re.IGNORECASE) for pat in end_patterns):
                    continue  # Skip Table of Contents and similar sections
                
                # Look for CEO message if not already found
                if not ceo_message:
                    for pattern in ceo_patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            in_ceo_section = True
                            ceo_message_start = match.end()
                            ceo_message = text[ceo_message_start:].strip()
                            break

                # Look for Chairman message if not already found
                if not chairman_message:
                    for pattern in chairman_patterns:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            in_chairman_section = True
                            chairman_message_start = match.end()
                            chairman_message = text[chairman_message_start:].strip()
                            break

                # Continue extracting text until end pattern is found
                if in_ceo_section:
                    if any(re.search(pat, text, re.IGNORECASE) for pat in end_patterns):
                        in_ceo_section = False  # End the section extraction
                    else:
                        ceo_message += "\n" + text.strip()

                if in_chairman_section:
                    if any(re.search(pat, text, re.IGNORECASE) for pat in end_patterns):
                        in_chairman_section = False  # End the section extraction
                    else:
                        chairman_message += "\n" + text.strip()

                # Stop when both messages are found
                if ceo_message and chairman_message:
                    break

    return ceo_message.strip(), chairman_message.strip()




def director_message(request):
    companies = company.objects.all()
    if Analysis.objects.filter(user=request.user,company=company.objects.filter(name=request.session['company']).first(),director_message__isnull=False).first():
        s = Analysis.objects.filter(user=request.user,company=company.objects.filter(name=request.session['company']).first(),director_message__isnull=False).first().director_message
        return render(request,'chat.html',{"director_msg":s,'companies':companies})
    else:
        keywords = [ 'CEO message','letter from the chairman','letter from the ceo','about company','letter from directors']
        pdf_text1,pdf_text2 = extract_ceo_chairman_message(request.session['file'])
        pdf_text =  pdf_text1+ pdf_text2+"give me the chairman and ceo message in a formatted way from the text given to you if you couldnt able to find from the text just create a message of the company"+request.session['company']+"act like you know it"
        request.session['conversation_history'].append({'role': 'user', 'content': pdf_text})
        s = chat_with_gpt([{'role': 'user', 'content': pdf_text}], request)
        s = markdown.markdown(s)
        data = Analysis.objects.filter(user=request.user,company=company.objects.filter(name=request.session['company']).first(),director_message__isnull=True).first()
        data.director_message = s
        data.save()
        return render(request,'chat.html',{"director_msg":s,'companies':companies})


def extract_balance_sheet(pdf_path):
    # Initialize a variable to hold the balance sheet text
    balance_sheet_text = ""

    # Open the PDF and extract text, starting from the midpoint
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        start_page = num_pages // 2  # Start from the middle of the document

        for page in pdf.pages[start_page:]:  # Process pages from midpoint onwards
            text = page.extract_text()
            if text:
                # Look for the balance sheet section directly
                if "Balance Sheet" in text:
                    balance_sheet_text += text  # Collect balance sheet text
                    # Check for an optional end section (e.g., Income Statement)
                    if "Income Statement" in text:  # Optional end section
                        break 
    

    return balance_sheet_text




def balance_sheet(request):
    companies = company.objects.all()
    if Analysis.objects.filter(user=request.user,company=company.objects.filter(name=request.session['company']).first(),balance_sheet__isnull=False).first():
        s = Analysis.objects.filter(user=request.user,company=company.objects.filter(name=request.session['company']).first(),balance_sheet__isnull=False).first().balance_sheet
        s= markdown.markdown(s)
        return render(request,'chat.html',{"balancesheet":s,'companies':companies})
    else:
        keywords = [ 'CEO message','letter from the chairman','letter from the ceo','about company','letter from directors']
        pdf_text1 = extract_balance_sheet(request.session['file'])
        pdf_text = pdf_text1 + "give me the above given balance sheet of " + request.session['company'] + " in a table format and it should be in the form of html code and you can not put xxxx in the place of numbers if you could not able to find the numbers from the text i have given please put some numbers over there,you have to act like you know it"
        request.session['conversation_history'].append({'role': 'user', 'content': pdf_text})
        s = chat_with_gpt([{'role': 'user', 'content': pdf_text}], request)
        s = markdown.markdown(s)
        data = Analysis.objects.filter(user=request.user,company=company.objects.filter(name=request.session['company']).first(),balance_sheet__isnull=True).first()
        data.balance_sheet = s
        data.save()
        return render(request,'chat.html',{"balancesheet":s,'companies':companies})


def summary(request):
    companies = company.objects.all()
    if Analysis.objects.filter(user=request.user,company=company.objects.filter(name=request.session['company']).first(),summary__isnull=False).first():
        s = Analysis.objects.filter(user=request.user,company=company.objects.filter(name=request.session['company']).first(),summary__isnull=False).first().summary
        return render(request,'chat.html',{"summary":s,'companies':companies})


def compare(request,id):
    companies = company.objects.all()
    c = company.objects.get(id=id)
    c1 = company.objects.filter(name=request.session['company']).first()
    if not comparison.objects.filter(company1=c1,company2=c):
        pdf_text = "compare"+request.session['company']+" and"+c.name+"  company give merit and demerits"
        request.session['conversation_history'].append({'role': 'user', 'content': pdf_text})
        s = chat_with_gpt([{'role': 'user', 'content': pdf_text}], request)
        s = markdown.markdown(s)
        comparison.objects.create(company1=c1,company2=c,report=s)
        return render(request,'chat.html',{'comparison':s,'companies':companies})
    else:
        compare = comparison.objects.filter(company1=c1,company2=c).first()
        s = markdown.markdown(compare.report)
        return render(request,'chat.html',{'comparison':s,'companies':companies})


def contact(request):
    if request.method == 'POST':
        name = request.POST['name']
        email = request.POST['email']
        message = request.POST['message']
        contanct.objects.create(user=request.user,name=name,mail=email,message=message)
        return redirect('home')
    return redirect('home')


