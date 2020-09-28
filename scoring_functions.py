# -*- coding: utf-8 -*-
"""
Created on Tue May 19 16:02:57 2020

@author: smenta
"""

from collections import Counter
import pandas as pd
import numpy as np
import PyPDF2
#import textract
import re
from textblob import TextBlob
import math 
import matplotlib.pyplot as plt
import pdfplumber
    
   



# Reading data
def read_resume(filename):
    #pdfFileObj = open(filename,'rb')               #open allows you to read the file
    # creating a pdf reader object 
    #pdfReader = PyPDF2.PdfFileReader(pdfFileObj) 

    # printing number of pages in pdf file 
    #print(pdfReader.numPages) 

    # creating a page object 
    #pageObj = pdfReader.getPage(0) 

    # extracting text from page 
    #print(pageObj.extractText()) 
    #text=pageObj.extractText()
    # closing the pdf file object 
    #pdfFileObj.close() 
	with pdfplumber.open(filename) as pdf:
		first_page = pdf.pages[0]
		text = first_page.extract_text()

	return text


# Data cleaning
def text_cleaning(text):
    text=text.lower()
    
    text=text.replace('\n','')
    
    keywords = re.findall(r'[a-zA-Z]\w+',str(text))
    
    #print("Total #.of words in resume :",len(keywords)                               )
    
    return keywords


# identify max n-grams needed
def find_ngrams(data):
    n_grams=[]
    for dom,det in zip(list(data['Domain']),list(data['Details'])):

        kwds=(data.loc[data.Domain==dom])

        kwds=[w.lower().lstrip().replace("'",'') for w in (np.array(kwds.iloc[:,2]))[0].split(",")]
        kwds = [re.sub('[^A-Za-z0-9]+', ' ', w) for w in kwds]
        
        n_grams=n_grams+[np.array([len(kwds[i].split()) for i in range(len(kwds))]).max()]
        #print(det,"--",np.array([len(kwds[i].split()) for i in range(len(kwds))]).max())
    return(np.array(n_grams).max())

# Function to generate n-grams from sentences.
def extract_ngrams(data, num):
    n_grams = TextBlob(data).ngrams(num)
    return [ ' '.join(grams) for grams in n_grams]

# convert list to string
def concatenate_list_data(list):
    result= ''
    for element in list:
        result += " "+str(element)
    return result

# Generate summary based on n-grams
def word_count(words,n_grams):
    wrd_cnt=pd.DataFrame()
    for i in range(n_grams):
        wrds=extract_ngrams(concatenate_list_data(words), i+1)
        d=Counter(wrds)
        df = pd.DataFrame.from_dict(d, orient='index').reset_index()
        df = df.rename(columns={'index':'keywords', 0:'number_of_times_word_appeared'})
        df=df.sort_values(['number_of_times_word_appeared'],ascending=False)
        wrd_cnt=pd.concat([wrd_cnt,df],axis=0)
    return wrd_cnt

	
def word_count_old(words):
    d=Counter(words)
    df = pd.DataFrame.from_dict(d, orient='index').reset_index()
    df = df.rename(columns={'index':'keywords', 0:'number_of_times_word_appeared'})
    df=df.sort_values(['number_of_times_word_appeared'],ascending=False)
    return df


# create dataframe with domain score for each profile
def get_score(data,word_count,filename):
    domain_kwds=pd.DataFrame()
    for dom,det in zip(list(data['Domain']),list(data['Details'])):

        kwds=(data.loc[data.Domain==dom])

        kwds=[w.lower().lstrip().replace("''",'') for w in (np.array(kwds.iloc[:,2]))[0].split(",")]

        #print(df.loc[df['keywords'].isin(kwds)]['number_of_times_word_appeared'].sum(),df['number_of_times_word_appeared'].sum(),df.loc[df['keywords'].isin(kwds)]['number_of_times_word_appeared'].sum()/df['number_of_times_word_appeared'].sum())
        total=word_count.loc[word_count['keywords'].isin(kwds)]['number_of_times_word_appeared'].sum()
        kwds_smry=pd.DataFrame([total],columns=[det])

        domain_kwds=pd.concat([domain_kwds,kwds_smry],axis=1)
    domain_kwds['Candidate']=filename.split("Resume")[0]
    domain_kwds=domain_kwds.set_index("Candidate")
    return domain_kwds

# Magnify data analytics score

def magnify_score(kwds):

    kwds_t=pd.melt(kwds).sort_values(['value'],ascending=False)


    if kwds_t['variable'][0] != "Data Analytics" and kwds_t['value'][0] >10:
        for i in range(1,kwds_t.shape[0]):
            if kwds_t['variable'][i] == "Data Analytics" and kwds_t['value'][i] >10:
                print(i,kwds_t['variable'][i])
                kwds_t['value'][i]= kwds_t['value'][i]*((kwds_t['value'][0]/kwds_t['value'][i])+1)

    kwds_tt=pd.pivot_table(kwds_t,values=["value"],columns=["variable"],aggfunc=[np.sum],fill_value=0)

    kwds_tt.columns=['_'.join(str(s).strip() for s in col if s)[4:] for col in pd.DataFrame(kwds_tt).columns]

    kwds_tt['Candidate']=kwds.index[0]
    kwds_tt=kwds_tt.set_index("Candidate")
    return kwds_tt

# create dataframe with domain specific keywords masterlist ; occurence of each word across domains ; #.of common words across domains
def keywords_smry(keywords):
    domain_kwds=pd.DataFrame()
    for dom,det in zip(list(keywords['Domain']),list(keywords['Details'])):

        kwds=(keywords.loc[keywords.Domain==dom])

        kwds=[w.lower().lstrip().replace("''",'') for w in (np.array(kwds.iloc[:,2]))[0].split(",")]
        keywd_df=pd.DataFrame()
        keywd_df['Keywords']=kwds
        keywd_df['Domain']=dom
        keywd_df['Details']=det
        keywd_df=keywd_df.iloc[:,[1,2,0]]
        domain_kwds=pd.concat([domain_kwds,keywd_df],axis=0)

    
    
    print("-----------------------------------------------")
    keywords_count = (domain_kwds['Keywords'].str.split(expand=True)
                  .stack()
                  .value_counts()
                  .rename_axis('Keywords')
                  .reset_index(name='count'))
    #pd.set_option('display.max_columns', 3)
    
    a=domain_kwds
    n=len(a.Domain.unique())
    dmn_lst=a.Domain.unique()
    rows, cols = (n,n) 
    arr = [[0 for i in range(cols)] for j in range(rows)] 
    for i in range(n):
        for j in range(n):
            a1=list(a.loc[a.Domain==dmn_lst[i]]['Keywords'])
            b1=list(a.loc[a.Domain==dmn_lst[j]]['Keywords'])
            arr[i][j]=len(set(a1) & set(b1))

    cmn_wrds=pd.DataFrame(arr,columns=dmn_lst,index=dmn_lst)
    return(domain_kwds,keywords_count,cmn_wrds)
	


	
	
def create_plots_old(domain_kwds,barchart_data):
	plt.figure(figsize = (12,9))
	plt.subplot(1, 1)
	smry=domain_kwds.melt().sort_values(by=['value'],ascending=False)
	smry=smry.loc[smry.value>0]
	lbls = list(smry.variable)
	sizes = list(smry.value)
	explode = ((0.05,) * len(smry.value)) 
    #plt.title(" Word share by Domain     ")
	plt.pie(sizes, explode=explode, labels=lbls, autopct='%1.1f%%', startangle=45, pctdistance=0.95,labeldistance=1.5)
	centre_circle = plt.Circle((0,0),0.70,fc='white')
	#fig = plt.gcf()
	#fig.gca().add_artist(centre_circle)
	#plt.axis('equal') 
	#plt.show()	
	plt.subplots_adjust(bottom=0.025)
	plt.subplot(1, 2)
	

    #plt.title(" Word share by Domain     ", bbox={'facecolor':'10', 'pad':5})
	#plt.tight_layout()
	
	
	#survey(results, category_names)
	#plt.tight_layout()
	
	return
    
import numpy as np
import matplotlib.pyplot as plt


category_names = ['Strongly disagree', 'Disagree',
                  'Neither agree nor disagree']


def survey(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')( np.linspace(0.15, 0.85, data.shape[1]))
    #category_colors = plt.get_cmap('RdYlGn')

    fig, ax = plt.subplots(figsize=(9.2, 5))
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    #ax.set_xlim(0, np.sum(data, axis=1).max())
    ax.set_xlim(0, 10)

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5,
                label=colname, color=color)
        xcenters = starts + widths / 2

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        for y, (x, c) in enumerate(zip(xcenters, widths)):
            ax.text(x, y, str(int(c)), ha='center', va='center',
                    color=text_color)
    #ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),              loc='lower left', fontsize='small')

    return fig, ax



	



 