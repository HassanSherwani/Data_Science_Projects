# -*- coding: utf-8 -*-
"""
Web Scrapping project from newegg website
"""

# Importing key libraries

import requests
import bs4 
from urllib.request import urlopen as ureq
from bs4 import BeautifulSoup as soup

# Reading the web page into Python

my_url= 'https://www.newegg.com/Video-Cards-Video-Devices/Category/ID-38'

uclint= ureq(my_url)
page_html= uclint.read()
page_html[0:500]
uclint.close()

r = requests.get('https://www.newegg.com/Video-Cards-Video-Devices/Category/ID-38')
print(r.text[0:500])

# Parsing the HTML using Beautiful Soup

page_soup= soup(page_html,"html.parser")
page_soup.head()
print(page_soup.h1) # it shows header on top of webpage
print(page_soup.p)
print (page_soup.body.span)

contain = page_soup.find_all('div', attrs={'class':'item-container'})
len(contain)
contain[0:3]

# Taking very 1st 

con=contain[0]
con
con.a # we want title

con.div

# Now to get "GIGAbyte" , I need to get title. It is not tag. Title is attribute inside image tag
brand=con.div.div.a.img["title"]
# Now 1st column of brand has been extracted. Let's make more.

con.find_all("a",{"class":"item-title"})

con.find("a")

con.find("a",{"class":"item-title"}).text #to get 1st object

# make this as find command for loop

title_container[0].text
title_container[0].i #just to check if i contains given info or not. It didn't contain so, we used text.

# Now we have got our title as well
title=title_container[0].text
title

# Going for third one. Which is shipping

con.find_all("li",{"class":"price-ship"})
ship[0].text

# there is some extra stuff \r\n so , to remove that, I ll use strip
shiping_price= ship[0].text.strip()
shiping_price

con.find("li",{"class":"price-ship"}).text.strip()
#make this our final result and use for loop.

#Finally combine them all
records = []
for con in contain:
    brand=con.div.div.a.img["title"]
    title=con.find("a",{"class":"item-title"}).text
    shiping_price= con.find("li",{"class":"price-ship"}).text.strip()
    records.append((brand, title,shiping_price ))
    
len(records)
records[0:3]


# Applying a tabular data structure

import pandas as pd
df = pd.DataFrame(records, columns=['brand', 'title', 'shiping_price'])
df.head()

# Exporting the dataset to a CSV file


df.to_csv('Newegg.csv', index=False, encoding='utf-8')
